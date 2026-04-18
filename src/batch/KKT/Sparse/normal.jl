# Batched variant of `NormalKKTSystem`. Mirrors `SparseUniformBatchKKTSystem`'s
# layout and data flow, but builds `A Σ⁻¹ Aᵀ` (m × m, PSD) per instance and
# factorizes it via `LoopedBatchLinearSolver` on shared CSC structure.
#
# All instances share: Jacobian sparsity (A, AT), normal-matrix structure
# (AAp, AAj), A_csr_map. Per-instance: jac values, pr_diag/du_diag, bound
# multipliers, normal-matrix nzvals, RHS buffers.

struct NormalUniformBatchKKTSystem{T, LS, MT, VT, VI, VI32, ATC, AUGC, BVS} <: AbstractBatchKKTSystem{T}
    # Jacobian (shared structure, per-instance values)
    A::MadNLP.SparseMatrixCOO{T, Int32, VT, VI32}  # (m, n_tot) with slack column; V is col 1
    A_vals::MT                                     # (nnzj + n_slack, bs) per-instance
    AT::ATC                                        # shared structure (CSC; CPU or CuSparseMatrixCSC)
    A_csr_map::VI
    jac_coo_view::VT                               # view into A.V for the non-slack block
    # Normal matrix: shared structure, per-instance values
    aug_com::AUGC                                  # m × m CSC (col 1 values used for symbolic factor)
    aug_com_nzvals::MT                             # (nnz_normal, bs)
    # Linear solver (looped over batch columns of aug_com_nzvals)
    batch_solver::LS
    # Buffers
    rhs_buffer::MT                                 # ((n_tot + m), bs) for full KKT solves
    r_primal::MT                                   # (n_tot, bs)
    r_dual::MT                                     # (m, bs)
    # Bound-related diagonals (per-instance)
    reg::MT                                        # (n_tot, bs)
    pr_diag::MT                                    # (n_tot, bs)
    du_diag::MT                                    # (m, bs)
    l_diag::MT
    u_diag::MT                                     # (nub, bs); empty for std-form
    l_lower::MT
    u_lower::MT
    # Metadata
    batch_size::Int
    batch_views::BVS
    n_tot::Int
    m::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

# Build the batched NormalKKTSystem from a batched callback.
function MadNLP.create_kkt_system(
    ::Type{NormalKKTSystem},
    bcb::UniformBatchCallback{T, VT, MT, VI},
    uniformbatch_linear_solver = LoopedBatchLinearSolver;
    opt_linear_solver = MadNLP.default_options(uniformbatch_linear_solver),
    batch_views,
) where {T, VT, MT, VI}
    bcb.nnzh == 0 || error("NormalKKTSystem supports only LPs (Hessian must be empty).")

    batch_size = bcb.batch_size
    n_slack = length(bcb.ind_ineq)
    n = bcb.nvar
    m = bcb.ncon
    n_tot = n + n_slack
    nnzj = bcb.nnzj

    jac_sparsity_I = MadNLP.create_array(bcb, Int32, nnzj)
    jac_sparsity_J = MadNLP.create_array(bcb, Int32, nnzj)
    MadNLP._jac_sparsity_wrapper!(bcb, jac_sparsity_I, jac_sparsity_J)

    # Assemble COO Jacobian with slack column appended.
    I = MadNLP.create_array(bcb, Int32, nnzj + n_slack)
    J = MadNLP.create_array(bcb, Int32, nnzj + n_slack)
    A_vals = similar(bcb.con_buffer, nnzj + n_slack, batch_size)
    fill!(A_vals, zero(T))
    V = _madnlp_unsafe_column_wrap(A_vals, nnzj + n_slack, 1, VT)
    I[1:nnzj] .= jac_sparsity_I
    J[1:nnzj] .= jac_sparsity_J
    I[nnzj + 1 : nnzj + n_slack] .= bcb.ind_ineq
    J[nnzj + 1 : nnzj + n_slack] .= (n + 1 : n + n_slack)
    A_coo = MadNLP.SparseMatrixCOO(m, n_tot, I, J, V)

    # Build A_csr_map by filling V with a continuous range and converting to CSR.
    V .= 1:(nnzj + n_slack)
    Ap, Aj, Ax = coo_to_csr(A_coo)
    A_csr_map = convert.(Int, Ax)

    # CSC type: SparseMatrixCSC on CPU, CuSparseMatrixCSC on GPU. Picked from VT.
    CSC = sparse_csc_format(VT)
    AT = CSC <: SparseArrays.SparseMatrixCSC ?
        CSC(n_tot, m, Ap, Aj, Ax) :
        CSC(Ap, Aj, Ax, (n_tot, m))

    # Normal-matrix sparsity: (AAp, AAj) structure shared, per-instance nzvals.
    AAp, AAj = build_normal_system(m, n_tot, Ap, Aj)
    nnz_normal = length(AAj)
    aug_com_nzvals = similar(A_vals, nnz_normal, batch_size)
    fill!(aug_com_nzvals, zero(T))
    AAx_ref = _madnlp_unsafe_column_wrap(aug_com_nzvals, nnz_normal, 1, VT)
    aug_com = CSC <: SparseArrays.SparseMatrixCSC ?
        CSC(m, m, AAp, AAj, AAx_ref) :
        CSC(AAp, AAj, AAx_ref, (m, m))

    # Looped batch linear solver over aug_com_nzvals.
    batch_solver = uniformbatch_linear_solver(aug_com, aug_com_nzvals, m; opt = opt_linear_solver)

    # Reset values and carve out a view to the non-slack Jacobian block.
    fill!(V, zero(T))
    jac_coo_view = MadNLP._madnlp_unsafe_wrap(V, nnzj, 1)

    rhs_buffer = similar(A_vals, n_tot + m, batch_size)
    r_primal = similar(A_vals, n_tot, batch_size)
    r_dual = similar(A_vals, m, batch_size)

    reg     = similar(A_vals, n_tot, batch_size)
    pr_diag = similar(A_vals, n_tot, batch_size)
    du_diag = similar(A_vals, m, batch_size)
    nlb = length(bcb.ind_lb)
    nub = length(bcb.ind_ub)
    l_diag  = similar(A_vals, nlb, batch_size)
    u_diag  = similar(A_vals, nub, batch_size)
    l_lower = similar(A_vals, nlb, batch_size)
    u_lower = similar(A_vals, nub, batch_size)

    return NormalUniformBatchKKTSystem{T, typeof(batch_solver), MT, VT, VI, typeof(I), typeof(AT), typeof(aug_com), typeof(batch_views)}(
        A_coo, A_vals, AT, A_csr_map, jac_coo_view,
        aug_com, aug_com_nzvals, batch_solver,
        rhs_buffer, r_primal, r_dual,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        batch_size, batch_views, n_tot, m,
        bcb.ind_ineq, bcb.ind_lb, bcb.ind_ub,
    )
end

# `pr_diag`/`du_diag` accessors expected by batch IPM kernels.
pr_diag(bkkt::NormalUniformBatchKKTSystem) = bkkt.pr_diag
du_diag(bkkt::NormalUniformBatchKKTSystem) = bkkt.du_diag

MadNLP.is_inertia_correct(::NormalUniformBatchKKTSystem, num_pos, num_zero, num_neg) =
    num_zero == 0  # normal matrix is PSD; we trust per-column factorizations

function MadNLP.initialize!(bkkt::NormalUniformBatchKKTSystem{T}) where {T}
    fill!(bkkt.reg, one(T))
    fill!(bkkt.pr_diag, one(T))
    fill!(bkkt.du_diag, zero(T))
    fill!(bkkt.u_lower, zero(T))
    fill!(bkkt.u_diag, one(T))
    fill!(bkkt.l_lower, zero(T))
    fill!(bkkt.l_diag, one(T))
    fill!(bkkt.r_dual, zero(T))
    fill!(bkkt.r_primal, zero(T))
    fill!(bkkt.aug_com_nzvals, zero(T))
    return
end

# Fill the slack column (-1) on eval. The non-slack block is filled by
# `eval_jac_wrapper!` below.
function MadNLP.compress_jacobian!(bkkt::NormalUniformBatchKKTSystem)
    n_slack = length(bkkt.ind_ineq)
    n_slack == 0 && return
    nnzj = size(bkkt.A_vals, 1) - n_slack
    view(bkkt.A_vals, nnzj + 1 : nnzj + n_slack, :) .= -one(eltype(bkkt.A_vals))
    return
end

MadNLP.compress_hessian!(::NormalUniformBatchKKTSystem) = nothing
MadNLP.get_jacobian(bkkt::NormalUniformBatchKKTSystem) = bkkt.jac_coo_view
MadNLP.get_hessian(bkkt::NormalUniformBatchKKTSystem{T}) where {T} = similar(bkkt.pr_diag, 0)

# Fill Jacobian values from the batched callback.
function MadNLP.eval_jac_wrapper!(
    batch_solver::AbstractBatchMPCSolver,
    bkkt::NormalUniformBatchKKTSystem,
)
    state = batch_solver.state
    bcb = batch_solver.problem.bcb
    ws = state.workspace
    nnzj = bcb.nnzj
    n_slack = length(bcb.ind_ineq)

    MadNLP.unpack_x!(ws.bx, bcb, state.x)
    jac_free = MadNLP._eval_jac_wrapper!(bcb, ws.bx, bcb.jac_buffer)
    view(bkkt.A_vals, 1:nnzj, :) .= jac_free
    if n_slack > 0
        view(bkkt.A_vals, nnzj + 1 : nnzj + n_slack, :) .= -one(eltype(bkkt.A_vals))
    end
    return
end

# No Hessian in LPs.
MadNLP.eval_lag_hess_wrapper!(::AbstractBatchMPCSolver, ::NormalUniformBatchKKTSystem) = nothing

# `jtprod!` — compute Aᵀ y (or slack-aware Aᵀ) in batch space by looping
# since A has shared structure but per-instance values.
function MadNLP.jtprod!(res::AbstractMatrix{T}, bkkt::NormalUniformBatchKKTSystem{T}, y) where {T}
    yfull = y isa BatchVector ? MadNLP.full(y) : y
    fill!(res, zero(T))
    n_tot = bkkt.n_tot
    # AT has shared structure; apply each instance's Jacobian values via A_csr_map
    # onto res column by column. A_csr_map[i] maps AT nzval index i → A.V index.
    Ap = SparseArrays.getcolptr(bkkt.AT)
    Ai = SparseArrays.rowvals(bkkt.AT)
    @inbounds for k in axes(res, 2)
        for j in 1:bkkt.m
            yjk = yfull[j, k]
            iszero(yjk) && continue
            for p in Ap[j]:(Ap[j + 1] - 1)
                v = bkkt.A_vals[bkkt.A_csr_map[p], k]
                res[Ai[p], k] += v * yjk
            end
        end
    end
    return res
end

MadNLP.jtprod!(jacl::BatchVector, bkkt::NormalUniformBatchKKTSystem, y) =
    (MadNLP.jtprod!(MadNLP.full(jacl), bkkt, y); jacl)

# Build `A Σ⁻¹ Aᵀ` per instance into `aug_com_nzvals`.
function MadNLP.build_kkt!(bkkt::NormalUniformBatchKKTSystem{T}) where {T}
    AAp = SparseArrays.getcolptr(bkkt.aug_com)
    AAj = SparseArrays.rowvals(bkkt.aug_com)
    Ap = SparseArrays.getcolptr(bkkt.AT)
    Ai = SparseArrays.rowvals(bkkt.AT)
    D = bkkt.r_primal  # reuse as scratch
    @inbounds for k in axes(bkkt.aug_com_nzvals, 2)
        for i in axes(D, 1)
            D[i, k] = one(T) / bkkt.pr_diag[i, k]
        end
        Av = view(bkkt.A_vals, :, k)
        # Loop over target rows of normal matrix (columns of aug_com CSC).
        for i in 1:bkkt.m
            for p in AAp[i]:(AAp[i + 1] - 1)
                j = AAj[p]  # row of aug_com[p, col i] — but since CSC symmetric, row is j
                # Compute (A Σ⁻¹ Aᵀ)[j, i] = sum_k A[j, k] * D[k] * A[i, k].
                acc = zero(T)
                # Walk AT[:, j] and AT[:, i] intersection via their sorted rowvals.
                pj = Ap[j]
                pi_ = Ap[i]
                while pj < Ap[j + 1] && pi_ < Ap[i + 1]
                    rj = Ai[pj]
                    ri = Ai[pi_]
                    if rj == ri
                        acc += Av[bkkt.A_csr_map[pj]] * D[rj, k] * Av[bkkt.A_csr_map[pi_]]
                        pj += 1; pi_ += 1
                    elseif rj < ri
                        pj += 1
                    else
                        pi_ += 1
                    end
                end
                bkkt.aug_com_nzvals[p, k] = acc
            end
        end
        # Add per-instance dual regularization (du_diag is -δ so subtract from diag).
        # aug_com structure is lower triangular / full? coo_to_csr + build_normal_system
        # keeps it symmetric-full. We add du_diag to each diagonal entry.
        for i in 1:bkkt.m
            # Find diagonal entry (j == i) in column i.
            for p in AAp[i]:(AAp[i + 1] - 1)
                if AAj[p] == i
                    bkkt.aug_com_nzvals[p, k] -= bkkt.du_diag[i, k]
                    break
                end
            end
        end
    end
    return
end

function MadNLP.factorize_kkt!(bkkt::NormalUniformBatchKKTSystem)
    factor_view = active_view(bkkt.batch_views)
    factorize_active!(bkkt.batch_solver, factor_view)
    return
end

# Per-column solve of the reduced dual system, then reconstruct primal.
function MadNLP.solve_kkt!(bkkt::NormalUniformBatchKKTSystem{T}, batch_solver::AbstractBatchMPCSolver) where {T}
    d = batch_solver.state.d
    # Reduce lb/ub rows of the unreduced vector into the primal block.
    lb_off = d.n + d.m
    _reduce_rhs_batch!(d.values, d.ind_lb, lb_off, bkkt.l_diag, d.ind_ub, lb_off + d.nlb, bkkt.u_diag)

    wx_mat = reshape(view(d.values, 1:bkkt.n_tot, :), bkkt.n_tot, size(d.values, 2))
    wy_mat = reshape(view(d.values, bkkt.n_tot + 1 : bkkt.n_tot + bkkt.m, :), bkkt.m, size(d.values, 2))

    # r1 = wx / Σ  (per-column broadcast).
    @. bkkt.r_primal = wx_mat / bkkt.pr_diag
    # r2 = wy
    copyto!(bkkt.r_dual, wy_mat)
    # r2 = A r1 - r2 via jtprod! with transposed signature; we need (A) * r_primal
    # but jtprod computes Aᵀ. Build A * r_primal by looping.
    _batch_mul_A!(bkkt.r_dual, bkkt, bkkt.r_primal, one(T), -one(T))

    # Solve per-column: aug_com dy = r_dual.
    solve_active!(bkkt.batch_solver, bkkt.r_dual, active_view(bkkt.batch_views))

    # dy back into wy
    copyto!(wy_mat, bkkt.r_dual)

    # wx = (wx - Aᵀ dy) / Σ. `wy_mat` is a `reshape(view(d.values, ...))`,
    # whose per-column views CUSPARSE can't multiply against; route through
    # the contiguous `bkkt.r_dual` (which already holds `dy` after the
    # solve / copyto! pair above) so the GPU jtprod sees a real CuMatrix.
    tmp = similar(bkkt.r_primal)
    MadNLP.jtprod!(tmp, bkkt, bkkt.r_dual)
    @. wx_mat = (wx_mat - tmp) / bkkt.pr_diag

    MadNLP.finish_aug_solve!(bkkt, batch_solver)
    return
end

# y += alpha * (A * x + beta * y)... here we implement `y <- alpha * A*x + beta * y`
# per column. Shared AT structure, per-instance values.
function _batch_mul_A!(y::AbstractMatrix{T}, bkkt::NormalUniformBatchKKTSystem{T}, x::AbstractMatrix{T}, alpha, beta) where {T}
    if beta != one(T)
        if iszero(beta)
            fill!(y, zero(T))
        else
            @. y *= beta
        end
    end
    Ap = SparseArrays.getcolptr(bkkt.AT)
    Ai = SparseArrays.rowvals(bkkt.AT)
    @inbounds for k in axes(y, 2)
        for j in 1:bkkt.m
            acc = zero(T)
            for p in Ap[j]:(Ap[j + 1] - 1)
                acc += bkkt.A_vals[bkkt.A_csr_map[p], k] * x[Ai[p], k]
            end
            y[j, k] += alpha * acc
        end
    end
    return y
end

function MadNLP.reduce_rhs!(bkkt::NormalUniformBatchKKTSystem, d::BatchUnreducedKKTVector)
    lb_off = d.n + d.m
    _reduce_rhs_batch!(d.values, d.ind_lb, lb_off, bkkt.l_diag, d.ind_ub, lb_off + d.nlb, bkkt.u_diag)
    return
end

function MadNLP.finish_aug_solve!(bkkt::NormalUniformBatchKKTSystem, batch_solver::AbstractBatchMPCSolver)
    d = batch_solver.state.d
    lb_off = d.n + d.m
    _finish_aug_solve_batch!(d.values, d.ind_lb, lb_off, bkkt.l_lower, bkkt.l_diag, d.ind_ub, lb_off + d.nlb, bkkt.u_lower, bkkt.u_diag)
    return
end

function LinearAlgebra.mul!(
    w::BatchUnreducedKKTVector{T},
    bkkt::NormalUniformBatchKKTSystem{T},
    x::BatchUnreducedKKTVector{T},
    alpha = one(T),
    beta = zero(T),
) where {T}
    # Full KKT matrix: [Σ  Aᵀ; A  -δI]. We implement via jtprod / mul_A.
    xp = MadNLP.primal(x)
    xd = MadNLP.dual(x)
    wp = MadNLP.primal(w)
    wd = MadNLP.dual(w)

    # wp = α (Σ xp + Aᵀ xd) + β wp
    scratch = similar(wp)
    MadNLP.jtprod!(scratch, bkkt, xd)   # Aᵀ xd
    if iszero(beta)
        @. wp = alpha * (bkkt.pr_diag * xp + scratch)
    else
        @. wp = beta * wp + alpha * (bkkt.pr_diag * xp + scratch)
    end
    # wd = α (A xp - δ xd) + β wd
    _batch_mul_A!(wd, bkkt, xp, alpha, beta)
    @. wd -= alpha * bkkt.du_diag * xd

    _kktmul!(w, x, bkkt.reg, bkkt.du_diag, bkkt.l_lower, bkkt.u_lower, bkkt.l_diag, bkkt.u_diag, alpha, beta)
    return w
end
