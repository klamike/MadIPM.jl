# ============================================================================
# NormalUniformBatchKKTSystem — batched `A Σ⁻¹ Aᵀ Δy = r` normal equations.
#
# LP-only (asserts `bcb.nnzh == 0`). Back-substitutes `(Δx, Δz)` from `Δy`.
# Shared structural data (`A` sparsity, `AT` CSC, `A_csr_map`, `aug_com`)
# lives once; `A_vals`, `aug_com_nzvals`, the regularization diagonals, and
# the RHS buffers are per-instance matrices.
#
# `j_op` / `jt_op` are `BatchSparseOperator`s so `mul!(y, A, x)` / `jtprod!`
# go through BQM's SpMV without touching the CSC each step.
# ============================================================================

struct NormalUniformBatchKKTSystem{T, LS, MT, VT, VI, VI32, ATC, AUGC, OPT, BVS} <:
        AbstractBatchKKTSystem{T}
    # Jacobian (shared structure, per-instance values)
    A::MadNLP.SparseMatrixCOO{T, Int32, VT, VI32}  # (m, n_tot), V is column 1
    A_vals::MT                                     # (nnzj + n_slack, bs)
    AT::ATC                                        # CSC; CPU or CuSparseMatrixCSC
    A_csr_map::VI
    jac_coo_view::VT                               # view into A.V (non-slack block)
    j_op::OPT                                      # A  (n_tot → m)
    jt_op::OPT                                     # Aᵀ (m → n_tot)

    # Normal matrix (m × m, symmetric)
    aug_com::AUGC                                  # shared CSC (col 1 values)
    aug_com_nzvals::MT                             # (nnz_normal, bs)
    batch_solver::LS

    # Scratch
    rhs_buffer::MT                                 # (n_tot + m, bs)
    r_primal::MT                                   # (n_tot, bs)
    r_dual::MT                                     # (m, bs)

    # Regularization / bounds (per-instance)
    reg::MT
    pr_diag::MT
    du_diag::MT
    l_diag::MT
    u_diag::MT
    l_lower::MT
    u_lower::MT

    batch_size::Int
    batch_views::BVS
    n_tot::Int
    m::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

pr_diag(bkkt::NormalUniformBatchKKTSystem) = bkkt.pr_diag
du_diag(bkkt::NormalUniformBatchKKTSystem) = bkkt.du_diag

MadNLP.is_inertia_correct(::NormalUniformBatchKKTSystem, _npos, nzero, _nneg) = nzero == 0
MadNLP.compress_hessian!(::NormalUniformBatchKKTSystem) = nothing
MadNLP.get_jacobian(bkkt::NormalUniformBatchKKTSystem) = bkkt.jac_coo_view
MadNLP.get_hessian(bkkt::NormalUniformBatchKKTSystem{T}) where {T} = similar(bkkt.pr_diag, 0)

# ---------- build ----------

function MadNLP.create_kkt_system(
    ::Type{NormalKKTSystem},
    bcb::UniformBatchCallback{T, VT, MT, VI},
    uniformbatch_linear_solver = LoopedBatchLinearSolver;
    opt_linear_solver = MadNLP.default_options(uniformbatch_linear_solver),
    batch_views,
) where {T, VT, MT, VI}
    bcb.nnzh == 0 || error("NormalKKTSystem supports LPs only (Hessian must be empty).")

    batch_size = bcb.batch_size
    n, m       = bcb.nvar, bcb.ncon
    ns         = length(bcb.ind_ineq)
    n_tot      = n + ns
    nnzj       = bcb.nnzj

    # ---------- Jacobian with slack column appended ----------
    jac_I = MadNLP.create_array(bcb, Int32, nnzj)
    jac_J = MadNLP.create_array(bcb, Int32, nnzj)
    MadNLP._jac_sparsity_wrapper!(bcb, jac_I, jac_J)

    I = MadNLP.create_array(bcb, Int32, nnzj + ns)
    J = MadNLP.create_array(bcb, Int32, nnzj + ns)
    A_vals = fill!(similar(bcb.con_buffer, nnzj + ns, batch_size), zero(T))
    V = _madnlp_unsafe_column_wrap(A_vals, nnzj + ns, 1, VT)

    I[1:nnzj]          .= jac_I
    J[1:nnzj]          .= jac_J
    I[nnzj+1:end]      .= bcb.ind_ineq
    J[nnzj+1:end]      .= (n+1):(n+ns)
    A_coo = MadNLP.SparseMatrixCOO(m, n_tot, I, J, V)

    # tag values by their original index to recover the COO→CSR permutation
    V .= 1:(nnzj + ns)
    Ap, Aj, Ax = coo_to_csr(A_coo)
    A_csr_map  = convert.(Int, Ax)

    # CSC type picked from VT (CPU vs GPU)
    CSC = sparse_csc_format(VT)
    AT  = CSC <: SparseArrays.SparseMatrixCSC ?
        CSC(n_tot, m, Ap, Aj, Ax) : CSC(Ap, Aj, Ax, (n_tot, m))

    # ---------- normal-matrix sparsity ----------
    AAp, AAj       = build_normal_system(m, n_tot, Ap, Aj)
    nnz_normal     = length(AAj)
    aug_com_nzvals = fill!(similar(A_vals, nnz_normal, batch_size), zero(T))
    AAx_ref        = _madnlp_unsafe_column_wrap(aug_com_nzvals, nnz_normal, 1, VT)
    aug_com = CSC <: SparseArrays.SparseMatrixCSC ?
        CSC(m, m, AAp, AAj, AAx_ref) : CSC(AAp, AAj, AAx_ref, (m, m))

    batch_solver = uniformbatch_linear_solver(
        aug_com, aug_com_nzvals, m; opt = opt_linear_solver,
    )

    # ---------- batched A / Aᵀ SpMV operators ----------
    n_jac_total = nnzj + ns
    nz_id       = similar(A_csr_map, n_jac_total)
    nz_id      .= 1:n_jac_total
    j_op  = _build_batch_op(A_vals, nz_id, J, I, m)
    jt_op = _build_batch_op(A_vals, nz_id, I, J, n_tot)

    fill!(V, zero(T))
    jac_coo_view = MadNLP._madnlp_unsafe_wrap(V, nnzj, 1)

    # ---------- scratch and diagonals ----------
    rhs_buffer = similar(A_vals, n_tot + m, batch_size)
    r_primal   = similar(A_vals, n_tot,     batch_size)
    r_dual     = similar(A_vals, m,         batch_size)

    reg     = similar(A_vals, n_tot, batch_size)
    pr_diag = similar(A_vals, n_tot, batch_size)
    du_diag = similar(A_vals, m,     batch_size)
    nlb, nub = length(bcb.ind_lb), length(bcb.ind_ub)
    l_diag  = similar(A_vals, nlb, batch_size); u_diag  = similar(A_vals, nub, batch_size)
    l_lower = similar(A_vals, nlb, batch_size); u_lower = similar(A_vals, nub, batch_size)

    return NormalUniformBatchKKTSystem{T, typeof(batch_solver), MT, VT, VI, typeof(I),
                                        typeof(AT), typeof(aug_com), typeof(j_op),
                                        typeof(batch_views)}(
        A_coo, A_vals, AT, A_csr_map, jac_coo_view,
        j_op, jt_op,
        aug_com, aug_com_nzvals, batch_solver,
        rhs_buffer, r_primal, r_dual,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        batch_size, batch_views, n_tot, m,
        bcb.ind_ineq, bcb.ind_lb, bcb.ind_ub,
    )
end

# ---------- initialize / value plumbing ----------

function MadNLP.initialize!(bkkt::NormalUniformBatchKKTSystem{T}) where {T}
    fill!(bkkt.reg,            one(T))
    fill!(bkkt.pr_diag,        one(T))
    fill!(bkkt.du_diag,        zero(T))
    fill!(bkkt.l_diag,         one(T));  fill!(bkkt.u_diag,  one(T))
    fill!(bkkt.l_lower,        zero(T)); fill!(bkkt.u_lower, zero(T))
    fill!(bkkt.r_primal,       zero(T))
    fill!(bkkt.r_dual,         zero(T))
    fill!(bkkt.aug_com_nzvals, zero(T))
    return nothing
end

function MadNLP.compress_jacobian!(bkkt::NormalUniformBatchKKTSystem)
    ns = length(bkkt.ind_ineq)
    ns == 0 && return nothing
    nnzj = size(bkkt.A_vals, 1) - ns
    view(bkkt.A_vals, nnzj+1:nnzj+ns, :) .= -one(eltype(bkkt.A_vals))
    return nothing
end

function MadNLP.eval_jac_wrapper!(batch_solver, bkkt::NormalUniformBatchKKTSystem)
    state = batch_solver.state
    bcb   = batch_solver.problem.bcb
    ws    = state.workspace
    nnzj  = bcb.nnzj
    ns    = length(bcb.ind_ineq)

    MadNLP.unpack_x!(ws.bx, bcb, state.x)
    jac_free = MadNLP._eval_jac_wrapper!(bcb, ws.bx, bcb.jac_buffer)
    view(bkkt.A_vals, 1:nnzj, :) .= jac_free
    ns > 0 && (view(bkkt.A_vals, nnzj+1:nnzj+ns, :) .= -one(eltype(bkkt.A_vals)))
    return nothing
end

MadNLP.eval_lag_hess_wrapper!(_bs, ::NormalUniformBatchKKTSystem) = nothing

# ---------- Jᵀ·y / J·x via the batched operators ----------

MadNLP.jtprod!(res::AbstractMatrix{T}, bkkt::NormalUniformBatchKKTSystem{T}, y::AbstractMatrix) where {T} =
    (batch_spmv!(res, bkkt.jt_op, y); res)

MadNLP.jtprod!(res::AbstractMatrix, bkkt::NormalUniformBatchKKTSystem, y::BatchVector) =
    MadNLP.jtprod!(res, bkkt, MadNLP.full(y))

MadNLP.jtprod!(jacl::BatchVector, bkkt::NormalUniformBatchKKTSystem, y) =
    (MadNLP.jtprod!(MadNLP.full(jacl), bkkt, y); jacl)

_batch_mul_A!(y::AbstractMatrix{T}, bkkt::NormalUniformBatchKKTSystem{T},
               x::AbstractMatrix{T}, alpha, beta) where {T} =
    (batch_spmv!(y, bkkt.j_op, x, alpha, beta); y)

# ---------- build + factor + solve ----------

# CPU build of `A Σ⁻¹ Aᵀ` per instance; GPU override in MadIPMCUDAExt.
function MadNLP.build_kkt!(bkkt::NormalUniformBatchKKTSystem{T}) where {T}
    AAp = SparseArrays.getcolptr(bkkt.aug_com); AAj = SparseArrays.rowvals(bkkt.aug_com)
    Ap  = SparseArrays.getcolptr(bkkt.AT);       Ai  = SparseArrays.rowvals(bkkt.AT)
    D   = bkkt.r_primal                          # reused as Σ⁻¹ scratch

    @inbounds for k in axes(bkkt.aug_com_nzvals, 2)
        for i in axes(D, 1); D[i, k] = one(T) / bkkt.pr_diag[i, k]; end
        Av = view(bkkt.A_vals, :, k)

        # Walk the normal matrix column by column.
        for i in 1:bkkt.m
            for p in AAp[i]:(AAp[i+1] - 1)
                j  = AAj[p]
                # (A Σ⁻¹ Aᵀ)[j, i] — merge-join over AT[:, j] and AT[:, i]
                pj, pi_ = Ap[j], Ap[i]
                acc = zero(T)
                while pj < Ap[j+1] && pi_ < Ap[i+1]
                    rj, ri = Ai[pj], Ai[pi_]
                    if rj == ri
                        acc += Av[bkkt.A_csr_map[pj]] * D[rj, k] * Av[bkkt.A_csr_map[pi_]]
                        pj += 1; pi_ += 1
                    elseif rj < ri
                        pj  += 1
                    else
                        pi_ += 1
                    end
                end
                bkkt.aug_com_nzvals[p, k] = acc
            end
        end

        # Subtract dual regularization from the diagonal (du_diag is ≤ 0).
        for i in 1:bkkt.m, p in AAp[i]:(AAp[i+1] - 1)
            AAj[p] == i || continue
            bkkt.aug_com_nzvals[p, k] -= bkkt.du_diag[i, k]
            break
        end
    end
    return nothing
end

function MadNLP.factorize_kkt!(bkkt::NormalUniformBatchKKTSystem)
    factorize_active!(bkkt.batch_solver, active_view(bkkt.batch_views))
    return nothing
end

function MadNLP.solve_kkt!(bkkt::NormalUniformBatchKKTSystem{T}, batch_solver) where {T}
    d      = batch_solver.state.d
    lb_off = d.n + d.m

    _reduce_rhs_batch!(d.values, d.ind_lb, lb_off, bkkt.l_diag,
                                  d.ind_ub, lb_off + d.nlb, bkkt.u_diag)

    bs     = size(d.values, 2)
    wx     = reshape(view(d.values, 1:bkkt.n_tot, :), bkkt.n_tot, bs)
    wy     = reshape(view(d.values, bkkt.n_tot+1:bkkt.n_tot+bkkt.m, :), bkkt.m, bs)

    # r_primal = Σ⁻¹ wx;  r_dual = A·r_primal − wy
    @. bkkt.r_primal = wx / bkkt.pr_diag
    copyto!(bkkt.r_dual, wy)
    _batch_mul_A!(bkkt.r_dual, bkkt, bkkt.r_primal, one(T), -one(T))

    solve_active!(bkkt.batch_solver, bkkt.r_dual, active_view(bkkt.batch_views))
    copyto!(wy, bkkt.r_dual)

    # wx ← (wx − Aᵀ·dy) / Σ. `wy` is a reshape-of-view that CUSPARSE can't
    # multiply against; route Aᵀ through contiguous `r_dual`. `r_primal`'s
    # earlier contents are no longer needed here, so reuse it as scratch and
    # avoid a per-iteration `similar`.
    tmp = bkkt.r_primal
    MadNLP.jtprod!(tmp, bkkt, bkkt.r_dual)
    @. wx = (wx - tmp) / bkkt.pr_diag

    MadNLP.finish_aug_solve!(bkkt, batch_solver)
    return nothing
end

function MadNLP.reduce_rhs!(bkkt::NormalUniformBatchKKTSystem, d::BatchUnreducedKKTVector)
    lb_off = d.n + d.m
    _reduce_rhs_batch!(d.values, d.ind_lb, lb_off, bkkt.l_diag,
                                  d.ind_ub, lb_off + d.nlb, bkkt.u_diag)
    return nothing
end

function MadNLP.finish_aug_solve!(bkkt::NormalUniformBatchKKTSystem, batch_solver)
    d      = batch_solver.state.d
    lb_off = d.n + d.m
    _finish_aug_solve_batch!(d.values, d.ind_lb, lb_off, bkkt.l_lower, bkkt.l_diag,
                                        d.ind_ub, lb_off + d.nlb, bkkt.u_lower, bkkt.u_diag)
    return nothing
end

# ---------- full KKT multiply ----------

function LinearAlgebra.mul!(
    w::BatchUnreducedKKTVector{T}, bkkt::NormalUniformBatchKKTSystem{T},
    x::BatchUnreducedKKTVector{T}, alpha = one(T), beta = zero(T),
) where {T}
    xp, xd = MadNLP.primal(x), MadNLP.dual(x)
    wp, wd = MadNLP.primal(w), MadNLP.dual(w)

    # wp ← α (Σ xp + Aᵀ xd) + β wp. `mul!` runs in IR between `solve_kkt!`
    # calls; solve_kkt! always rewrites `r_primal` first, so using it as
    # scratch here is safe and saves a per-call allocation.
    scratch = bkkt.r_primal
    MadNLP.jtprod!(scratch, bkkt, xd)
    if iszero(beta)
        @. wp = alpha * (bkkt.pr_diag * xp + scratch)
    else
        @. wp = beta * wp + alpha * (bkkt.pr_diag * xp + scratch)
    end

    # wd ← α (A xp − δ xd) + β wd
    _batch_mul_A!(wd, bkkt, xp, alpha, beta)
    @. wd -= alpha * bkkt.du_diag * xd

    _kktmul!(w, x, bkkt.reg, bkkt.du_diag,
             bkkt.l_lower, bkkt.u_lower, bkkt.l_diag, bkkt.u_diag,
             alpha, beta)
    return w
end
