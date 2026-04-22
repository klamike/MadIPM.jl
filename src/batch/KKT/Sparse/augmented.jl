# ============================================================================
# SparseUniformBatchKKTSystem — batched analogue of MadNLP's `SparseKKTSystem`.
#
# The full augmented system `[H + Σ   Aᵀ; A  -Δ]` is stored as a single COO
# triple `(aug_I, aug_J, nzVals)` with `aug_I` / `aug_J` shared across the
# batch and `nzVals` per-instance `(aug_mat_length × batch_size)`. The CSC
# form `aug_com_nzvals` is derived from `nzVals` via the precomputed
# `batch_csc_map` — no symbolic rebuild per iteration.
#
#   batch_solver      — per-instance factorization driver (default
#                       `LoopedBatchLinearSolver`; GPU ext swaps in CUDSS).
#   hess_op / jt_op / j_op
#                     — `BatchSparseOperator`s backing MadNLP's `mul!` /
#                       `jtprod!` without reconstructing a sparse matrix.
# ============================================================================

struct SparseUniformBatchKKTSystem{T, LS, MT, VI, VI32, OPT, BVS} <: AbstractBatchKKTSystem{T}
    nzVals::MT              # (aug_mat_length × bs)  COO nonzero values
    aug_I::VI32             # shared row indices
    aug_J::VI32             # shared column indices
    batch_solver::LS
    rhs_buffer::MT          # (n + m) × bs for batch solve
    compact_buffer::MT      # scratch for GPU compact (max(nnz_csc, n+m) × bs)
    batch_size::Int
    batch_views::BVS
    aug_com_nzvals::MT      # (nnz_csc × bs) CSC nonzero values
    batch_csc_map::VI       # flattened COO → CSC map (all instances)
    n_tot::Int              # n + n_slack
    m::Int
    nnzh::Int

    # Diagonal / bound data (consumed by `_kktmul!`)
    reg::MT                 # (n_tot × bs)
    l_diag::MT              # (nlb × bs)
    u_diag::MT              # (nub × bs)
    l_lower::MT             # (nlb × bs)
    u_lower::MT             # (nub × bs)

    # Batch SpMV operators
    hess_op::OPT
    jt_op::OPT
    j_op::OPT
end

pr_diag(bkkt::SparseUniformBatchKKTSystem) = view(bkkt.nzVals, 1:bkkt.n_tot, :)
function du_diag(bkkt::SparseUniformBatchKKTSystem)
    du_off = size(bkkt.nzVals, 1) - bkkt.m
    return view(bkkt.nzVals, du_off+1:du_off+bkkt.m, :)
end

# ---------- build ----------

function MadNLP.create_kkt_system(
    ::Type{MadNLP.SparseKKTSystem},
    bcb::UniformBatchCallback{T, VT, MT, VI},
    uniformbatch_linear_solver = LoopedBatchLinearSolver;
    opt_linear_solver = MadNLP.default_options(uniformbatch_linear_solver),
    batch_views,
) where {T, VT, MT, VI}
    batch_size = bcb.batch_size
    n, m       = bcb.nvar, bcb.ncon
    n_slack    = length(bcb.ind_ineq)
    n_tot      = n + n_slack
    nlb, nub   = length(bcb.ind_lb), length(bcb.ind_ub)

    # ---------- sparsity ----------
    jac_I = MadNLP.create_array(bcb, Int32, bcb.nnzj)
    jac_J = MadNLP.create_array(bcb, Int32, bcb.nnzj)
    MadNLP._jac_sparsity_wrapper!(bcb, jac_I, jac_J)
    hess_I, hess_J = MadNLP.build_hessian_structure(bcb, MadNLP.ExactHessian)
    MadNLP.force_lower_triangular!(hess_I, hess_J)
    n_jac, n_hess = length(jac_I), length(hess_I)

    aug_vec_length = n_tot + m
    aug_mat_length = n_tot + m + n_hess + n_jac + n_slack

    I = MadNLP.create_array(bcb, Int32, aug_mat_length)
    J = MadNLP.create_array(bcb, Int32, aug_mat_length)
    nzVals = similar(bcb.con_buffer, aug_mat_length, batch_size)
    fill!(nzVals, zero(T))
    V = _madnlp_unsafe_column_wrap(nzVals, aug_mat_length, 1, VT)

    # layout: [primal diag | hess | jac | slack cols | dual diag]
    a, b, c, d = n_tot, n_tot + n_hess, n_tot + n_hess + n_jac,
                 n_tot + n_hess + n_jac + n_slack
    I[1:a]       .= 1:n_tot
    I[a+1:b]       = hess_I
    I[b+1:c]     .= jac_I .+ n_tot
    I[c+1:d]     .= bcb.ind_ineq .+ n_tot
    I[d+1:d+m]   .= (n_tot+1):(n_tot+m)
    J[1:a]       .= 1:n_tot
    J[a+1:b]       = hess_J
    J[b+1:c]     .= jac_J
    J[c+1:d]     .= (n+1):(n+n_slack)
    J[d+1:d+m]   .= (n_tot+1):(n_tot+m)

    aug_raw = MadNLP.SparseMatrixCOO(aug_vec_length, aug_vec_length, I, J, V)
    aug_com, aug_csc_map = MadNLP.coo_to_csc(aug_raw)

    nnz_csc        = SparseArrays.nnz(aug_com)
    aug_com_nzvals = fill!(similar(nzVals, nnz_csc, batch_size), zero(T))
    csc_offsets    = similar(aug_csc_map, 1, batch_size)
    csc_offsets   .= (0:batch_size-1)' .* nnz_csc
    batch_csc_map  = vec(aug_csc_map .+ csc_offsets)

    batch_ls = uniformbatch_linear_solver(aug_com, aug_com_nzvals, aug_vec_length;
                                          opt = opt_linear_solver)

    rhs_buffer     = fill!(similar(nzVals, aug_vec_length, batch_size), zero(T))
    compact_buffer = similar(nzVals, max(nnz_csc, aug_vec_length), batch_size)

    jac_range = (n_tot+n_hess+1):(n_tot+n_hess+n_jac+n_slack)
    hess_op   = _build_hess_op(I, J, n_tot, n_hess,   nzVals, aug_csc_map)
    jt_op     = _build_jt_op(  I, J, jac_range, n_tot,        nzVals, aug_csc_map)
    j_op      = _build_j_op(   I, J, jac_range, n_tot, m,     nzVals, aug_csc_map)

    reg     = similar(nzVals, n_tot, batch_size)
    l_diag  = similar(nzVals, nlb,   batch_size)
    u_diag  = similar(nzVals, nub,   batch_size)
    l_lower = similar(nzVals, nlb,   batch_size)
    u_lower = similar(nzVals, nub,   batch_size)

    return SparseUniformBatchKKTSystem{T, typeof(batch_ls), MT, VI, typeof(I),
                                        typeof(jt_op), typeof(batch_views)}(
        nzVals, I, J, batch_ls, rhs_buffer, compact_buffer, batch_size, batch_views,
        aug_com_nzvals, batch_csc_map, n_tot, m, n_hess,
        reg, l_diag, u_diag, l_lower, u_lower,
        hess_op, jt_op, j_op,
    )
end

# ---------- factor / solve ----------

function MadNLP.factorize_kkt!(bkkt::SparseUniformBatchKKTSystem)
    fv = active_view(bkkt.batch_views)
    is_identity_view(fv) ||
        compact_active_columns_inplace!(bkkt.aug_com_nzvals, fv, bkkt.compact_buffer)
    factorize_active!(bkkt.batch_solver, fv)
    return nothing
end

function MadNLP.solve_linear_system!(bkkt::SparseUniformBatchKKTSystem, rhs::AbstractMatrix)
    av = active_view(bkkt.batch_views)
    is_identity_view(av) ||
        compact_active_columns_inplace!(rhs, av, bkkt.compact_buffer)
    solve_active!(bkkt.batch_solver, rhs, av)
    return rhs
end

# ---------- bound-dual RHS reduction / unwind ----------

function _reduce_rhs_batch!(values, ind_lb, lb_off, l_diag, ind_ub, ub_off, u_diag)
    length(ind_lb) > 0 && (view(values, ind_lb, :) .-=
        view(values, lb_off+1:lb_off+length(ind_lb), :) ./ l_diag)
    length(ind_ub) > 0 && (view(values, ind_ub, :) .-=
        view(values, ub_off+1:ub_off+length(ind_ub), :) ./ u_diag)
    return nothing
end

function _finish_aug_solve_batch!(values, ind_lb, lb_off, l_lower, l_diag,
                                          ind_ub, ub_off, u_lower, u_diag)
    if length(ind_lb) > 0
        lb = view(values, lb_off+1:lb_off+length(ind_lb), :)
        x  = view(values, ind_lb, :)
        @. lb = (-lb + l_lower * x) / l_diag
    end
    if length(ind_ub) > 0
        ub = view(values, ub_off+1:ub_off+length(ind_ub), :)
        x  = view(values, ind_ub, :)
        @. ub = (ub - u_lower * x) / u_diag
    end
    return nothing
end

function MadNLP.reduce_rhs!(bkkt::SparseUniformBatchKKTSystem, d::BatchUnreducedKKTVector)
    lb_off = d.n + d.m
    _reduce_rhs_batch!(d.values, d.ind_lb, lb_off, bkkt.l_diag,
                                  d.ind_ub, lb_off + d.nlb, bkkt.u_diag)
    return nothing
end

function MadNLP.finish_aug_solve!(bkkt::SparseUniformBatchKKTSystem, batch_solver)
    d      = batch_solver.state.d
    lb_off = d.n + d.m
    _finish_aug_solve_batch!(d.values, d.ind_lb, lb_off, bkkt.l_lower, bkkt.l_diag,
                                        d.ind_ub, lb_off + d.nlb, bkkt.u_lower, bkkt.u_diag)
    return nothing
end

function MadNLP.solve_kkt!(bkkt::SparseUniformBatchKKTSystem, batch_solver)
    # Extension path: the batch_solver object is either a `UniformBatchMPCSolver`
    # (reach into its `state.d`) or already a `BatchUnreducedKKTVector` (pass
    # through directly) — both are forwarded to the per-`d` method below.
    d = batch_solver isa BatchUnreducedKKTVector ? batch_solver : batch_solver.state.d
    return MadNLP.solve_kkt!(bkkt, d)
end

function MadNLP.solve_kkt!(bkkt::SparseUniformBatchKKTSystem, d::BatchUnreducedKKTVector)
    rhs = bkkt.rhs_buffer
    pd  = MadNLP.primal_dual(d)
    av  = active_view(bkkt.batch_views)

    MadNLP.reduce_rhs!(bkkt, d)
    copyto!(rhs, pd)
    MadNLP.solve_linear_system!(bkkt, rhs)
    is_identity_view(av) ? copyto!(pd, rhs) :
                           scatter_batch_view_columns!(pd, rhs, av)

    _finish_aug_solve_batch!(
        d.values, d.ind_lb, d.n + d.m, bkkt.l_lower, bkkt.l_diag,
                  d.ind_ub, d.n + d.m + d.nlb, bkkt.u_lower, bkkt.u_diag,
    )
    return nothing
end

# ---------- COO → CSC scatter / eval wrappers ----------

function MadNLP.build_kkt!(bkkt::SparseUniformBatchKKTSystem)
    _scatter_to_csc!(vec(bkkt.aug_com_nzvals), vec(bkkt.nzVals), bkkt.batch_csc_map)
    return nothing
end

# CPU; GPU override in MadIPMCUDAExt.
function _scatter_to_csc!(dest, src, map)
    fill!(dest, zero(eltype(dest)))
    @inbounds for i in eachindex(map)
        dest[map[i]] += src[i]
    end
    return nothing
end

MadNLP.jtprod!(res::AbstractMatrix, bkkt::SparseUniformBatchKKTSystem, y::BatchVector) =
    (batch_spmv!(res, bkkt.jt_op, MadNLP.full(y)); res)

MadNLP.jtprod!(jacl::BatchVector, bkkt::SparseUniformBatchKKTSystem, y::BatchVector) =
    MadNLP.jtprod!(MadNLP.full(jacl), bkkt, y)

function MadNLP.eval_jac_wrapper!(batch_solver, bkkt::SparseUniformBatchKKTSystem)
    state   = batch_solver.state
    bcb     = batch_solver.problem.bcb
    ws      = state.workspace
    nzVals  = bkkt.nzVals
    n_slack = length(bcb.ind_ineq)
    jac_off = bkkt.n_tot + bkkt.nnzh

    MadNLP.unpack_x!(ws.bx, bcb, state.x)
    jac_free = MadNLP._eval_jac_wrapper!(bcb, ws.bx, bcb.jac_buffer)
    view(nzVals, jac_off+1:jac_off+bcb.nnzj, :) .= jac_free
    n_slack > 0 && (view(nzVals, jac_off+bcb.nnzj+1:jac_off+bcb.nnzj+n_slack, :) .=
                     -one(eltype(nzVals)))
    return nothing
end

function MadNLP.eval_lag_hess_wrapper!(batch_solver, bkkt::SparseUniformBatchKKTSystem)
    bkkt.nnzh > 0 || return nothing

    problem = batch_solver.problem
    state   = batch_solver.state
    bcb     = problem.bcb
    ws      = state.workspace
    hess    = view(bkkt.nzVals, bkkt.n_tot+1:bkkt.n_tot+bkkt.nnzh, :)

    MadNLP.unpack_x!(ws.bx, bcb, state.x)
    bf_mat = reshape(ws.bf, 1, problem.batch_size)
    @. bf_mat = bcb.obj_sign * bcb.obj_scale
    MadNLP._eval_lag_hess_wrapper!(
        bcb, ws.bx, MadNLP.full(state.y), ws.bv, hess;
        obj_weight = ws.bf,
    )
    return nothing
end

function MadNLP.initialize!(bkkt::SparseUniformBatchKKTSystem{T}) where {T}
    pr_diag(bkkt) .= one(T)
    if bkkt.nnzh > 0
        view(bkkt.nzVals, bkkt.n_tot+1:bkkt.n_tot+bkkt.nnzh, :) .= zero(T)
    end
    du_diag(bkkt) .= zero(T)

    fill!(bkkt.reg,            zero(T))
    fill!(bkkt.l_diag,         one(T))
    fill!(bkkt.u_diag,         one(T))
    fill!(bkkt.l_lower,        zero(T))
    fill!(bkkt.u_lower,        zero(T))
    fill!(bkkt.aug_com_nzvals, zero(T))
    return nothing
end

# ---------- K · x (batch) ----------

function LinearAlgebra.mul!(
    w::BatchUnreducedKKTVector{T}, bkkt::SparseUniformBatchKKTSystem{T},
    x::BatchUnreducedKKTVector{T}, alpha = one(T), beta = zero(T),
) where {T}
    xv = MadNLP.full(x)
    batch_spmv!(MadNLP.primal(w), bkkt.hess_op, xv, alpha, beta)
    batch_spmv!(MadNLP.primal(w), bkkt.jt_op,   xv, alpha, one(T); val_offset = bkkt.n_tot)
    batch_spmv!(MadNLP.dual(w),   bkkt.j_op,    xv, alpha, beta)
    _kktmul!(w, x, bkkt.reg, du_diag(bkkt),
             bkkt.l_lower, bkkt.u_lower, bkkt.l_diag, bkkt.u_diag,
             alpha, beta)
    return w
end
