# ============================================================================
# ScaledSparseUniformBatchKKTSystem — K2.5 batched analogue of MadNLP's
# `ScaledSparseKKTSystem` (see MadNLP/KKT/Sparse/scaled_augmented.jl).
#
# K2.5 pre-multiplies the primal rows/columns of the unreduced augmented
# system by `√(X_L ⊙ X_U)`; the resulting factor has bounded condition number
# as μ → 0, which matters a lot for CUDSS-LDL on bound-constrained LPs that
# would otherwise silently produce a bad direction near the central path.
# The batch variant lays the `scaling_factor` and K2.5 buffers out per-column
# ((n_tot × bs) and (aug_vec_length × bs)) and applies per-column transforms
# in build_kkt! / solve_kkt!.
# ============================================================================

struct ScaledSparseUniformBatchKKTSystem{T, LS, MT, VI, VI32, OPT, BVS} <: AbstractBatchKKTSystem{T}
    nzVals::MT              # (aug_mat_length × bs) unscaled COO values
    aug_I::VI32             # shared row indices
    aug_J::VI32             # shared column indices
    batch_solver::LS
    rhs_buffer::MT          # (aug_vec_length × bs) scratch
    compact_buffer::MT      # ≥ max(nnz_csc, aug_vec_length) × bs
    batch_size::Int
    batch_views::BVS
    aug_com_nzvals::MT      # (nnz_csc × bs) scaled CSC values
    scaled_nzVals::MT       # (aug_mat_length × bs) scaled COO values (scratch)
    batch_csc_map::VI
    n_tot::Int
    m::Int
    nnzh::Int

    reg::MT                 # (n_tot × bs)
    l_diag::MT              # (nlb × bs)
    u_diag::MT              # (nub × bs)
    l_lower::MT
    u_lower::MT
    scaling_factor::MT      # (n_tot × bs)
    buffer1::MT             # (aug_vec_length × bs) — solve-time scratch (r3)
    buffer2::MT             # (aug_vec_length × bs) — solve-time scratch (r4)

    hess_op::OPT
    jt_op::OPT
    j_op::OPT
end

pr_diag(bkkt::ScaledSparseUniformBatchKKTSystem) = view(bkkt.nzVals, 1:bkkt.n_tot, :)
function du_diag(bkkt::ScaledSparseUniformBatchKKTSystem)
    du_off = size(bkkt.nzVals, 1) - bkkt.m
    return view(bkkt.nzVals, du_off+1:du_off+bkkt.m, :)
end

# ---------- build ----------

function MadNLP.create_kkt_system(
    ::Type{MadNLP.ScaledSparseKKTSystem},
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
    nzVals = fill!(similar(bcb.con_buffer, aug_mat_length, batch_size), zero(T))

    a = n_tot; b = n_tot + n_hess; c = n_tot + n_hess + n_jac; d = n_tot + n_hess + n_jac + n_slack
    I[1:a]     .= 1:n_tot
    I[a+1:b]     = hess_I
    I[b+1:c]   .= jac_I .+ n_tot
    I[c+1:d]   .= bcb.ind_ineq .+ n_tot
    I[d+1:d+m] .= (n_tot+1):(n_tot+m)
    J[1:a]     .= 1:n_tot
    J[a+1:b]     = hess_J
    J[b+1:c]   .= jac_J
    J[c+1:d]   .= (n+1):(n+n_slack)
    J[d+1:d+m] .= (n_tot+1):(n_tot+m)

    V_col1 = _madnlp_unsafe_column_wrap(nzVals, aug_mat_length, 1, VT)
    aug_raw = MadNLP.SparseMatrixCOO(aug_vec_length, aug_vec_length, I, J, V_col1)
    aug_com, aug_csc_map = MadNLP.coo_to_csc(aug_raw)

    nnz_csc        = SparseArrays.nnz(aug_com)
    aug_com_nzvals = fill!(similar(nzVals, nnz_csc, batch_size), zero(T))
    scaled_nzVals  = similar(nzVals)
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

    scaling_factor = fill!(similar(nzVals, n_tot, batch_size), one(T))
    buffer1        = similar(nzVals, aug_vec_length, batch_size)
    buffer2        = similar(nzVals, aug_vec_length, batch_size)

    return ScaledSparseUniformBatchKKTSystem{T, typeof(batch_ls), MT, VI, typeof(I),
                                               typeof(jt_op), typeof(batch_views)}(
        nzVals, I, J, batch_ls, rhs_buffer, compact_buffer, batch_size, batch_views,
        aug_com_nzvals, scaled_nzVals, batch_csc_map, n_tot, m, n_hess,
        reg, l_diag, u_diag, l_lower, u_lower,
        scaling_factor, buffer1, buffer2,
        hess_op, jt_op, j_op,
    )
end

# ---------- initialize ----------

function MadNLP.initialize!(bkkt::ScaledSparseUniformBatchKKTSystem{T}) where {T}
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
    fill!(bkkt.scaling_factor, one(T))
    fill!(bkkt.aug_com_nzvals, zero(T))
    return nothing
end

# ---------- aug-diagonal hook (K2.5 pr_diag + scaling_factor) ----------
#
# Called from `_set_aug_diagonal_reg_{un,}masked!` after `kkt.reg`, `l_diag`,
# `u_diag`, `l_lower`, `u_lower` are populated. Mirrors
# `_set_aug_diagonal!(::ScaledSparseKKTSystem)` column-by-column:
#   scaling[i] ← √l_diag[i]  (std-form: all vars are lb → every column)
#   pr_diag[i] ← zl[i] + reg[i] · scaling[i]²
#
# `_aug_l_diag_sign(ScaledSparse) = +1` already ensures `l_diag = x - xl ≥ 0`.

function _finalize_aug_diagonal_scaled_full!(bkkt::ScaledSparseUniformBatchKKTSystem{T},
                                              ind_lb, ind_ub) where {T}
    sf = bkkt.scaling_factor
    fill!(sf, one(T))
    if length(ind_lb) > 0
        view(sf, ind_lb, :) .*= sqrt.(bkkt.l_diag)
    end
    if length(ind_ub) > 0
        view(sf, ind_ub, :) .*= sqrt.(bkkt.u_diag)
    end
    # pr_diag = Δx (bound-complementarity) + reg · scaling². We index
    # `bkkt.nzVals` directly with `ind_lb` / `ind_ub` — a single view of
    # the underlying matrix. Going through `pr_diag(bkkt)` (a UnitRange
    # view of nzVals) and then nesting a `CuVector` index triggers scalar
    # indexing on GPU (GPUArrays can't materialize a nested `view(view(CuMatrix,
    # range, :), CuVector, :)`).
    _pr = pr_diag(bkkt)
    fill!(_pr, zero(T))
    if length(ind_lb) > 0
        view(bkkt.nzVals, ind_lb, :) .+= bkkt.l_lower           # +zl at ind_lb
    end
    if length(ind_ub) > 0
        view(bkkt.nzVals, ind_ub, :) .+= bkkt.u_lower           # +zu at ind_ub
    end
    _pr .+= bkkt.reg .* sf .^ 2
    return
end

@inline function _finalize_aug_diagonal!(bkkt::ScaledSparseUniformBatchKKTSystem,
                                          s::UniformBatchMPCSolver)
    _finalize_aug_diagonal_scaled_full!(bkkt, _get_ind_lb(s), _get_ind_ub(s))
    return
end

@inline function _finalize_aug_diagonal_masked!(bkkt::ScaledSparseUniformBatchKKTSystem,
                                                 s::UniformBatchMPCSolver)
    # In-place ifelse variant: touch only active-column entries.
    mask = s.state.workspace.active_mask
    ind_lb, ind_ub = _get_ind_lb(s), _get_ind_ub(s)
    sf = bkkt.scaling_factor
    # Recompute scaling_factor for active columns.
    sf_active = similar(sf)
    fill!(sf_active, one(eltype(sf)))
    length(ind_lb) > 0 && (view(sf_active, ind_lb, :) .*= sqrt.(bkkt.l_diag))
    length(ind_ub) > 0 && (view(sf_active, ind_ub, :) .*= sqrt.(bkkt.u_diag))
    @. sf = ifelse(mask == 1, sf_active, sf)
    # pr_diag for active columns. Fresh buffer sized like nzVals primal block
    # (n_tot × bs), avoiding nested views into `pr_diag(bkkt)`.
    _pr = pr_diag(bkkt)
    pr_active = similar(bkkt.scaling_factor)
    fill!(pr_active, zero(eltype(_pr)))
    length(ind_lb) > 0 && (view(pr_active, ind_lb, :) .+= bkkt.l_lower)
    length(ind_ub) > 0 && (view(pr_active, ind_ub, :) .+= bkkt.u_lower)
    pr_active .+= bkkt.reg .* sf .^ 2
    @. _pr = ifelse(mask == 1, pr_active, _pr)
    return
end

@inline _aug_l_diag_sign(::ScaledSparseUniformBatchKKTSystem) = one(Float64)  # +1: l_diag = x - xl
@inline _aug_u_diag_sign(::ScaledSparseUniformBatchKKTSystem) = one(Float64)

# ---------- build_kkt! — per-position scaling then scatter ----------
#
# Layout of `nzVals` rows, shared across the batch via aug_I/aug_J:
#   [1              :n_tot]            primal diag (already = pr_diag with reg·sf²)
#   [n_tot+1        :n_tot+n_hess]     hess (lower triangular, i,j ≤ n_tot)
#   [n_tot+n_hess+1 :n_tot+n_hess+n_jac]  jac (i > n_tot, j ≤ n_tot)
#   [...+1          :...+n_slack]       slack columns (i > n_tot, j ≤ n_tot)
#   [end-m+1        :end]               dual diag
#
# Scaling per segment (K2.5):
#   primal diag  :  copy (pr_diag already includes reg·sf²)
#   hess(i,j)    :  × sf[i] · sf[j]
#   jac(i > n_tot, j ≤ n_tot) : × sf[j]
#   dual diag    :  copy
function _scale_batch_nzvals!(scaled::AbstractMatrix{T}, src::AbstractMatrix{T},
                               I::AbstractVector, J::AbstractVector,
                               sf::AbstractMatrix{T}, n_tot::Int, m::Int) where {T}
    @inbounds for k in eachindex(I)
        i = Int(I[k]); j = Int(J[k])
        if k <= n_tot
            for b in axes(scaled, 2); scaled[k, b] = src[k, b]; end                   # primal diag
        elseif i <= n_tot && j <= n_tot
            for b in axes(scaled, 2); scaled[k, b] = src[k, b] * sf[i, b] * sf[j, b]; end  # hess
        elseif n_tot + 1 <= i <= n_tot + m && j <= n_tot
            for b in axes(scaled, 2); scaled[k, b] = src[k, b] * sf[j, b]; end         # jac
        elseif n_tot + 1 <= i <= n_tot + m && n_tot + 1 <= j <= n_tot + m
            for b in axes(scaled, 2); scaled[k, b] = src[k, b]; end                   # dual diag
        end
    end
end

function MadNLP.build_kkt!(bkkt::ScaledSparseUniformBatchKKTSystem)
    _scale_batch_nzvals!(bkkt.scaled_nzVals, bkkt.nzVals,
                         bkkt.aug_I, bkkt.aug_J, bkkt.scaling_factor, bkkt.n_tot, bkkt.m)
    _scatter_to_csc!(vec(bkkt.aug_com_nzvals), vec(bkkt.scaled_nzVals), bkkt.batch_csc_map)
    return nothing
end

# ---------- factor / solve (reuse shared SparseKKT plumbing) ----------

function MadNLP.factorize_kkt!(bkkt::ScaledSparseUniformBatchKKTSystem)
    fv = active_view(bkkt.batch_views)
    is_identity_view(fv) ||
        compact_active_columns_inplace!(bkkt.aug_com_nzvals, fv, bkkt.compact_buffer)
    factorize_active!(bkkt.batch_solver, fv)
    return nothing
end

function MadNLP.solve_linear_system!(bkkt::ScaledSparseUniformBatchKKTSystem, rhs::AbstractMatrix)
    av = active_view(bkkt.batch_views)
    is_identity_view(av) ||
        compact_active_columns_inplace!(rhs, av, bkkt.compact_buffer)
    solve_active!(bkkt.batch_solver, rhs, av)
    return rhs
end

# ---------- K2.5 solve_kkt! ----------
# Mirrors `solve_kkt!(::ScaledSparseKKTSystem)`: transform bound-dual rhs
# through √l_diag / √u_diag, scale the primal rhs by `scaling_factor`, solve
# the (well-conditioned) scaled system, then unscale both the primal solution
# and the bound-dual updates.

function MadNLP.solve_kkt!(bkkt::ScaledSparseUniformBatchKKTSystem, batch_solver)
    d = batch_solver isa BatchUnreducedKKTVector ? batch_solver : batch_solver.state.d
    return MadNLP.solve_kkt!(bkkt, d)
end

function MadNLP.solve_kkt!(bkkt::ScaledSparseUniformBatchKKTSystem{T},
                            d::BatchUnreducedKKTVector{T}) where {T}
    n, m = bkkt.n_tot, bkkt.m
    ind_lb, ind_ub = d.ind_lb, d.ind_ub
    nlb, nub = length(ind_lb), length(ind_ub)
    av = active_view(bkkt.batch_views)

    r3 = bkkt.buffer1
    r4 = bkkt.buffer2
    fill!(r3, zero(T)); fill!(r4, zero(T))

    wzl = MadNLP.dual_lb(d)   # (nlb × bs)
    wzu = MadNLP.dual_ub(d)   # (nub × bs)

    # r3[ind_lb] = wzl ./ sqrt(l_diag); r3[ind_ub] *= sqrt(u_diag)
    if nlb > 0
        view(r3, ind_lb, :) .= wzl
        view(r3, ind_lb, :) ./= sqrt.(bkkt.l_diag)
    end
    if nub > 0
        view(r3, ind_ub, :) .*= sqrt.(bkkt.u_diag)
    end

    # r4[ind_ub] = wzu ./ sqrt(u_diag); r4[ind_lb] *= sqrt(l_diag)
    if nub > 0
        view(r4, ind_ub, :) .= wzu
        view(r4, ind_ub, :) ./= sqrt.(bkkt.u_diag)
    end
    if nlb > 0
        view(r4, ind_lb, :) .*= sqrt.(bkkt.l_diag)
    end

    # Build primal rhs = scaling_factor .* xp + (r3 + r4)   on the primal block.
    xp = MadNLP.primal(d)                 # (n × bs)
    r3p = view(r3, 1:n, :)
    r4p = view(r4, 1:n, :)
    xp .= xp .* bkkt.scaling_factor .+ r3p .+ r4p

    # Copy into rhs_buffer and solve.
    rhs = bkkt.rhs_buffer
    pd  = MadNLP.primal_dual(d)           # primal+dual slice
    copyto!(rhs, pd)
    MadNLP.solve_linear_system!(bkkt, rhs)
    is_identity_view(av) ? copyto!(pd, rhs) :
                           scatter_batch_view_columns!(pd, rhs, av)

    # Unpack: primal solution back-scales by scaling_factor; bound-dual updates
    # follow the same closed-form as the scalar ScaledSparseKKT.
    MadNLP.primal(d) .*= bkkt.scaling_factor

    # Avoid nested views: index the underlying `d.values` CuMatrix directly
    # with `ind_lb` / `ind_ub` (nested UnitRange + CuVector views trigger
    # scalar indexing on GPU).
    if nlb > 0
        xp_lr = view(d.values, ind_lb, :)
        @. wzl = (wzl - bkkt.l_lower * xp_lr) / bkkt.l_diag
    end
    if nub > 0
        xp_ur = view(d.values, ind_ub, :)
        @. wzu = (-wzu + bkkt.u_lower * xp_ur) / bkkt.u_diag
    end
    return nothing
end

# ---------- K · x (batch) ----------
# Same block structure as SparseUniformBatch's `mul!` but with the scaling
# factor folded into the primal blocks (matches the scalar ScaledSparse
# `mul!`). `_kktmul!` handles du_diag and the bound-dual coupling unchanged.

function LinearAlgebra.mul!(
    w::BatchUnreducedKKTVector{T}, bkkt::ScaledSparseUniformBatchKKTSystem{T},
    x::BatchUnreducedKKTVector{T}, alpha = one(T), beta = zero(T),
) where {T}
    xv = MadNLP.full(x)
    batch_spmv!(MadNLP.primal(w), bkkt.hess_op, xv, alpha, beta)
    batch_spmv!(MadNLP.primal(w), bkkt.jt_op,   xv, alpha, one(T); val_offset = bkkt.n_tot)
    batch_spmv!(MadNLP.dual(w),   bkkt.j_op,    xv, alpha, beta)
    # _kktmul uses `reg` (not pr_diag) — consistent with ScaledSparseKKTSystem.
    _kktmul!(w, x, bkkt.reg, du_diag(bkkt),
             bkkt.l_lower, bkkt.u_lower, bkkt.l_diag, bkkt.u_diag,
             alpha, beta)
    # Mehrotra residual uses scaled primal: add reg · scaling² · xp already
    # done in `_kktmul!`; the scaling-factor coupling is a no-op on primal
    # because we don't premultiply inputs here (consistent with scalar path).
    return w
end

# ---------- eval wrappers (same as SparseUniformBatch) ----------

function MadNLP.eval_jac_wrapper!(batch_solver, bkkt::ScaledSparseUniformBatchKKTSystem)
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

function MadNLP.eval_lag_hess_wrapper!(batch_solver, bkkt::ScaledSparseUniformBatchKKTSystem)
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

MadNLP.jtprod!(res::AbstractMatrix, bkkt::ScaledSparseUniformBatchKKTSystem, y::BatchVector) =
    (batch_spmv!(res, bkkt.jt_op, MadNLP.full(y)); res)

MadNLP.jtprod!(jacl::BatchVector, bkkt::ScaledSparseUniformBatchKKTSystem, y::BatchVector) =
    MadNLP.jtprod!(MadNLP.full(jacl), bkkt, y)
