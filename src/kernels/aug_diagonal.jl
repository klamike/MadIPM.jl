# Augmented-system diagonal assembly: sets `kkt.reg`, `du_diag`, `l_diag`,
# `u_diag`, `l_lower`, `u_lower`, `pr_diag` from the current iterate.
# The masked path (`_set_aug_diagonal_reg_masked!`) is taken only when the
# batch active-set view is narrower than the root batch — it preserves the
# diagonal values for inactive (converged) instances so their last
# factorization stays valid.

# ---------- KKT index accessors ----------
# Thin wrappers so `_pr_diag_*_view` and `_set_*` functions can pull the
# matching index vectors from both scalar `AbstractKKTSystem` and batch
# `AbstractBatchKKTSystem` uniformly.

@inline pr_diag(kkt::MadNLP.AbstractKKTSystem) = kkt.pr_diag
@inline du_diag(kkt::MadNLP.AbstractKKTSystem) = kkt.du_diag
@inline _kkt_ind_lb(kkt::MadNLP.AbstractKKTSystem, ::MPCSolver)         = kkt.ind_lb
@inline _kkt_ind_ub(kkt::MadNLP.AbstractKKTSystem, ::MPCSolver)         = kkt.ind_ub
@inline _kkt_ind_lb(::AbstractBatchKKTSystem, s::UniformBatchMPCSolver) = _get_ind_lb(s)
@inline _kkt_ind_ub(::AbstractBatchKKTSystem, s::UniformBatchMPCSolver) = _get_ind_ub(s)
@inline _pr_diag_lb_view(kkt::MadNLP.AbstractKKTSystem, s::MPCSolver)              = view(pr_diag(kkt), _kkt_ind_lb(kkt, s))
@inline _pr_diag_ub_view(kkt::MadNLP.AbstractKKTSystem, s::MPCSolver)              = view(pr_diag(kkt), _kkt_ind_ub(kkt, s))
@inline _pr_diag_lb_view(kkt::AbstractBatchKKTSystem, s::UniformBatchMPCSolver)   = view(pr_diag(kkt), _kkt_ind_lb(kkt, s), :)
@inline _pr_diag_ub_view(kkt::AbstractBatchKKTSystem, s::UniformBatchMPCSolver)   = view(pr_diag(kkt), _kkt_ind_ub(kkt, s), :)
# `SparseUniformBatchKKTSystem.pr_diag` is itself a `view(nzVals, 1:n_tot, :)` —
# nesting a CuArray-indexed view through a UnitRange-indexed parent triggers
# scalar getindex on the UnitRange. View the underlying nzVals directly.
@inline _pr_diag_lb_view(kkt::SparseUniformBatchKKTSystem, s::UniformBatchMPCSolver) = view(kkt.nzVals, _kkt_ind_lb(kkt, s), :)
@inline _pr_diag_ub_view(kkt::SparseUniformBatchKKTSystem, s::UniformBatchMPCSolver) = view(kkt.nzVals, _kkt_ind_ub(kkt, s), :)

# ---------- entry points ----------
# Scalar: always the full-rewrite path. Batch: masked path kicks in once the
# active-set view shrinks from the root batch.

function set_aug_diagonal_reg!(kkt::MadNLP.AbstractKKTSystem{T}, s::MPCSolver{T}) where {T}
    _set_aug_diagonal_reg_unmasked!(kkt, s)
end
function set_aug_diagonal_reg!(kkt::AbstractBatchKKTSystem, s::UniformBatchMPCSolver)
    if is_identity_view(active_view(s.problem.batch_views))
        _set_aug_diagonal_reg_unmasked!(kkt, s)
    else
        _set_aug_diagonal_reg_masked!(kkt, s)
    end
end

@inline _aug_l_diag_sign(kkt) = -one(eltype(kkt.reg))
@inline _aug_u_diag_sign(kkt) = -one(eltype(kkt.reg))
@inline _aug_l_diag_sign(kkt::MadNLP.ScaledSparseKKTSystem) = one(eltype(kkt.reg))
@inline _aug_u_diag_sign(kkt::MadNLP.ScaledSparseKKTSystem) = one(eltype(kkt.reg))

function _set_aug_diagonal_reg_unmasked!(kkt, s::MaybeBatchMPCSolver)
    kkt.reg .= _del_w(s)
    du_diag(kkt) .= _del_c(s)
    sl = _aug_l_diag_sign(kkt)
    su = _aug_u_diag_sign(kkt)
    kkt.l_diag .= sl .* (_x_lr(s) .- _xl_r(s))
    kkt.u_diag .= su .* (_xu_r(s) .- _x_ur(s))
    kkt.l_lower .= _zl_r(s)
    kkt.u_lower .= _zu_r(s)
    _finalize_aug_diagonal!(kkt, s)
    return
end

@inline function _finalize_aug_diagonal!(kkt, s::MaybeBatchMPCSolver)
    pr_diag(kkt) .= kkt.reg
    _pr_diag_lb_view(kkt, s) .-= kkt.l_lower ./ kkt.l_diag
    _pr_diag_ub_view(kkt, s) .-= kkt.u_lower ./ kkt.u_diag
    return
end

@inline function _finalize_aug_diagonal!(kkt::MadNLP.ScaledSparseKKTSystem, ::MaybeBatchMPCSolver)
    MadNLP._set_aug_diagonal!(kkt)
    return
end

function _set_aug_diagonal_reg_masked!(kkt, solver::UniformBatchMPCSolver)
    state = solver.state
    sl = _aug_l_diag_sign(kkt)
    su = _aug_u_diag_sign(kkt)
    xl_r = lower(state.xl); x_lr = lower(state.x); zl_r = lower(state.zl)
    xu_r = upper(state.xu); x_ur = upper(state.x); zu_r = upper(state.zu)
    mask = state.workspace.active_mask
    _du = du_diag(kkt)
    @. kkt.reg     = ifelse(mask == 1, state.del_w,           kkt.reg)
    @. _du         = ifelse(mask == 1, state.del_c,           _du)
    @. kkt.l_diag  = ifelse(mask == 1, sl * (x_lr - xl_r),    kkt.l_diag)
    @. kkt.u_diag  = ifelse(mask == 1, su * (xu_r - x_ur),    kkt.u_diag)
    @. kkt.l_lower = ifelse(mask == 1, zl_r,                  kkt.l_lower)
    @. kkt.u_lower = ifelse(mask == 1, zu_r,                  kkt.u_lower)
    _finalize_aug_diagonal_masked!(kkt, solver)
    return
end

@inline function _finalize_aug_diagonal_masked!(kkt, s::UniformBatchMPCSolver)
    mask = s.state.workspace.active_mask
    _pr = pr_diag(kkt)
    @. _pr = ifelse(mask == 1, kkt.reg, _pr)
    pr_diag_lb = _pr_diag_lb_view(kkt, s)
    pr_diag_ub = _pr_diag_ub_view(kkt, s)
    @. pr_diag_lb = ifelse(mask == 1, pr_diag_lb - kkt.l_lower / kkt.l_diag, pr_diag_lb)
    @. pr_diag_ub = ifelse(mask == 1, pr_diag_ub - kkt.u_lower / kkt.u_diag, pr_diag_ub)
    return
end
