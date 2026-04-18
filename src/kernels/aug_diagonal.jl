# Unified diagonal-block accessors so the same broadcasted writes work for
# scalar (`AbstractKKTSystem` exposes `pr_diag` / `du_diag` as fields) and
# batch (`AbstractBatchKKTSystem` packs them as views into `nzVals` —
# already provided as `pr_diag(::AbstractBatchKKTSystem)` /
# `du_diag(::AbstractBatchKKTSystem)`).

@inline pr_diag(kkt::MadNLP.AbstractKKTSystem) = kkt.pr_diag
@inline du_diag(kkt::MadNLP.AbstractKKTSystem) = kkt.du_diag

# ---------- unified (scalar SparseKKT + batch unmasked) ----------

# Scalar's KKT exposes `kkt.ind_lb` / `kkt.ind_ub` directly; batch keeps
# the indices on the callback and exposes them through the solver.
@inline _kkt_ind_lb(kkt::MadNLP.AbstractKKTSystem, ::MPCSolver)         = kkt.ind_lb
@inline _kkt_ind_ub(kkt::MadNLP.AbstractKKTSystem, ::MPCSolver)         = kkt.ind_ub
@inline _kkt_ind_lb(::AbstractBatchKKTSystem, s::AbstractBatchMPCSolver) = _get_ind_lb(s)
@inline _kkt_ind_ub(::AbstractBatchKKTSystem, s::AbstractBatchMPCSolver) = _get_ind_ub(s)

# `view(pr_diag(kkt), idx)` for scalar (1D) vs `view(pr_diag(kkt), idx, :)`
# for batch (2D matrix), so the mutating broadcast picks up the right shape.
@inline _pr_diag_lb_view(kkt::MadNLP.AbstractKKTSystem, s::MPCSolver)              = view(pr_diag(kkt), _kkt_ind_lb(kkt, s))
@inline _pr_diag_ub_view(kkt::MadNLP.AbstractKKTSystem, s::MPCSolver)              = view(pr_diag(kkt), _kkt_ind_ub(kkt, s))
@inline _pr_diag_lb_view(kkt::AbstractBatchKKTSystem, s::AbstractBatchMPCSolver)   = view(pr_diag(kkt), _kkt_ind_lb(kkt, s), :)
@inline _pr_diag_ub_view(kkt::AbstractBatchKKTSystem, s::AbstractBatchMPCSolver)   = view(pr_diag(kkt), _kkt_ind_ub(kkt, s), :)

function set_aug_diagonal_reg!(kkt::MadNLP.AbstractKKTSystem{T}, s::MPCSolver{T}) where {T}
    _set_aug_diagonal_reg_unmasked!(kkt, s)
end
function set_aug_diagonal_reg!(kkt::AbstractBatchKKTSystem, s::AbstractBatchMPCSolver)
    if is_identity_view(active_view(s.problem.batch_views))
        _set_aug_diagonal_reg_unmasked!(kkt, s)
    else
        _set_aug_diagonal_reg_masked!(kkt, s)
    end
end

# Same math, different storage convention:
#   * basic scalar / batch KKT store `l_diag = xl - x` (≤ 0) and
#     `u_diag = x - xu` (≤ 0); the lb / ub pr_diag updates flow through
#     `_finalize_aug_diagonal!`.
#   * `MadNLP.ScaledSparseKKTSystem` expects the positive convention
#     `l_diag = x - xl`, `u_diag = xu - x`, and its `_set_aug_diagonal!`
#     handles the (different) pr_diag layout with the scaling factor.
@inline _aug_l_diag_sign(kkt) = -one(eltype(kkt.reg))
@inline _aug_u_diag_sign(kkt) = -one(eltype(kkt.reg))
@inline _aug_l_diag_sign(kkt::MadNLP.ScaledSparseKKTSystem) = one(eltype(kkt.reg))
@inline _aug_u_diag_sign(kkt::MadNLP.ScaledSparseKKTSystem) = one(eltype(kkt.reg))

function _set_aug_diagonal_reg_unmasked!(kkt, s::AnyMPCSolver)
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

@inline function _finalize_aug_diagonal!(kkt, s::AnyMPCSolver)
    pr_diag(kkt) .= kkt.reg
    _pr_diag_lb_view(kkt, s) .-= kkt.l_lower ./ kkt.l_diag
    _pr_diag_ub_view(kkt, s) .-= kkt.u_lower ./ kkt.u_diag
    return
end

@inline function _finalize_aug_diagonal!(kkt::MadNLP.ScaledSparseKKTSystem, ::AnyMPCSolver)
    # Delegates to MadNLP's scaled implementation: it lays out `pr_diag`
    # itself (computing scaling_factor and the scaled regularization).
    MadNLP._set_aug_diagonal!(kkt)
    return
end

# ---------- batch (masked active-set variant) ----------
#
# Mirrors the unmasked body but gates each write behind the active mask so
# converged instances retain their previous diagonal values. Today only the
# basic batch KKT systems exist, but the sign / pr_diag-finalize work goes
# through the same hooks as the unmasked path so a future
# `ScaledSparseUniformBatchKKTSystem` would be supported by adding the
# matching `_aug_*_diag_sign` / `_finalize_aug_diagonal_masked!` overrides
# without forking this body.

function _set_aug_diagonal_reg_masked!(kkt, solver::AbstractBatchMPCSolver)
    state = solver.state
    sl = _aug_l_diag_sign(kkt)
    su = _aug_u_diag_sign(kkt)
    xl_r = lower(state.xl); x_lr = lower(state.x); zl_r = lower(state.zl)
    xu_r = upper(state.xu); x_ur = upper(state.x); zu_r = upper(state.zu)
    mask = state.workspace.active_mask
    _du = du_diag(kkt)
    @. kkt.reg     = ifelse(mask == 1, state.del_w,           kkt.reg)
    @. _du         = ifelse(mask == 1, state.del_c,           _du)
    @. kkt.l_diag  = ifelse(mask == 1, sl * (xl_r - x_lr),    kkt.l_diag)
    @. kkt.u_diag  = ifelse(mask == 1, su * (x_ur - xu_r),    kkt.u_diag)
    @. kkt.l_lower = ifelse(mask == 1, zl_r,                  kkt.l_lower)
    @. kkt.u_lower = ifelse(mask == 1, zu_r,                  kkt.u_lower)
    _finalize_aug_diagonal_masked!(kkt, solver)
    return
end

@inline function _finalize_aug_diagonal_masked!(kkt, s::AbstractBatchMPCSolver)
    mask = s.state.workspace.active_mask
    _pr = pr_diag(kkt)
    @. _pr = ifelse(mask == 1, kkt.reg, _pr)
    pr_diag_lb = view(pr_diag(kkt), _get_ind_lb(s), :)
    pr_diag_ub = view(pr_diag(kkt), _get_ind_ub(s), :)
    @. pr_diag_lb = ifelse(mask == 1, pr_diag_lb - kkt.l_lower / kkt.l_diag, pr_diag_lb)
    @. pr_diag_ub = ifelse(mask == 1, pr_diag_ub - kkt.u_lower / kkt.u_diag, pr_diag_ub)
    return
end

