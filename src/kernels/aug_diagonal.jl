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

function _set_aug_diagonal_reg_unmasked!(kkt, s::AnyMPCSolver)
    kkt.reg .= _del_w(s)
    du_diag(kkt) .= _del_c(s)
    kkt.l_diag .= _xl_r(s) .- _x_lr(s)
    kkt.u_diag .= _x_ur(s) .- _xu_r(s)
    kkt.l_lower .= _zl_r(s)
    kkt.u_lower .= _zu_r(s)
    pr_diag(kkt) .= kkt.reg
    _pr_diag_lb_view(kkt, s) .-= kkt.l_lower ./ kkt.l_diag
    _pr_diag_ub_view(kkt, s) .-= kkt.u_lower ./ kkt.u_diag
    return
end

# ---------- scalar (Scaled-KKT specialization) ----------

function set_aug_diagonal_reg!(kkt::MadNLP.ScaledSparseKKTSystem{T}, solver::MPCSolver{T}) where {T}
    state = solver.state
    fill!(kkt.reg, state.del_w)
    fill!(kkt.du_diag, state.del_c)
    kkt.l_diag .= state.x_lr
    copyto!(kkt.l_lower, state.zl_r)
    MadNLP._set_aug_diagonal!(kkt)
    return
end

# ---------- batch (masked active-set variant) ----------

function _set_aug_diagonal_reg_masked!(kkt, solver::AbstractBatchMPCSolver)
    state = solver.state
    xl_r = lower(state.xl)
    x_lr = lower(state.x)
    zl_r = lower(state.zl)
    xu_r = upper(state.xu)
    x_ur = upper(state.x)
    zu_r = upper(state.zu)
    mask = state.workspace.active_mask
    _du = du_diag(kkt)
    _pr = pr_diag(kkt)
    @. kkt.reg = ifelse(mask == 1, state.del_w, kkt.reg)
    @. _du = ifelse(mask == 1, state.del_c, _du)
    @. kkt.l_diag = ifelse(mask == 1, xl_r - x_lr, kkt.l_diag)
    @. kkt.u_diag = ifelse(mask == 1, x_ur - xu_r, kkt.u_diag)
    @. kkt.l_lower = ifelse(mask == 1, zl_r, kkt.l_lower)
    @. kkt.u_lower = ifelse(mask == 1, zu_r, kkt.u_lower)
    @. _pr = ifelse(mask == 1, kkt.reg, _pr)
    pr_diag_lb = view(pr_diag(kkt), _get_ind_lb(solver), :)
    pr_diag_ub = view(pr_diag(kkt), _get_ind_ub(solver), :)
    @. pr_diag_lb = ifelse(mask == 1, pr_diag_lb - kkt.l_lower / kkt.l_diag, pr_diag_lb)
    @. pr_diag_ub = ifelse(mask == 1, pr_diag_ub - kkt.u_lower / kkt.u_diag, pr_diag_ub)
    return
end

