# ---------- scalar ----------

function set_aug_diagonal_reg!(kkt::MadNLP.AbstractKKTSystem{T}, solver::MPCSolver{T}) where {T}
    state = solver.state
    fill!(kkt.reg, state.del_w)
    fill!(kkt.du_diag, state.del_c)
    kkt.l_diag .= .-state.x_lr
    copyto!(kkt.l_lower, state.zl_r)
    copyto!(kkt.pr_diag, kkt.reg)
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    return
end

function set_aug_diagonal_reg!(kkt::MadNLP.ScaledSparseKKTSystem{T}, solver::MPCSolver{T}) where {T}
    state = solver.state
    fill!(kkt.reg, state.del_w)
    fill!(kkt.du_diag, state.del_c)
    kkt.l_diag .= state.x_lr
    copyto!(kkt.l_lower, state.zl_r)
    MadNLP._set_aug_diagonal!(kkt)
    return
end

# ---------- batch ----------

function _set_aug_diagonal_reg_unmasked!(kkt, solver::AbstractBatchMPCSolver)
    state = solver.state
    kkt.reg .= state.del_w
    du_diag(kkt) .= state.del_c
    kkt.l_diag .= lower(state.xl) .- lower(state.x)
    kkt.u_diag .= upper(state.x) .- upper(state.xu)
    kkt.l_lower .= lower(state.zl)
    kkt.u_lower .= upper(state.zu)
    pr_diag(kkt) .= kkt.reg
    pr_diag_lb = view(pr_diag(kkt), _get_ind_lb(solver), :)
    pr_diag_ub = view(pr_diag(kkt), _get_ind_ub(solver), :)
    pr_diag_lb .-= kkt.l_lower ./ kkt.l_diag
    pr_diag_ub .-= kkt.u_lower ./ kkt.u_diag
    return
end

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

function set_aug_diagonal_reg!(kkt, solver::AbstractBatchMPCSolver)
    if is_identity_view(active_view(solver.problem.batch_views))
        _set_aug_diagonal_reg_unmasked!(kkt, solver)
    else
        _set_aug_diagonal_reg_masked!(kkt, solver)
    end
end
