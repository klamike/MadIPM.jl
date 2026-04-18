# Batch IPM kernels for the std-form solver: LB-only variables, equality
# constraints. Upper-bound / inequality paths from the general MPC have been
# removed.

function dual_objective!(dual_obj, y_vals, rhs_vals, zl_r, xl_r, sum_lb, nlb)
    T = eltype(dual_obj)
    batch_mapreduce!(*, +, zero(T), dual_obj, y_vals, rhs_vals)
    dual_obj .*= -one(T)
    if nlb > 0
        batch_mapreduce!(*, +, zero(T), sum_lb, zl_r, xl_r)
        dual_obj .+= sum_lb
    end
    return dual_obj
end

function set_initial_primal_rhs!(solver::AbstractBatchMPCSolver)
    p = solver.p
    fill!(MadNLP.full(p), 0.0)
    py = MadNLP.dual(p)
    b = MadNLP.full(solver.c)
    py .= .-b
    return
end

function set_initial_dual_rhs!(solver::AbstractBatchMPCSolver)
    p = solver.p
    fill!(MadNLP.full(p), 0.0)
    px = MadNLP.primal(p)
    c = MadNLP.primal(solver.f)
    px .= .-c
    return
end

function set_predictive_rhs!(solver::AbstractBatchMPCSolver, kkt::AbstractBatchKKTSystem)
    px  = MadNLP.primal(solver.p)
    py  = MadNLP.dual(solver.p)
    pzl = MadNLP.dual_lb(solver.p)
    f   = MadNLP.primal(solver.f)
    c   = MadNLP.full(solver.c)
    zl  = MadNLP.full(solver.zl)
    jacl = MadNLP.full(solver.jacl)
    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)

    fill!(MadNLP.full(solver.p), 0.0)
    px  .= .-f .+ zl .- jacl
    py  .= .-c
    pzl .= (xl_r .- x_lr) .* zl_r
    return
end

function set_correction_rhs!(bs::AbstractBatchMPCSolver, kkt::AbstractBatchKKTSystem, mu, correction_lb, ind_lb)
    px  = MadNLP.primal(bs.p)
    py  = MadNLP.dual(bs.p)
    pzl = MadNLP.dual_lb(bs.p)
    f   = MadNLP.primal(bs.f)
    c   = MadNLP.full(bs.c)
    zl  = MadNLP.full(bs.zl)
    jacl = MadNLP.full(bs.jacl)
    xl_r = lower(bs.xl)
    x_lr = lower(bs.x)
    zl_r = lower(bs.zl)

    px  .= .-f .+ zl .- jacl
    py  .= .-c
    pzl .= (xl_r .- x_lr) .* zl_r .+ mu .- correction_lb
    return
end

function get_correction!(batch_solver::AbstractBatchMPCSolver, correction_lb)
    dlb  = MadNLP.dual_lb(batch_solver.d)
    dx_lr = xp_lr(batch_solver.d)
    correction_lb .= dx_lr .* dlb
    return
end

function _set_aug_diagonal_reg_unmasked!(kkt, solver::AbstractBatchMPCSolver)
    kkt.reg .= solver.del_w
    du_diag(kkt) .= solver.del_c
    kkt.l_diag .= lower(solver.xl) .- lower(solver.x)
    kkt.l_lower .= lower(solver.zl)
    pr_diag(kkt) .= kkt.reg
    pr_diag_lb = view(pr_diag(kkt), _get_ind_lb(solver), :)
    pr_diag_lb .-= kkt.l_lower ./ kkt.l_diag
    return
end

function _set_aug_diagonal_reg_masked!(kkt, solver::AbstractBatchMPCSolver)
    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    mask = solver.workspace.active_mask
    _du = du_diag(kkt)
    _pr = pr_diag(kkt)
    @. kkt.reg = ifelse(mask == 1, solver.del_w, kkt.reg)
    @. _du = ifelse(mask == 1, solver.del_c, _du)
    @. kkt.l_diag = ifelse(mask == 1, xl_r - x_lr, kkt.l_diag)
    @. kkt.l_lower = ifelse(mask == 1, zl_r, kkt.l_lower)
    @. _pr = ifelse(mask == 1, kkt.reg, _pr)
    pr_diag_lb = view(pr_diag(kkt), _get_ind_lb(solver), :)
    @. pr_diag_lb = ifelse(mask == 1, pr_diag_lb - kkt.l_lower / kkt.l_diag, pr_diag_lb)
    return
end

function set_aug_diagonal_reg!(kkt, solver::AbstractBatchMPCSolver)
    if is_identity_view(active_view(solver.batch_views))
        _set_aug_diagonal_reg_unmasked!(kkt, solver)
    else
        _set_aug_diagonal_reg_masked!(kkt, solver)
    end
end

function get_complementarity_measure!(solver::AbstractBatchMPCSolver)
    ws = solver.workspace
    nlb = solver.d.nlb
    T = eltype(ws.mu_curr)

    if nlb == 0
        fill!(ws.mu_curr, zero(T))
        return ws.mu_curr
    end

    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)

    batch_mapreduce!((x, xl, z) -> (x - xl) * z, +, zero(T), ws.sum_lb, x_lr, xl_r, zl_r)
    @. ws.mu_curr = ws.sum_lb / nlb
    return ws.mu_curr
end

function get_affine_complementarity_measure!(solver::AbstractBatchMPCSolver, alpha_p, alpha_d)
    ws = solver.workspace
    nlb = solver.d.nlb
    T = eltype(ws.mu_affine)

    if nlb == 0
        fill!(ws.mu_affine, zero(T))
        return ws.mu_affine
    end

    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    dx_lr = xp_lr(solver.d)
    dzlb  = MadNLP.dual_lb(solver.d)

    _affine_compl_lb!(ws.sum_lb, x_lr, xl_r, zl_r, dx_lr, dzlb, alpha_p, alpha_d)
    @. ws.mu_affine = ws.sum_lb / nlb
    return ws.mu_affine
end

function update_barrier!(::Mehrotra, solver::AbstractBatchMPCSolver, mu_affine)
    ws = solver.workspace
    T = eltype(ws.mu_curr)
    has_inequalities = solver.d.nlb > 0

    get_complementarity_measure!(solver)

    if has_inequalities
        @. ws.mu_batch = clamp((ws.mu_affine / ws.mu_curr) ^ 3, T(1e-6), T(10.0))
        @. ws.mu_batch = max(solver.opt.mu_min, ws.mu_batch * ws.mu_curr)
    else
        @. ws.mu_batch = max(solver.opt.mu_min, ws.mu_curr)
    end
    return
end

function get_fraction_to_boundary_step!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, xl = batch_solver.x, batch_solver.xl
    zl, d = batch_solver.zl, batch_solver.d
    nlb = d.nlb
    T = eltype(ws.alpha_p)

    if nlb > 0
        _ftb_primal_lb!(ws.alpha_xl, xp_lr(d), lower(x), lower(xl), ws.tau)
        _ftb_dual_lb!(ws.alpha_zl, MadNLP.dual_lb(d), lower(zl), ws.tau)
    else
        fill!(ws.alpha_xl, one(T))
        fill!(ws.alpha_zl, one(T))
    end

    ws.alpha_p .= min.(ws.alpha_xl, one(T))
    ws.alpha_d .= min.(ws.alpha_zl, one(T))
    return
end

function _ftb_primal_lb!(alpha_out, dx, x, xb, tau)
    T = eltype(alpha_out)
    n, bs = size(dx)
    @inbounds for j in 1:bs
        a = T(Inf)
        τ = tau[1, j]
        for i in 1:n
            d = dx[i, j]
            d < zero(T) || continue
            a = min(a, (-x[i, j] + xb[i, j]) * τ / d)
        end
        alpha_out[1, j] = a
    end
end

function _ftb_dual_lb!(alpha_out, dz, z, tau)
    T = eltype(alpha_out)
    n, bs = size(dz)
    @inbounds for j in 1:bs
        a = T(Inf)
        τ = tau[1, j]
        for i in 1:n
            d = dz[i, j]
            d < zero(T) || continue
            a = min(a, -z[i, j] * τ / d)
        end
        alpha_out[1, j] = a
    end
end

function _affine_compl_lb!(out, x, xl, z, dx, dz, αp, αd)
    T = eltype(out)
    n, bs = size(x)
    @inbounds for j in 1:bs
        s = zero(T)
        ap = αp[1, j]; ad = αd[1, j]
        for i in 1:n
            s += (x[i,j] + ap * dx[i,j] - xl[i,j]) * (z[i,j] + ad * dz[i,j])
        end
        out[1, j] = s
    end
end

function set_tau!(rule::ConservativeStep, batch_solver::AbstractBatchMPCSolver)
    fill!(batch_solver.workspace.tau, rule.tau)
end
function set_tau!(rule::AdaptiveStep, batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    ws.tau .= max.(1 .- ws.mu_batch, rule.tau_min)
end
function update_step!(rule::Union{ConservativeStep, AdaptiveStep}, batch_solver::AbstractBatchMPCSolver)
    set_tau!(rule, batch_solver)
    get_fraction_to_boundary_step!(batch_solver)
    return
end

function _mehrotra_step!(
    alpha_p, alpha_d, mu, gamma_f,
    dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
    d_vals, dlb_off,
)
    for j in axes(alpha_p, 2)
        _mehrotra_step_column!(
            j, alpha_p, alpha_d, mu[1, j], gamma_f,
            dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
            d_vals, dlb_off,
        )
    end
end

@inline function _mehrotra_step_column!(
    j, alpha_p, alpha_d, mu_j::T, gamma_f::T,
    dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
    d_vals, dlb_off,
) where T
    max_ap = alpha_p[1, j]
    max_ad = alpha_d[1, j]

    best_xl = T(Inf); i_xl = 0
    @inbounds for i in 1:nlb
        d = dx_lr[i, j]
        d < zero(T) || continue
        v = (xl_r[i, j] - x_lr[i, j]) / d
        v < best_xl && (best_xl = v; i_xl = i)
    end
    best_zl = T(Inf); i_zl = 0
    @inbounds for i in 1:nlb
        d = dzlb[i, j]
        d < zero(T) || continue
        v = -zl_r[i, j] / d
        v < best_zl && (best_zl = v; i_zl = i)
    end

    corrected_p = one(T)
    @inbounds if max_ap < one(T) && i_xl > 0
        zl_stepped = zl_r[i_xl, j] + max_ad * d_vals[dlb_off + i_xl, j]
        corrected_p = (x_lr[i_xl, j] - xl_r[i_xl, j] - mu_j / zl_stepped) / (-dx_lr[i_xl, j])
    end
    alpha_p[1, j] = max(corrected_p, gamma_f * max_ap)

    corrected_d = one(T)
    @inbounds if max_ad < one(T) && i_zl > 0
        x_gap = x_lr[i_zl, j] + max_ap * dx_lr[i_zl, j] - xl_r[i_zl, j]
        corrected_d = -(zl_r[i_zl, j] - mu_j / x_gap) / d_vals[dlb_off + i_zl, j]
    end
    alpha_d[1, j] = max(corrected_d, gamma_f * max_ad)
    return
end

function update_step!(rule::MehrotraAdaptiveStep, batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, xl = batch_solver.x, batch_solver.xl
    zl, d = batch_solver.zl, batch_solver.d
    nlb = d.nlb
    T = eltype(ws.alpha_p)
    gamma_f = T(rule.gamma_f)
    gamma_a = one(T) / (one(T) - gamma_f)

    fill!(ws.tau, one(T))
    get_fraction_to_boundary_step!(batch_solver)

    get_affine_complementarity_measure!(batch_solver, ws.alpha_p, ws.alpha_d)
    mu_full = ws.mu_curr
    @. mu_full = ws.mu_affine / gamma_a

    dlb_off = d.n + d.m

    _mehrotra_step!(
        ws.alpha_p, ws.alpha_d, mu_full, gamma_f,
        xp_lr(d), lower(x), lower(xl), nlb, MadNLP.dual_lb(d), lower(zl),
        d.values, dlb_off,
    )
    return
end

function _adjust_boundary_active!(x_lr::AbstractMatrix{T}, xl_r, mu, mask) where {T}
    c2 = eps(T)^(T(3)/T(4))
    c1 = eps(T)
    xl_r .= ifelse.(
        (mask .!= 0) .& (x_lr .- xl_r .< (c1 .* mu)),
        xl_r .- c2 .* max.(one(T), abs.(x_lr)),
        xl_r,
    )
end

function init_regularization!(solver::AbstractBatchMPCSolver, ::NoRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, 0.0)
end
update_regularization!(solver::AbstractBatchMPCSolver, reg) =
    update_regularization!(solver, reg, solver.workspace.active_mask)
function update_regularization!(solver::AbstractBatchMPCSolver, ::NoRegularization, mask)
    solver.del_w .= ifelse.(mask .== 1, 0.0, solver.del_w)
    solver.del_c .= ifelse.(mask .== 1, 0.0, solver.del_c)
end
function init_regularization!(solver::AbstractBatchMPCSolver, reg::FixedRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, reg.delta_d)
end
function update_regularization!(solver::AbstractBatchMPCSolver, reg::FixedRegularization, mask)
    solver.del_w .= ifelse.(mask .== 1, reg.delta_p, solver.del_w)
    solver.del_c .= ifelse.(mask .== 1, reg.delta_d, solver.del_c)
end
function init_regularization!(solver::AbstractBatchMPCSolver, reg::AdaptiveRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, reg.delta_d)
end
function update_regularization!(solver::AbstractBatchMPCSolver, reg::AdaptiveRegularization, mask)
    reg.delta_p = max(reg.delta_p / 10.0, reg.delta_min)
    reg.delta_d = min(reg.delta_d / 10.0, -reg.delta_min)
    solver.del_w .= ifelse.(mask .== 1, reg.delta_p, solver.del_w)
    solver.del_c .= ifelse.(mask .== 1, reg.delta_d, solver.del_c)
end
