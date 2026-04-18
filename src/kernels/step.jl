# Step-rule kernels (fraction-to-boundary + Mehrotra adaptive).
# Stays specialized: scalar uses scalar-state mapreduce; batch uses
# per-column kernels writing into matrix scratch.

# ---------- scalar ----------

function get_alpha_max(v::AbstractVector{T}, dv, tau::T) where {T}
    return mapreduce(
        (dvi, vi, i) -> ((dvi < 0 ? (-vi) * tau / dvi : Inf), i),
        (a, b) -> a[1] < b[1] ? a : b,
        dv, v, eachindex(v);
        init = (one(T), 0),
    )
end

function get_fraction_to_boundary_step(solver::MPCSolver, tau)
    state = solver.state
    alpha_x, _ = get_alpha_max(state.x_lr, state.dx_lr, tau)
    alpha_z, _ = get_alpha_max(state.zl_r, MadNLP.dual_lb(state.d), tau)
    return alpha_x, alpha_z
end

get_tau(rule::ConservativeStep, solver::MPCSolver) = rule.tau
get_tau(rule::AdaptiveStep, solver::MPCSolver) =
    max(one(typeof(solver.state.mu)) - solver.state.mu, typeof(solver.state.mu)(rule.tau_min))

function update_step!(rule::Union{ConservativeStep, AdaptiveStep}, solver::MPCSolver)
    state = solver.state
    tau = get_tau(rule, solver)
    state.alpha_p, state.alpha_d = get_fraction_to_boundary_step(solver, tau)
    return
end

function update_step!(rule::MehrotraAdaptiveStep, solver::MPCSolver)
    state = solver.state
    T = eltype(state.y)
    gamma_a = one(T) / (one(T) - T(rule.gamma_f))
    d_zl = MadNLP.dual_lb(state.d)
    alpha_x, i_x = get_alpha_max(state.x_lr, state.dx_lr, one(T))
    alpha_z, i_z = get_alpha_max(state.zl_r, d_zl, one(T))

    mu_full = get_affine_complementarity_measure(solver, alpha_x, alpha_z) / gamma_a
    alpha_p = one(T)
    alpha_d = one(T)

    if alpha_x < one(T)
        tmp = mu_full / (state.zl_r[i_x] + alpha_z * d_zl[i_x])
        alpha_p = (state.x_lr[i_x] - tmp) / (-state.dx_lr[i_x])
    end
    if alpha_z < one(T)
        tmp = mu_full / (state.x_lr[i_z] + alpha_x * state.dx_lr[i_z])
        alpha_d = -(state.zl_r[i_z] - tmp) / d_zl[i_z]
    end

    state.alpha_p = max(alpha_p, T(rule.gamma_f) * alpha_x)
    state.alpha_d = max(alpha_d, T(rule.gamma_f) * alpha_z)
    return
end

# ---------- batch ----------

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

function get_fraction_to_boundary_step!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, xl, xu = batch_solver.x, batch_solver.xl, batch_solver.xu
    zl, zu, d = batch_solver.zl, batch_solver.zu, batch_solver.d
    nlb, nub = d.nlb, d.nub
    T = eltype(ws.alpha_p)

    if nlb > 0
        _ftb_primal_lb!(ws.alpha_xl, xp_lr(d), lower(x), lower(xl), ws.tau)
        _ftb_dual_lb!(ws.alpha_zl, MadNLP.dual_lb(d), lower(zl), ws.tau)
    else
        fill!(ws.alpha_xl, one(T))
        fill!(ws.alpha_zl, one(T))
    end

    if nub > 0
        _ftb_primal_ub!(ws.alpha_xu, xp_ur(d), upper(x), upper(xu), ws.tau)
        _ftb_dual_ub!(ws.alpha_zu, MadNLP.dual_ub(d), upper(zu), ws.tau)
    else
        fill!(ws.alpha_xu, one(T))
        fill!(ws.alpha_zu, one(T))
    end

    ws.alpha_p .= min.(ws.alpha_xl, ws.alpha_xu, one(T))
    ws.alpha_d .= min.(ws.alpha_zl, ws.alpha_zu, one(T))
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

function _ftb_primal_ub!(alpha_out, dx, x, xb, tau)
    T = eltype(alpha_out)
    n, bs = size(dx)
    @inbounds for j in 1:bs
        a = T(Inf)
        τ = tau[1, j]
        for i in 1:n
            d = dx[i, j]
            d > zero(T) || continue
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

function _ftb_dual_ub!(alpha_out, dz, z, tau)
    T = eltype(alpha_out)
    n, bs = size(dz)
    @inbounds for j in 1:bs
        a = T(Inf)
        τ = tau[1, j]
        for i in 1:n
            d = dz[i, j]
            (d < zero(T) && z[i, j] + d < zero(T)) || continue
            a = min(a, -z[i, j] * τ / d)
        end
        alpha_out[1, j] = a
    end
end

function _mehrotra_step!(
    alpha_p, alpha_d, mu, gamma_f,
    dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
    dx_ur, x_ur, xu_r, nub, dzub, zu_r,
    d_vals, ind_lb, ind_ub, dlb_off, dub_off,
)
    for j in axes(alpha_p, 2)
        _mehrotra_step_column!(
            j, alpha_p, alpha_d, mu[1, j], gamma_f,
            dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
            dx_ur, x_ur, xu_r, nub, dzub, zu_r,
            d_vals, ind_lb, ind_ub, dlb_off, dub_off,
        )
    end
end

@inline function _mehrotra_step_column!(
    j, alpha_p, alpha_d, mu_j::T, gamma_f::T,
    dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
    dx_ur, x_ur, xu_r, nub, dzub, zu_r,
    d_vals, ind_lb, ind_ub, dlb_off, dub_off,
) where T
    max_ap = alpha_p[1, j]
    max_ad = alpha_d[1, j]

    # primal lb
    best_xl = T(Inf); i_xl = 0
    @inbounds for i in 1:nlb
        d = dx_lr[i, j]
        d < zero(T) || continue
        v = (xl_r[i, j] - x_lr[i, j]) / d
        v < best_xl && (best_xl = v; i_xl = i)
    end
    # primal ub
    best_xu = T(Inf); i_xu = 0
    @inbounds for i in 1:nub
        d = dx_ur[i, j]
        d > zero(T) || continue
        v = (xu_r[i, j] - x_ur[i, j]) / d
        v < best_xu && (best_xu = v; i_xu = i)
    end
    # dual lb
    best_zl = T(Inf); i_zl = 0
    @inbounds for i in 1:nlb
        d = dzlb[i, j]
        d < zero(T) || continue
        v = -zl_r[i, j] / d
        v < best_zl && (best_zl = v; i_zl = i)
    end
    # dual ub
    best_zu = T(Inf); i_zu = 0
    @inbounds for i in 1:nub
        d = dzub[i, j]
        (d < zero(T) && zu_r[i, j] + d < zero(T)) || continue
        v = -zu_r[i, j] / d
        v < best_zu && (best_zu = v; i_zu = i)
    end

    # primal step
    corrected_p = one(T)
    @inbounds if max_ap < one(T)
        if best_xl <= best_xu && i_xl > 0
            zl_stepped = zl_r[i_xl, j] + max_ad * d_vals[dlb_off + i_xl, j]
            corrected_p = (x_lr[i_xl, j] - xl_r[i_xl, j] - mu_j / zl_stepped) / (-dx_lr[i_xl, j])
        elseif i_xu > 0
            zu_stepped = zu_r[i_xu, j] + max_ad * d_vals[dub_off + i_xu, j]
            corrected_p = (xu_r[i_xu, j] - x_ur[i_xu, j] - mu_j / zu_stepped) / dx_ur[i_xu, j]
        end
    end
    alpha_p[1, j] = max(corrected_p, gamma_f * max_ap)

    # dual step
    corrected_d = one(T)
    @inbounds if max_ad < one(T)
        if best_zl <= best_zu && i_zl > 0
            x_gap = x_lr[i_zl, j] + max_ap * dx_lr[i_zl, j] - xl_r[i_zl, j]
            corrected_d = -(zl_r[i_zl, j] - mu_j / x_gap) / d_vals[dlb_off + i_zl, j]
        elseif i_zu > 0
            x_gap = xu_r[i_zu, j] - x_ur[i_zu, j] - max_ap * dx_ur[i_zu, j]
            corrected_d = -(zu_r[i_zu, j] - mu_j / x_gap) / d_vals[dub_off + i_zu, j]
        end
    end
    alpha_d[1, j] = max(corrected_d, gamma_f * max_ad)
    return
end

function update_step!(rule::MehrotraAdaptiveStep, batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, xl, xu = batch_solver.x, batch_solver.xl, batch_solver.xu
    zl, zu, d = batch_solver.zl, batch_solver.zu, batch_solver.d
    nlb, nub = d.nlb, d.nub
    T = eltype(ws.alpha_p)
    gamma_f = T(rule.gamma_f)
    gamma_a = one(T) / (one(T) - gamma_f)

    fill!(ws.tau, one(T))
    get_fraction_to_boundary_step!(batch_solver)

    get_affine_complementarity_measure!(batch_solver, ws.alpha_p, ws.alpha_d)
    mu_full = ws.mu_curr
    @. mu_full = ws.mu_affine / gamma_a

    dlb_off = d.n + d.m
    dub_off = d.n + d.m + d.nlb
    bs = batch_solver.batch_size

    _mehrotra_step!(
        ws.alpha_p, ws.alpha_d, mu_full, gamma_f,
        xp_lr(d), lower(x), lower(xl), nlb, MadNLP.dual_lb(d), lower(zl),
        xp_ur(d), upper(x), upper(xu), nub, MadNLP.dual_ub(d), upper(zu),
        d.values, d.ind_lb, d.ind_ub, dlb_off, dub_off,
    )
    return
end

# FIXME: make it a kernel
function _adjust_boundary_active!(x_lr::AbstractMatrix{T}, xl_r, x_ur, xu_r, mu, mask) where {T}
    c2 = eps(T)^(T(3)/T(4))
    c1 = eps(T)
    xl_r .= ifelse.(
        (mask .!= 0) .& (x_lr .- xl_r .< (c1 .* mu)),
        xl_r .- c2 .* max.(one(T), abs.(x_lr)),
        xl_r,
    )
    xu_r .= ifelse.(
        (mask .!= 0) .& (xu_r .- x_ur .< (c1 .* mu)),
        xu_r .+ c2 .* max.(one(T), abs.(x_ur)),
        xu_r,
    )
end
