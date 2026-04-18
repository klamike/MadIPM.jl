# ---------- scalar ----------

function _xz_sum(solver::MPCSolver)
    x = solver.state.x_lr
    isempty(x) && return zero(eltype(x))
    return mapreduce(*, +, x, solver.state.zl_r; init = zero(eltype(x)))
end

get_complementarity_measure(solver::MPCSolver) =
    isempty(solver.state.x_lr) ? zero(eltype(solver.state.y)) :
        _xz_sum(solver) / length(solver.state.x_lr)

function update_barrier!(rule::Mehrotra, solver::MPCSolver{T}, mu_affine) where {T}
    problem = solver.problem
    state   = solver.state
    mu_curr = get_complementarity_measure(solver)
    sigma = if problem.nlb > 0
        iszero(mu_curr) ? one(T) : clamp((mu_affine / mu_curr)^3, T(1e-6), T(10))
    else
        one(T)
    end
    state.mu = max(T(problem.opt.mu_min), sigma * mu_curr)
    return mu_curr
end

function get_affine_complementarity_measure(solver::MPCSolver, alpha_p, alpha_d)
    state = solver.state
    isempty(state.x_lr) && return zero(eltype(state.x_lr))
    return mapreduce(
        (x, dx, z, dz) -> (x + alpha_p * dx) * (z + alpha_d * dz),
        +,
        state.x_lr, state.dx_lr, state.zl_r, MadNLP.dual_lb(state.d);
        init = zero(eltype(state.x_lr)),
    ) / length(state.x_lr)
end

# ---------- batch ----------

function get_complementarity_measure!(solver::AbstractBatchMPCSolver)
    ws = solver.workspace
    nlb, nub = solver.d.nlb, solver.d.nub
    T = eltype(ws.mu_curr)

    if nlb + nub == 0
        fill!(ws.mu_curr, zero(T))
        return ws.mu_curr
    end

    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    xu_r = upper(solver.xu)
    x_ur = upper(solver.x)
    zu_r = upper(solver.zu)

    batch_mapreduce!((x, xl, z) -> (x - xl) * z, +, zero(T), ws.sum_lb, x_lr, xl_r, zl_r)
    batch_mapreduce!((xu, x, z) -> (xu - x) * z, +, zero(T), ws.sum_ub, xu_r, x_ur, zu_r)
    @. ws.mu_curr = (ws.sum_lb + ws.sum_ub) / (nlb + nub)
    return ws.mu_curr
end

function get_affine_complementarity_measure!(solver::AbstractBatchMPCSolver, alpha_p, alpha_d)
    ws = solver.workspace
    nlb, nub = solver.d.nlb, solver.d.nub
    T = eltype(ws.mu_affine)

    if nlb + nub == 0
        fill!(ws.mu_affine, zero(T))
        return ws.mu_affine
    end

    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    xu_r = upper(solver.xu)
    x_ur = upper(solver.x)
    zu_r = upper(solver.zu)
    dx_lr = xp_lr(solver.d)
    dx_ur = xp_ur(solver.d)
    dzlb = MadNLP.dual_lb(solver.d)
    dzub = MadNLP.dual_ub(solver.d)

    _affine_compl_lb!(ws.sum_lb, x_lr, xl_r, zl_r, dx_lr, dzlb, alpha_p, alpha_d)
    _affine_compl_ub!(ws.sum_ub, xu_r, x_ur, zu_r, dx_ur, dzub, alpha_p, alpha_d)
    @. ws.mu_affine = (ws.sum_lb + ws.sum_ub) / (nlb + nub)
    return ws.mu_affine
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

function _affine_compl_ub!(out, xu, x, z, dx, dz, αp, αd)
    T = eltype(out)
    n, bs = size(x)
    @inbounds for j in 1:bs
        s = zero(T)
        ap = αp[1, j]; ad = αd[1, j]
        for i in 1:n
            s += (xu[i,j] - (x[i,j] + ap * dx[i,j])) * (z[i,j] + ad * dz[i,j])
        end
        out[1, j] = s
    end
end

function update_barrier!(::Mehrotra, solver::AbstractBatchMPCSolver, mu_affine)
    ws = solver.workspace
    T = eltype(ws.mu_curr)

    has_inequalities = (solver.d.nlb + solver.d.nub) > 0

    get_complementarity_measure!(solver)

    if has_inequalities
        @. ws.mu_batch = clamp((ws.mu_affine / ws.mu_curr) ^ 3, T(1e-6), T(10.0))
        @. ws.mu_batch = max(solver.opt.mu_min, ws.mu_batch * ws.mu_curr)
    else
        @. ws.mu_batch = max(solver.opt.mu_min, ws.mu_curr)
    end
    return
end
