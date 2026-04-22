# ---------- complementarity / barrier update (scalar) ----------
# `μ = ⟨x_L, z_L⟩ / nlb` is the IPM's central-path parameter. The solver
# tracks both the current iterate's `μ_curr` and the predictor-step's
# `μ_affine`; Mehrotra shrinks toward `(μ_affine / μ_curr)^3 * μ_curr`.

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
    return
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



# ---------- complementarity / barrier update (batch) ----------
# Mirrors the scalar path but stores per-instance `mu_curr`, `mu_affine`,
# `mu_batch` as `(1, nbatch)` matrices so the broadcasts stay on-backend.

function get_complementarity_measure!(solver::UniformBatchMPCSolver)
    state = solver.state
    ws = state.workspace
    nlb, nub = state.d.nlb, state.d.nub
    T = eltype(ws.mu_curr)

    if nlb + nub == 0
        fill!(ws.mu_curr, zero(T))
        return ws.mu_curr
    end

    xl_r = lower(state.xl)
    x_lr = lower(state.x)
    zl_r = lower(state.zl)
    xu_r = upper(state.xu)
    x_ur = upper(state.x)
    zu_r = upper(state.zu)

    batch_mapreduce!((x, xl, z) -> (x - xl) * z, +, zero(T), ws.sum_lb, x_lr, xl_r, zl_r)
    batch_mapreduce!((xu, x, z) -> (xu - x) * z, +, zero(T), ws.sum_ub, xu_r, x_ur, zu_r)
    @. ws.mu_curr = (ws.sum_lb + ws.sum_ub) / (nlb + nub)
    return ws.mu_curr
end

function get_affine_complementarity_measure!(solver::UniformBatchMPCSolver, alpha_p, alpha_d)
    state = solver.state
    ws = state.workspace
    nlb, nub = state.d.nlb, state.d.nub
    T = eltype(ws.mu_affine)

    if nlb + nub == 0
        fill!(ws.mu_affine, zero(T))
        return ws.mu_affine
    end

    xl_r = lower(state.xl)
    x_lr = lower(state.x)
    zl_r = lower(state.zl)
    xu_r = upper(state.xu)
    x_ur = upper(state.x)
    zu_r = upper(state.zu)
    dx_lr = xp_lr(state.d)
    dx_ur = xp_ur(state.d)
    dzlb = MadNLP.dual_lb(state.d)
    dzub = MadNLP.dual_ub(state.d)

    _affine_compl_lb!(ws.sum_lb, x_lr, xl_r, zl_r, dx_lr, dzlb, alpha_p, alpha_d)
    _affine_compl_ub!(ws.sum_ub, xu_r, x_ur, zu_r, dx_ur, dzub, alpha_p, alpha_d)
    @. ws.mu_affine = (ws.sum_lb + ws.sum_ub) / (nlb + nub)
    return ws.mu_affine
end

# CPU affine-step complementarity scans (batched). `lb` case uses the
# `(x - xl)` primal slack; `ub` case uses `(xu - x)`. GPU ext overrides
# both with KA kernels since SubArray-of-CuMatrix can't be scalar-indexed.
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

function update_barrier!(::Mehrotra, solver::UniformBatchMPCSolver, mu_affine)
    state = solver.state
    ws = state.workspace
    T = eltype(ws.mu_curr)

    has_inequalities = (state.d.nlb + state.d.nub) > 0

    get_complementarity_measure!(solver)

    mu_min = solver.problem.opt.mu_min
    if has_inequalities
        @. ws.mu_batch = clamp((ws.mu_affine / ws.mu_curr) ^ 3, T(1e-6), T(10.0))
        @. ws.mu_batch = max(mu_min, ws.mu_batch * ws.mu_curr)
    else
        @. ws.mu_batch = max(mu_min, ws.mu_curr)
    end
    return
end
