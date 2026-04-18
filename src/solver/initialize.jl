# Solver initialization: warm-start primal/dual variables and prepare
# scratch state. The first half (factorize + two LS solves + initial zl
# from A'y + c) is unified across scalar and batch via accessors. The
# second half — Mehrotra's per-instance shifts (δx / δz) — stays
# specialized because scalar uses scalar `min` / `dot` / `sum` while batch
# uses per-column `batch_mapreduce!`.

function _init_starting_point_solve!(s::AnyMPCSolver)
    kkt = _kkt(s)
    kkt.reg .= _del_w(s)
    pr_diag(kkt) .= _del_w(s)
    du_diag(kkt) .= _del_c(s)

    MadNLP.factorize_wrapper!(s)

    set_initial_primal_rhs!(s)
    solve_system!(_d(s), s, _p(s))
    MadNLP.primal(_x(s)) .+= MadNLP.primal(_d(s))

    set_initial_dual_rhs!(s)
    solve_system!(_d(s), s, _p(s))
    _y(s) .= MadNLP.dual(_d(s))

    res = _jacl(s)
    MadNLP.jtprod!(res, kkt, s.state.y)
    res .+= MadNLP.primal(_f(s))
    MadNLP.full(_zl(s)) .= res
    return
end

# ---------- scalar ----------

function init_starting_point!(solver::MPCSolver)
    state = solver.state
    T = eltype(state.y)
    _init_starting_point_solve!(solver)
    x = MadNLP.primal(state.x)
    z = state.zl_r

    delta_x = max(zero(T), -T(1.5) * minimum(x; init = zero(T)))
    delta_z = max(zero(T), -T(1.5) * minimum(z; init = zero(T)))

    x .+= delta_x
    z .+= one(T) + delta_z

    μ = isempty(z) ? zero(eltype(z)) : dot(x, z)
    sumz = sum(z)
    sumx = sum(x)
    delta_x2 = iszero(sumz) ? zero(eltype(z)) : μ / (T(2) * sumz)
    delta_z2 = iszero(sumx) ? zero(eltype(z)) : μ / (T(2) * sumx)

    x .+= delta_x2
    z .+= delta_z2
    return
end

function initialize!(solver::MPCSolver{T}) where {T}
    problem = solver.problem
    state = solver.state
    x = MadNLP.variable(state.x)
    x .= max.(NLPModels.get_x0(problem.nlp), T(problem.opt.bound_push))
    state.y .= NLPModels.get_y0(problem.nlp)
    state.rhs .= NLPModels.get_lcon(problem.nlp)
    fill!(state.jacl, zero(T))

    MadNLP.initialize!(problem.kkt)
    init_regularization!(solver, problem.regularization)

    state.obj_val = MadNLP.eval_f_wrapper(solver, state.x)
    MadNLP.eval_jac_wrapper!(solver, problem.kkt, state.x)
    MadNLP.eval_grad_f_wrapper!(solver, state.f, state.x)
    MadNLP.eval_cons_wrapper!(solver, state.c, state.x)
    MadNLP.eval_lag_hess_wrapper!(solver, problem.kkt, state.x, state.y)

    state.norm_b = norm(state.rhs, Inf)
    state.norm_c = norm(MadNLP.primal(state.f), Inf)

    init_starting_point!(solver)

    state.mu = T(problem.opt.mu_init)
    state.best_complementarity = typemax(typeof(state.best_complementarity))
    state.status = MadNLP.REGULAR
    MadNLP.jtprod!(state.jacl, problem.kkt, state.y)
    return
end

# ---------- batch ----------

function init_starting_point!(batch_solver::AbstractBatchMPCSolver{T}) where T
    state = batch_solver.state
    _init_starting_point_solve!(batch_solver)
    x = MadNLP.primal(state.x)             # (n_tot, bs)
    z = lower(state.zl)                    # (nlb=n_tot, bs)
    ws = state.workspace
    delta_x  = ws.alpha_xl   # (1, bs) scratch
    delta_z  = ws.alpha_xu
    sumz     = ws.sum_lb
    sumx     = ws.sum_ub
    μ        = ws.mu_curr
    delta_x2 = ws.alpha_zl
    delta_z2 = ws.alpha_zu

    batch_mapreduce!(identity, min, T(Inf), delta_x, x)
    batch_mapreduce!(identity, min, T(Inf), delta_z, z)
    @. delta_x = max(zero(T), T(-1.5) * delta_x)
    @. delta_z = max(zero(T), T(-1.5) * delta_z)

    x .+= delta_x
    z .+= one(T) .+ delta_z

    batch_mapreduce!(*, +, zero(T), μ, x, z)
    batch_mapreduce!(identity, +, zero(T), sumz, z)
    batch_mapreduce!(identity, +, zero(T), sumx, x)
    @. delta_x2 = ifelse(iszero(sumz), zero(T), μ / (T(2) * sumz))
    @. delta_z2 = ifelse(iszero(sumx), zero(T), μ / (T(2) * sumx))

    x .+= delta_x2
    z .+= delta_z2
    return
end

function initialize!(batch_solver::AbstractBatchMPCSolver{T}) where T
    problem = batch_solver.problem
    state = batch_solver.state
    opt = problem.opt
    bcb = problem.bcb
    ws = state.workspace

    # Mirrors scalar `initialize!`: simple `max(get_x0, bound_push)` projection,
    # populate y/rhs from the callback, zero jacl. Std form's lvar=0/uvar=Inf
    # are written into the batch xl/xu matrices (scalar holds them implicitly).
    x_full = MadNLP.full(state.x)
    x_full .= max.(MadNLP.get_x0(bcb), T(opt.bound_push))
    MadNLP.full(state.xl) .= MadNLP.get_lvar(bcb)
    MadNLP.full(state.xu) .= MadNLP.get_uvar(bcb)
    MadNLP.full(state.y) .= MadNLP.get_y0(bcb)
    MadNLP.full(state.rhs) .= MadNLP.get_lcon(bcb)
    fill!(MadNLP.full(state.jacl), zero(T))

    MadNLP.initialize!(problem.kkt)
    init_regularization!(batch_solver, problem.regularization)

    MadNLP.unpack_x!(ws.bx, bcb, state.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_jac_wrapper!(batch_solver, problem.kkt)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, problem.kkt)

    batch_mapreduce!(abs, max, typemin(T), ws.norm_b, MadNLP.full(state.rhs))
    batch_mapreduce!(abs, max, typemin(T), ws.norm_c, MadNLP.full(state.f))

    init_starting_point!(batch_solver)
    initialize_solver_state!(batch_solver)

    MadNLP.jtprod!(state.jacl, problem.kkt, state.y)
    return
end

function initialize_solver_state!(batch_solver::AbstractBatchMPCSolver{T}) where T
    problem = batch_solver.problem
    state = batch_solver.state
    ws = state.workspace
    fill!(ws.mu_batch, problem.opt.mu_init)
    fill!(ws.best_complementarity, typemax(T))
    fill!(ws.status, MadNLP.REGULAR)
    reset_active_view!(problem.batch_views)
    _update_active_mask!(batch_solver)
    fill!(ws.inf_pr, zero(T))
    fill!(ws.inf_du, zero(T))
    fill!(ws.inf_compl, zero(T))
    fill!(ws.dual_obj, zero(T))
    fill!(ws.alpha_p, zero(T))
    fill!(ws.alpha_d, zero(T))
    t_now = time()
    bcnt = state.batch_cnt
    bcnt.start_time[] = t_now
    fill!(bcnt.k, 0)
    bcnt.linear_solver_time[] = 0.0
    bcnt.eval_function_time[] = 0.0
    bcnt.obj_cnt[] = 0
    bcnt.obj_grad_cnt[] = 0
    bcnt.con_cnt[] = 0
    return
end
