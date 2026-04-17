function init_starting_point!(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    T = eltype(state.y)
    x = MadNLP.primal(state.x)
    z = state.zl_r
    res = state.jacl

    problem.kkt.reg .= state.del_w
    problem.kkt.pr_diag .= state.del_w
    problem.kkt.du_diag .= state.del_c

    MadNLP.factorize_wrapper!(solver)

    set_initial_primal_rhs!(solver)
    solve_system!(state.d, solver, state.p)
    axpy!(one(T), MadNLP.primal(state.d), x)

    set_initial_dual_rhs!(solver)
    solve_system!(state.d, solver, state.p)
    state.y .= MadNLP.dual(state.d)

    MadNLP.jtprod!(res, problem.kkt, state.y)
    axpy!(one(T), MadNLP.primal(state.f), res)
    copyto!(state.zl.values, res)

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

function update_termination_criteria!(solver::MPCSolver{T}) where {T}
    problem = solver.problem
    state = solver.state
    dobj = -dot(state.y, state.rhs)
    state.inf_pr = MadNLP.get_inf_pr(state.c) / max(one(T), state.norm_b)
    state.inf_du = norm(MadNLP.primal(state.f) .+ state.jacl .- MadNLP.full(state.zl), Inf) / max(one(T), state.norm_c)
    state.inf_compl = _xz_sum(solver) / max(one(T), state.norm_c)
    state.best_complementarity = min(state.best_complementarity, state.inf_compl)

    if max(state.inf_pr, state.inf_du, state.inf_compl) <= problem.opt.tol
        state.status = MadNLP.SOLVE_SUCCEEDED
    elseif ((state.inf_compl > problem.opt.divergence_tol * state.best_complementarity) &&
            (dobj > max(problem.opt.divergence_scale * abs(state.obj_val), one(T))))
        state.status = MadNLP.INFEASIBLE_PROBLEM_DETECTED
    elseif state.obj_val < -problem.opt.divergence_tol * max(problem.opt.divergence_scale * abs(dobj), one(T))
        state.status = MadNLP.DIVERGING_ITERATES
    elseif state.cnt.k >= problem.opt.max_iter
        state.status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    elseif time() - state.cnt.start_time >= problem.opt.max_wall_time
        state.status = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
    end
    return
end

function affine_direction!(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    set_predictive_rhs!(solver, problem.kkt)
    solve_system!(state.d, solver, state.p)
    return
end

function prediction_step!(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    affine_direction!(solver)
    alpha_aff_p, alpha_aff_d = get_fraction_to_boundary_step(solver, one(eltype(state.y)))
    mu_affine = get_affine_complementarity_measure(solver, alpha_aff_p, alpha_aff_d)
    get_correction!(solver, state.correction_lb)
    state.mu_curr = update_barrier!(problem.barrier_update, solver, mu_affine)
    return
end

function mehrotra_correction_direction!(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    set_correction_rhs!(solver, problem.kkt, state.mu, state.correction_lb)
    solve_system!(state.d, solver, state.p)
    return
end

function apply_step!(solver::MPCSolver)
    state = solver.state
    axpy!(state.alpha_p, MadNLP.primal(state.d), MadNLP.primal(state.x))
    axpy!(state.alpha_d, MadNLP.dual(state.d), state.y)
    state.zl_r .+= state.alpha_d .* MadNLP.dual_lb(state.d)
    state.cnt.k += 1
    return
end

function evaluate_model!(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    state.obj_val = MadNLP.eval_f_wrapper(solver, state.x)
    MadNLP.eval_cons_wrapper!(solver, state.c, state.x)
    MadNLP.eval_grad_f_wrapper!(solver, state.f, state.x)
    MadNLP.jtprod!(state.jacl, problem.kkt, state.y)
    return
end

function MadNLP.MadNLPExecutionStats(solver::MPCSolver{T}) where {T}
    problem = solver.problem
    state = solver.state
    n = NLPModels.get_nvar(problem.original_nlp)
    m = NLPModels.get_ncon(problem.original_nlp)
    x_template = NLPModels.get_x0(problem.original_nlp)
    y_template = NLPModels.get_y0(problem.original_nlp)
    solution = similar(x_template, n)
    constraints = similar(y_template, m)
    multipliers = similar(y_template, m)
    multipliers_L = similar(x_template, n)
    multipliers_U = similar(x_template, n)
    return MadNLP.MadNLPExecutionStats(
        problem.opt,
        state.status,
        solution,
        zero(T),
        constraints,
        zero(T),
        zero(T),
        multipliers,
        multipliers_L,
        multipliers_U,
        0,
        state.cnt,
    )
end

function update_solution!(stats::MadNLP.MadNLPExecutionStats, solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    stats.status = state.status
    recover_primal!(stats.solution, problem.workspace, MadNLP.variable(state.x))
    BatchQuadraticModels._gather_dual!(stats.multipliers, problem.workspace.con_start.row, state.y)
    recover_variable_multipliers!(stats.multipliers_L, stats.multipliers_U, problem.workspace, MadNLP.variable(state.zl))
    NLPModels.cons!(problem.original_nlp, stats.solution, stats.constraints)
    stats.objective = NLPModels.obj(problem.original_nlp, stats.solution)
    stats.dual_feas = state.inf_du
    stats.primal_feas = state.inf_pr
    stats.iter = state.cnt.k
    return stats
end

function mpc_step!(solver::MPCSolver)
    update_regularization!(solver, solver.problem.regularization)
    factorize_regularized_system!(solver)
    prediction_step!(solver)
    mehrotra_correction_direction!(solver)
    update_step!(solver.problem.step_rule, solver)
    apply_step!(solver)
    evaluate_model!(solver)
    return
end

function mpc!(solver::MPCSolver)
    while true
        MadNLP.print_iter(solver)
        update_termination_criteria!(solver)
        solver.state.status != MadNLP.REGULAR && return
        mpc_step!(solver)
    end
end

solve!(solver::MPCSolver) = solve!(solver, MadNLP.MadNLPExecutionStats(solver))

function solve!(solver::MPCSolver, stats::MadNLP.MadNLPExecutionStats)
    problem = solver.problem
    state = solver.state
    try
        MadNLP.@notice(problem.logger, "This is MadIPM, running with $(MadNLP.introduce(problem.kkt.linear_solver))\n")
        initialize!(solver)
        mpc!(solver)
    catch e
        if e isa MadNLP.InvalidNumberException
            if e.callback == :obj
                state.status = MadNLP.INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                state.status = MadNLP.INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                state.status = MadNLP.INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                state.status = MadNLP.INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                state.status = MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                state.status = MadNLP.INVALID_NUMBER_DETECTED
            end
        elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
            state.status = MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa MadNLP.LinearSolverException
            state.status = MadNLP.ERROR_IN_STEP_COMPUTATION
            problem.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            state.status = MadNLP.USER_REQUESTED_STOP
            problem.opt.rethrow_error && rethrow(e)
        else
            state.status = MadNLP.INTERNAL_ERROR
            problem.opt.rethrow_error && rethrow(e)
        end
    finally
        state.cnt.total_time = time() - state.cnt.start_time
        if !(state.status < MadNLP.SOLVE_SUCCEEDED)
            MadNLP.print_summary(solver)
        end
        MadNLP.@notice(problem.logger, "EXIT: $(MadNLP.get_status_output(state.status, problem.opt))")
        finalize(problem.logger)
        update_solution!(stats, solver)
    end

    return stats
end

function madipm(m; kwargs...)
    solver = MadIPM.MPCSolver(m; kwargs...)
    return MadIPM.solve!(solver)
end
