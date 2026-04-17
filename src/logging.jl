function MadNLP.print_iter(solver::MPCSolver; options...)
    problem = solver.problem
    state = solver.state
    T = typeof(state.obj_val)
    obj_scale = problem.cb.obj_scale[]
    mod(state.cnt.k, 10) == 0 && MadNLP.@info(problem.logger, @sprintf(
        "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr"))
    MadNLP.@info(problem.logger, Printf.@sprintf(
        "%4i%s% 10.7e %6.2e %6.2e %5.1f %6.2e %s %6.2e %6.2e",
        state.cnt.k,
        " ",
        state.obj_val / obj_scale,
        state.inf_pr, state.inf_du, log10(state.mu),
        state.cnt.k == 0 ? zero(T) : norm(MadNLP.primal(state.d), Inf),
        iszero(state.del_w) ? "   - " : @sprintf("%5.1f", log(10, state.del_w)),
        state.alpha_d, state.alpha_p))
    return
end

function MadNLP.print_summary(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    obj_scale = problem.cb.obj_scale[]
    state.cnt.solver_time = state.cnt.total_time - state.cnt.linear_solver_time - state.cnt.eval_function_time

    MadNLP.@notice(problem.logger, "")
    MadNLP.@notice(problem.logger, "Number of Iterations....: $(state.cnt.k)\n")
    MadNLP.@notice(problem.logger, "                                   (scaled)                 (unscaled)")
    MadNLP.@notice(problem.logger, @sprintf("Objective...............:  % 1.16e   % 1.16e", state.obj_val, state.obj_val / obj_scale))
    MadNLP.@notice(problem.logger, @sprintf("Dual infeasibility......:   %1.16e    %1.16e", state.inf_du, state.inf_du / obj_scale))
    MadNLP.@notice(problem.logger, @sprintf("Constraint violation....:   %1.16e    %1.16e", norm(state.c, Inf), state.inf_pr))
    MadNLP.@notice(problem.logger, @sprintf("Complementarity.........:   %1.16e    %1.16e", state.inf_compl * obj_scale, state.inf_compl))
    MadNLP.@notice(problem.logger, @sprintf("Overall NLP error.......:   %1.16e    %1.16e\n", max(state.inf_du * obj_scale, norm(state.c, Inf), state.inf_compl), max(state.inf_du, state.inf_pr, state.inf_compl)))

    MadNLP.@notice(problem.logger, "Number of objective function evaluations              = $(state.cnt.obj_cnt)")
    MadNLP.@notice(problem.logger, "Number of objective gradient evaluations              = $(state.cnt.obj_grad_cnt)")
    MadNLP.@notice(problem.logger, "Number of constraint evaluations                      = $(state.cnt.con_cnt)")
    MadNLP.@notice(problem.logger, "Number of constraint Jacobian evaluations             = $(state.cnt.con_jac_cnt)")
    MadNLP.@notice(problem.logger, "Number of Lagrangian Hessian evaluations              = $(state.cnt.lag_hess_cnt)")
    MadNLP.@notice(problem.logger, "Number of KKT factorizations                          = $(state.cnt.factorization_cnt)")
    MadNLP.@notice(problem.logger, "Number of KKT backsolves                              = $(state.cnt.backsolve_cnt)\n")
    MadNLP.@notice(problem.logger, "Total wall secs in initialization                     = $(MadNLP.format_time(state.cnt.init_time))")
    MadNLP.@notice(problem.logger, "Total wall secs in linear solver                      = $(MadNLP.format_time(state.cnt.linear_solver_time))")
    MadNLP.@notice(problem.logger, "Total wall secs in NLP function evaluations           = $(MadNLP.format_time(state.cnt.eval_function_time))")
    MadNLP.@notice(problem.logger, "Total wall secs in solver (w/o init./fun./lin. alg.)  = $(MadNLP.format_time(state.cnt.total_time - state.cnt.init_time - state.cnt.linear_solver_time - state.cnt.eval_function_time))")
    MadNLP.@notice(problem.logger, "Total wall secs                                       = $(MadNLP.format_time(state.cnt.total_time))\n")
    return
end
