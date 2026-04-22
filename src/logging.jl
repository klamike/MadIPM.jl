# ============================================================================
# Per-iteration and end-of-solve log formatting (scalar).
# Batch equivalents live in `solver/entry.jl` next to the batch `solve!`.
# ============================================================================

function MadNLP.print_iter(solver::MPCSolver; options...)
    state, problem = solver.state, solver.problem
    logger         = problem.logger
    obj_scale      = problem.cb.obj_scale[]
    T              = typeof(state.obj_val)

    mod(state.cnt.k, 10) == 0 && MadNLP.@info(logger,
        @sprintf("iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr"))

    d_norm = state.cnt.k == 0 ? zero(T) : norm(MadNLP.primal(state.d), Inf)
    rg_str = iszero(state.del_w) ? "   - " : @sprintf("%5.1f", log10(state.del_w))

    MadNLP.@info(logger, Printf.@sprintf(
        "%4i %s% 10.7e %6.2e %6.2e %5.1f %6.2e %s %6.2e %6.2e",
        state.cnt.k, " ",
        state.obj_val / obj_scale,
        state.inf_pr, state.inf_du, log10(state.mu),
        d_norm, rg_str,
        state.alpha_d, state.alpha_p))
    return nothing
end

function MadNLP.print_summary(solver::MPCSolver)
    state, problem = solver.state, solver.problem
    logger         = problem.logger
    obj_scale      = problem.cb.obj_scale[]
    cnt            = state.cnt
    cnt.solver_time = cnt.total_time - cnt.linear_solver_time - cnt.eval_function_time

    overall_scaled   = max(state.inf_du * obj_scale, norm(state.c, Inf), state.inf_compl)
    overall_unscaled = max(state.inf_du, state.inf_pr, state.inf_compl)

    MadNLP.@notice(logger, "")
    MadNLP.@notice(logger, "Number of Iterations....: $(cnt.k)\n")
    MadNLP.@notice(logger, "                                   (scaled)                 (unscaled)")
    MadNLP.@notice(logger, @sprintf("Objective...............:  % 1.16e   % 1.16e",
        state.obj_val, state.obj_val / obj_scale))
    MadNLP.@notice(logger, @sprintf("Dual infeasibility......:   %1.16e    %1.16e",
        state.inf_du, state.inf_du / obj_scale))
    MadNLP.@notice(logger, @sprintf("Constraint violation....:   %1.16e    %1.16e",
        norm(state.c, Inf), state.inf_pr))
    MadNLP.@notice(logger, @sprintf("Complementarity.........:   %1.16e    %1.16e",
        state.inf_compl * obj_scale, state.inf_compl))
    MadNLP.@notice(logger, @sprintf("Overall NLP error.......:   %1.16e    %1.16e\n",
        overall_scaled, overall_unscaled))

    MadNLP.@notice(logger, "Number of objective function evaluations              = $(cnt.obj_cnt)")
    MadNLP.@notice(logger, "Number of objective gradient evaluations              = $(cnt.obj_grad_cnt)")
    MadNLP.@notice(logger, "Number of constraint evaluations                      = $(cnt.con_cnt)")
    MadNLP.@notice(logger, "Number of constraint Jacobian evaluations             = $(cnt.con_jac_cnt)")
    MadNLP.@notice(logger, "Number of Lagrangian Hessian evaluations              = $(cnt.lag_hess_cnt)")
    MadNLP.@notice(logger, "Number of KKT factorizations                          = $(cnt.factorization_cnt)")
    MadNLP.@notice(logger, "Number of KKT backsolves                              = $(cnt.backsolve_cnt)\n")

    other = cnt.total_time - cnt.init_time - cnt.linear_solver_time - cnt.eval_function_time
    MadNLP.@notice(logger, "Total wall secs in initialization                     = $(MadNLP.format_time(cnt.init_time))")
    MadNLP.@notice(logger, "Total wall secs in linear solver                      = $(MadNLP.format_time(cnt.linear_solver_time))")
    MadNLP.@notice(logger, "Total wall secs in NLP function evaluations           = $(MadNLP.format_time(cnt.eval_function_time))")
    MadNLP.@notice(logger, "Total wall secs in solver (w/o init./fun./lin. alg.)  = $(MadNLP.format_time(other))")
    MadNLP.@notice(logger, "Total wall secs                                       = $(MadNLP.format_time(cnt.total_time))\n")
    return nothing
end
