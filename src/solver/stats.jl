# Execution-stats construction and solution recovery.

# ---------- scalar ----------

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

# ---------- batch ----------

function update_solution!(stats::BatchExecutionStats, batch_solver::AbstractBatchMPCSolver)
    state = batch_solver.state
    ws = state.workspace
    bcb = batch_solver.problem.bcb
    x, zl, zu = state.x, state.zl, state.zu

    stats.status .= ws.status
    stats.iter .= state.batch_cnt.k

    MadNLP.unpack_x!(stats.solution, bcb, x)
    MadNLP.unpack_y!(stats.multipliers, bcb, MadNLP.full(state.y))
    MadNLP.unpack_z!(stats.multipliers_L, bcb, MadNLP.variable(zl))
    MadNLP.unpack_z!(stats.multipliers_U, bcb, MadNLP.variable(zu))
    unpack_obj!(stats.objective, bcb, ws.obj_val)
    MadNLP.unpack_cons!(stats.constraints, bcb, MadNLP.full(state.c), MadNLP.full(state.rhs), bcb.ind_ineq, MadNLP.slack(x))

    stats.dual_feas .= vec(ws.inf_du)
    stats.primal_feas .= vec(ws.inf_pr)
    stats.total_time .= state.batch_cnt.total_time
    return stats
end
