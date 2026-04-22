# ============================================================================
# Solution unpacking.
#
# `MadNLPExecutionStats(solver)` allocates return buffers sized for the
# ORIGINAL model. `update_solution!` scatters the std-form iterate back into
# the original variable/constraint space via the BQM presolve workspace.
# ============================================================================

# ---------- scalar ----------

function MadNLP.MadNLPExecutionStats(solver::MPCSolver{T}) where {T}
    orig = solver.problem.original_nlp
    n, m = NLPModels.get_nvar(orig), NLPModels.get_ncon(orig)
    xs, ys = NLPModels.get_x0(orig), NLPModels.get_y0(orig)
    return MadNLP.MadNLPExecutionStats(
        solver.problem.opt, solver.state.status,
        similar(xs, n),                 # solution
        zero(T),                        # objective
        similar(ys, m),                 # constraints
        zero(T), zero(T),               # dual_feas, primal_feas
        similar(ys, m),                 # multipliers
        similar(xs, n), similar(xs, n), # multipliers_L, _U
        0,                              # iter
        solver.state.cnt,
    )
end

function update_solution!(stats::MadNLP.MadNLPExecutionStats, solver::MPCSolver)
    problem, state = solver.problem, solver.state
    ws             = problem.workspace

    stats.status = state.status
    recover_primal!(stats.solution, ws, MadNLP.variable(state.x))
    BatchQuadraticModels._gather_dual!(stats.multipliers, ws.con_start.row, state.y)
    recover_variable_multipliers!(stats.multipliers_L, stats.multipliers_U,
                                  ws, MadNLP.variable(state.zl))
    NLPModels.cons!(problem.original_nlp, stats.solution, stats.constraints)
    stats.objective   = NLPModels.obj(problem.original_nlp, stats.solution)
    stats.dual_feas   = state.inf_du
    stats.primal_feas = state.inf_pr
    stats.iter        = state.cnt.k
    return stats
end

# ---------- batch ----------
#
# Two paths: `original_nlp === nothing` means the batch was built directly
# on a standard-form NLP (`_std!`); otherwise we map std → orig via the BQM
# workspace (`_orig!`).

function update_solution!(stats::BatchExecutionStats,
                          batch_solver::UniformBatchMPCSolver)
    state = batch_solver.state
    ws    = state.workspace

    stats.status      .= ws.status
    stats.iter        .= state.cnt.k
    stats.dual_feas   .= vec(ws.inf_du)
    stats.primal_feas .= vec(ws.inf_pr)
    stats.total_time  .= state.cnt.total_time

    batch_solver.problem.original_nlp === nothing ?
        _update_solution_std!(stats, batch_solver) :
        _update_solution_orig!(stats, batch_solver)
    return stats
end

function _update_solution_std!(stats::BatchExecutionStats,
                                batch_solver::UniformBatchMPCSolver)
    state = batch_solver.state
    bcb   = batch_solver.problem.bcb
    x, zl, zu = state.x, state.zl, state.zu

    MadNLP.unpack_x!(stats.solution,      bcb, x)
    MadNLP.unpack_y!(stats.multipliers,   bcb, MadNLP.full(state.y))
    MadNLP.unpack_z!(stats.multipliers_L, bcb, MadNLP.variable(zl))
    MadNLP.unpack_z!(stats.multipliers_U, bcb, MadNLP.variable(zu))
    unpack_obj!(stats.objective, bcb, state.workspace.obj_val)
    MadNLP.unpack_cons!(stats.constraints, bcb,
        MadNLP.full(state.c), MadNLP.full(state.rhs),
        bcb.ind_ineq, MadNLP.slack(x))
    return nothing
end

function _update_solution_orig!(stats::BatchExecutionStats,
                                 batch_solver::UniformBatchMPCSolver)
    problem = batch_solver.problem
    state   = batch_solver.state
    ws_bqm  = problem.workspace
    bcb     = problem.bcb

    recover_primal!(stats.solution, ws_bqm, MadNLP.variable(state.x))
    recover_variable_multipliers!(stats.multipliers_L, stats.multipliers_U,
                                  ws_bqm, MadNLP.variable(state.zl))
    _gather_orig_dual!(stats.multipliers, ws_bqm, MadNLP.full(state.y))

    # objective: std obj includes c_std'z + ½ z'Q z; add the presolve shift
    unpack_obj!(stats.objective, bcb, state.workspace.obj_val)
    stats.objective .+= ws_bqm.c0_batch

    # constraints in original space: cᵢ = A·xᵢ per instance
    _orig_constraints!(stats.constraints, problem.original_nlp, stats.solution)
    return nothing
end

@inline _gather_orig_dual!(mult, ws::StandardFormBatchWorkspace, std_y) =
    BatchQuadraticModels._gather_dual!(mult, ws.con_start.row, std_y)

@inline _orig_constraints!(c, bnlp::BatchQuadraticModel, x) = mul!(c, bnlp.A, x)
