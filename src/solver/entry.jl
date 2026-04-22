# ============================================================================
# Public entry points — scalar (`madipm`) and batch (`madipm_batch`).
# ============================================================================

"""
    solve!(solver::MPCSolver[, stats])

Run the Mehrotra predictor-corrector IPM loop. Returns a
`MadNLP.MadNLPExecutionStats` (allocated if not supplied). MadNLP's callback
and linear-solver exceptions are caught and translated into termination
statuses; `solver.problem.opt.rethrow_error = true` re-raises.
"""
solve!(solver::MPCSolver) = solve!(solver, MadNLP.MadNLPExecutionStats(solver))

function solve!(solver::MPCSolver, stats::MadNLP.MadNLPExecutionStats)
    problem, state = solver.problem, solver.state
    try
        MadNLP.@notice(problem.logger,
            "This is MadIPM, running with $(MadNLP.introduce(problem.kkt.linear_solver))\n")
        initialize!(solver)
        mpc!(solver)
    catch e
        state.status = _translate_exception(e, problem.opt.rethrow_error)
    finally
        state.cnt.total_time = time() - state.cnt.start_time
        state.status < MadNLP.SOLVE_SUCCEEDED || MadNLP.print_summary(solver)
        MadNLP.@notice(problem.logger,
            "EXIT: $(MadNLP.get_status_output(state.status, problem.opt))")
        finalize(problem.logger)
        update_solution!(stats, solver)
    end
    return stats
end

# ---------- exception translation ----------

function _translate_exception(e, rethrow_error::Bool)
    if e isa MadNLP.InvalidNumberException
        return _invalid_number_status(e.callback)
    elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
        return MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
    elseif e isa MadNLP.LinearSolverException
        rethrow_error && rethrow(e)
        return MadNLP.ERROR_IN_STEP_COMPUTATION
    elseif e isa InterruptException
        rethrow_error && rethrow(e)
        return MadNLP.USER_REQUESTED_STOP
    else
        rethrow_error && rethrow(e)
        return MadNLP.INTERNAL_ERROR
    end
end

@inline _invalid_number_status(cb::Symbol) =
    cb === :obj  ? MadNLP.INVALID_NUMBER_OBJECTIVE            :
    cb === :grad ? MadNLP.INVALID_NUMBER_GRADIENT             :
    cb === :cons ? MadNLP.INVALID_NUMBER_CONSTRAINTS          :
    cb === :jac  ? MadNLP.INVALID_NUMBER_JACOBIAN             :
    cb === :hess ? MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN   :
                   MadNLP.INVALID_NUMBER_DETECTED

"""
    madipm(nlp; kwargs...)

Build an [`MPCSolver`](@ref) for `nlp` (a `LinearModel` or `QuadraticModel`)
and run it. Kwargs forward to [`IPMOptions`](@ref) / [`load_options`](@ref).
"""
madipm(nlp; kwargs...) = solve!(MPCSolver(nlp; kwargs...))

# ============================================================================
# Batch
# ============================================================================

"""
    solve!(batch_solver::UniformBatchMPCSolver)

Run the batched IPM loop. Returns a [`BatchExecutionStats`](@ref) carrying
per-instance status / solution / multipliers. Per-instance failures mark
`stats.status[i]` and let the rest of the batch finish;
`rethrow_error = true` re-raises.
"""
function solve!(batch_solver::UniformBatchMPCSolver{T, MT, VT}) where {T, MT, VT}
    problem = batch_solver.problem
    state   = batch_solver.state
    ws      = state.workspace
    bcb     = problem.bcb
    bs      = problem.batch_size

    nvar_nlp, ncon = problem.original_nlp === nothing ?
        (bcb.nlp.meta.nvar, bcb.ncon) :
        (NLPModels.get_nvar(problem.original_nlp),
         NLPModels.get_ncon(problem.original_nlp))
    stats = BatchExecutionStats(MT, VT, nvar_nlp, ncon, bs)

    try
        MadNLP.@notice(problem.logger, "MadIPM batch solve ($bs problems)\n")
        initialize!(batch_solver)
        mpc!(batch_solver)
    catch e
        for i in 1:bs
            ws.status[i] == MadNLP.REGULAR && (ws.status[i] = MadNLP.INTERNAL_ERROR)
        end
        problem.opt.rethrow_error && rethrow(e)
    finally
        state.cnt.total_time .= time() .- state.cnt.start_time
        update_solution!(stats, batch_solver)
        _log_batch_summary(problem.logger, problem.opt, ws.status, bs)
    end
    return stats
end

function _log_batch_summary(logger, opt, statuses, bs)
    counts = Dict{MadNLP.Status, Int}()
    for i in 1:bs
        counts[statuses[i]] = get(counts, statuses[i], 0) + 1
    end
    for (s, n) in counts
        MadNLP.@notice(logger, "$(MadNLP.get_status_output(s, opt)): $n/$bs")
    end
    return nothing
end

"""
    madipm_batch(bnlp::AbstractBatchNLPModel; kwargs...)

Build a [`UniformBatchMPCSolver`](@ref) and run it. The batch must share
Jacobian/Hessian sparsity and bound kinds (enforced by `standard_form`).
"""
madipm_batch(bnlp::NLPModels.AbstractBatchNLPModel; kwargs...) =
    solve!(UniformBatchMPCSolver(bnlp; kwargs...))

function IPMOptions(
    ::NLPModels.AbstractBatchNLPModel{T};
    linear_solver = MadNLP.LDLSolver,
    kwargs...,
) where {T}
    return IPMOptions(; linear_solver = linear_solver, kwargs...)
end

# ---------- batch iteration printer ----------

function MadNLP.print_iter(batch_solver::UniformBatchMPCSolver)
    problem, state = batch_solver.problem, batch_solver.state
    logger         = problem.logger
    MadNLP.get_level(logger) > MadNLP.INFO && return nothing

    ws   = state.workspace
    bcnt = state.cnt
    k    = maximum(bcnt.k)
    na   = active_batch_size(batch_solver)
    bs   = problem.batch_size

    mod(k, 10) == 0 && MadNLP.@info(logger, @sprintf(
        " iter  active  max_inf_pr  max_inf_du  max_inf_compl  max_alpha_p"))
    MadNLP.@info(logger, @sprintf("%4i  ", k) * lpad("$na/$bs", 6) *
        @sprintf("   %6.2e     %6.2e      %7.2e      %6.2e",
                 maximum(ws.inf_pr), maximum(ws.inf_du),
                 maximum(ws.inf_compl), maximum(ws.alpha_p)))
    return nothing
end
