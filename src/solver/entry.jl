# Public entry points (`solve!`, `madipm`, `madipm_batch`) and batch
# `print_iter` overload. Wraps the IPM loop with try/finally for status
# reporting.

# ---------- scalar ----------

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

# ---------- batch ----------

function solve!(batch_solver::AbstractBatchMPCSolver{T, MT, VT}) where {T, MT, VT}
    problem = batch_solver.problem
    state = batch_solver.state
    ws = state.workspace
    bcb = problem.bcb
    bs = problem.batch_size

    nvar_nlp = bcb.nlp.meta.nvar
    ncon = bcb.ncon
    stats = BatchExecutionStats(MT, VT, nvar_nlp, ncon, bs)

    try
        MadNLP.@notice(problem.logger, "MadIPM batch solve ($bs problems)\n")
        initialize!(batch_solver)
        mpc!(batch_solver)
    catch e
        for i in 1:bs
            if ws.status[i] == MadNLP.REGULAR
                ws.status[i] = MadNLP.INTERNAL_ERROR
            end
        end
        problem.opt.rethrow_error && rethrow(e)
    finally
        bcnt = state.cnt
        t_end = time()
        bcnt.total_time .= t_end .- bcnt.start_time
        update_solution!(stats, batch_solver)
        status_counts = Dict{MadNLP.Status, Int}()
        for i in 1:bs
            s = ws.status[i]
            status_counts[s] = get(status_counts, s, 0) + 1
        end
        for (s, cnt) in status_counts
            MadNLP.@notice(problem.logger, "$(MadNLP.get_status_output(s, problem.opt)): $cnt/$bs")
        end
    end

    return stats
end

"""
    stats = madipm_batch(bnlp::ObjRHSBatchQuadraticModel; kwargs...)

Solve a batch of LP/QP instances by reformulating each into standard form
(`Ax = b, z ≥ 0`) via [`standard_form`](@ref), running the batch IPM over the
shared std-form KKT, and recovering each primal/dual in the original space.

The input batch must share the Jacobian/Hessian sparsity and bound kinds
across instances (enforced by `standard_form`). Keyword arguments (other
than `regularization`, `step_rule`, `barrier_update`, `print_level`, etc.)
are forwarded to [`IPMOptions`](@ref).
"""
function madipm_batch(bnlp::ObjRHSBatchQuadraticModel; kwargs...)
    std_bnlp, ws_batch = standard_form(bnlp)
    batch_solver = UniformBatchMPCSolver(std_bnlp; kwargs...)
    std_stats = solve!(batch_solver)
    # Recover solution / multipliers in original space.
    nbatch = std_stats.solution |> size |> last
    orig_stats = BatchExecutionStats(typeof(bnlp.c_batch), typeof(bnlp.data.c), NLPModels.get_nvar(bnlp), NLPModels.get_ncon(bnlp), nbatch)
    copyto!(orig_stats.status, std_stats.status)
    recover_primal!(orig_stats.solution, ws_batch, std_stats.solution)
    recover_variable_multipliers!(orig_stats.multipliers_L, orig_stats.multipliers_U, ws_batch, std_stats.multipliers_L)
    BatchQuadraticModels._batch_gather_dual!(orig_stats.multipliers, ws_batch.con_start.row, std_stats.multipliers)
    copyto!(orig_stats.dual_feas,   std_stats.dual_feas)
    copyto!(orig_stats.primal_feas, std_stats.primal_feas)
    copyto!(orig_stats.iter,        std_stats.iter)
    copyto!(orig_stats.total_time,  std_stats.total_time)
    # Objective in orig space: std obj already includes c_std' z + 1/2 z'Q z;
    # add the presolve shift ws_batch.c0_batch.
    copyto!(orig_stats.objective, std_stats.objective)
    orig_stats.objective .+= ws_batch.c0_batch
    # Constraints in original space: cᵢ = A xᵢ (per instance). A is shared.
    mul!(orig_stats.constraints, bnlp.data.A, orig_stats.solution)
    return orig_stats
end

# Fallback for other batch NLP types — no std-form wrapping.
function madipm_batch(bnlp::NLPModels.AbstractBatchNLPModel; kwargs...)
    batch_solver = UniformBatchMPCSolver(bnlp; kwargs...)
    return solve!(batch_solver)
end

function IPMOptions(
    bnlp::NLPModels.AbstractBatchNLPModel{T};
    linear_solver = MadNLP.LDLSolver,
    kwargs...,
) where T
    return IPMOptions(; linear_solver = linear_solver, kwargs...)
end

function MadNLP.print_iter(batch_solver::AbstractBatchMPCSolver)
    problem = batch_solver.problem
    state = batch_solver.state
    logger = problem.logger
    MadNLP.get_level(logger) > MadNLP.INFO && return
    ws = state.workspace
    bcnt = state.cnt
    na = active_batch_size(batch_solver)
    bs = problem.batch_size
    k = maximum(bcnt.k)

    active_str = "$na/$bs"
    mod(k, 10) == 0 && MadNLP.@info(logger, @sprintf(
        " iter  active  max_inf_pr  max_inf_du  max_inf_compl  max_alpha_p"))
    MadNLP.@info(logger, @sprintf(
        "%4i  ", k) * lpad(active_str, 6) * @sprintf(
        "   %6.2e     %6.2e      %7.2e      %6.2e",
        maximum(ws.inf_pr), maximum(ws.inf_du),
        maximum(ws.inf_compl), maximum(ws.alpha_p),
    ))
    return
end
