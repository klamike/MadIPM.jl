using Base.Threads
using Polyester

NVTX.@annotate function sync_active_streams!(batch_solver::UniformBatchSolver)
    for i in 1:batch_solver.bkkt.active_batch_size[]
        solver_idx = batch_solver.bkkt.batch_map_rev[i]
        CUDA.synchronize(batch_solver.streams[solver_idx], blocking=true)
    end
end

function batch_func(batch_solver::UniformBatchSolver, func)
    for i in 1:batch_solver.bkkt.active_batch_size[]
        solver_idx = batch_solver.bkkt.batch_map_rev[i]
        solver = batch_solver[solver_idx]
        NVTX.@range "$i - $func" begin
            stream!(batch_solver.streams[solver_idx]) do
                func(solver);
            end
        end
    end
end
NVTX.@annotate function for_active(batch_solver, funcs...)
    for func in funcs
        batch_func(batch_solver, func)
    end
end

function batch_func_withindex(batch_solver::AbstractBatchSolver, func)
    for i in 1:batch_solver.bkkt.active_batch_size[]
        solver_idx = batch_solver.bkkt.batch_map_rev[i]
        solver = batch_solver[solver_idx]
        NVTX.@range "$i - $func" begin
            stream!(batch_solver.streams[solver_idx]) do
                func(i, solver);
            end
        end
    end
end
NVTX.@annotate function for_active_withindex(batch_solver, funcs...)
    for func in funcs
        batch_func_withindex(batch_solver, func)
    end
end

NVTX.@annotate function batch_initialize!(batch_solver::AbstractBatchSolver)
    for_active(batch_solver,
        MadIPM.pre_initialize!,
        MadIPM.init_starting_point_solve!,
        MadIPM.post_initialize!,
        MadIPM.update_jacl!
    )
    return
end
NVTX.@annotate function batch_mpc!(batch_solver::AbstractBatchSolver)
    while true
        NVTX.@range "Step $(batch_solver[batch_solver.bkkt.batch_map_rev[1]].cnt.k+1) ($(n_active(batch_solver))/$(length(batch_solver)))" begin
            # Check termination criteria
            for_active(batch_solver,
                MadNLP.print_iter,
                MadIPM.update_termination_criteria!,
            )
            update_batch!(batch_solver)
            all_done(batch_solver) && return

            # Run MPC step
            for_active(batch_solver,
                MadIPM.update_regularization!,
                MadIPM.factorize_regularized_system!,
                MadIPM.set_predictive_rhs!,
                MadIPM.solve_system!,
                MadIPM.prediction_step_size!,
                MadIPM.set_correction_rhs!,
                MadIPM.solve_system!,
                MadIPM.gondzio_correction_direction!,
                MadIPM.update_step_size!,
                MadIPM.apply_step!,
                MadIPM.evaluate_model!
            )
        end
    end
end


NVTX.@annotate function MadIPM.solve!(batch_solver::AbstractBatchSolver)
    batch_stats = [MadNLP.MadNLPExecutionStats(solver) for solver in batch_solver]  # TODO: BatchExecutionStats?

    try
        MadNLP.@notice(first(batch_solver).logger,"This is MadIPM, running with $(MadNLP.introduce(batch_solver.bkkt.linear_solver)), batch size $(length(batch_solver))\n")
        batch_initialize!(batch_solver)
        batch_mpc!(batch_solver)
    catch e
        rethrow(e)  # FIXME
    finally
        for (stats, solver) in zip(batch_stats, batch_solver)
            MadIPM.finalize!(stats, solver)
        end
    end

    return batch_stats
end


NVTX.@annotate function MadIPM.madipm(ms::AbstractVector{NLPModel}; kwargs...) where {NLPModel <: NLPModels.AbstractNLPModel}
    # NVTX.@range "Allocating MPCSolvers" begin
    #     solvers = MadIPM.MPCSolver.(ms; linear_solver = NoLinearSolver, kwargs...) # TODO: special constructor to share kkt/cb memory/set NoLinearSolver
    # end
    NVTX.@range "Init UniformBatchSolver" begin
        batch_solver = UniformBatchSolver(ms; linear_solver = MadNLPGPU.CUDSSSolver, kwargs...)
    end
    return MadIPM.solve!(batch_solver)
end