function batch_evaluate_model!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        evaluate_model!(solver)
    end
    return
end
function batch_update_termination_criteria!(batch_solver)
    all_done = true
    for solver in batch_solver
        is_done(solver) && continue
        update_termination_criteria!(solver)
        all_done = all_done && is_done(solver)
    end
    return all_done
end
function batch_factorize_system!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        factorize_system!(solver)
    end
    return
end
function batch_prediction_step!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        prediction_step!(solver)
    end
    return
end
function batch_mehrotra_correction_direction!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        mehrotra_correction_direction!(solver)
    end
    return
end
function batch_gondzio_correction_direction!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        gondzio_correction_direction!(solver)
    end
    return
end
function batch_update_step_size!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        update_step_size!(solver)
    end
    return
end
function batch_apply_step!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        apply_step!(solver)
    end
    return
end
function batch_mpc!(batch_solver)
    
    while true
        # Check termination criteria
        batch_print_iter(batch_solver)
        all_done = batch_update_termination_criteria!(batch_solver)
        all_done && return

        # Factorize KKT system
        batch_factorize_system!(batch_solver)

        # Prediction step
        batch_prediction_step!(batch_solver)

        # Mehrotra's Correction step
        batch_mehrotra_correction_direction!(batch_solver)

        # Gondzio's additional correction
        batch_gondzio_correction_direction!(batch_solver)

        # Update step size
        batch_update_step_size!(batch_solver)

        # Apply step
        batch_apply_step!(batch_solver)

        # Evaluate model at new iterate
        batch_evaluate_model!(batch_solver)
    end
end

function batch_print_iter(batch_solver)
    for solver in batch_solver
        MadNLP.print_iter(solver)
    end
    return
end

function batch_initialize!(batch_solver)
    for solver in batch_solver
        initialize!(solver)
    end
    return
end
function batch_solve!(
    batch_solver;
    kwargs...
)
    batch_stats = [MadNLP.MadNLPExecutionStats(solver) for solver in batch_solver]

    start_time = time()
    for solver in batch_solver
        solver.cnt.start_time = start_time
    end

    if !isempty(kwargs)
        for solver in batch_solver
            MadNLP.set_options!(solver.opt, kwargs)
        end
    end

    try
        for solver in batch_solver
            MadNLP.@notice(solver.logger,"This is MadIPM, running with $(MadNLP.introduce(solver.kkt.linear_solver))\n")
        end
        # batch_initialize!(batch_solver)
        for solver in batch_solver
            initialize!(solver)
            mpc!(solver)
        end
    catch e
        rethrow(e)  # FIXME
    finally
        total_time = time() - start_time  # FIXME: track when each one finishes?
        for (i, solver) in enumerate(batch_solver)
            solver.cnt.total_time = total_time
            if !(solver.status < MadNLP.SOLVE_SUCCEEDED)
                MadNLP.print_summary(solver)
            end
            MadNLP.@notice(solver.logger,"EXIT: $(MadNLP.get_status_output(solver.status, solver.opt))")
            finalize(solver.logger)

            update_solution!(batch_stats[i], solver)
        end
    end

    return batch_stats
end


function madipm_batch(ms; kwargs...)
    solver = MadIPM.BatchMPCSolver(ms; kwargs...)
    return MadIPM.batch_solve!(solver)
end

function madipm_foreach(ms; kwargs...)
    results = []
    for m in ms
        r = madipm(m; kwargs...)
        push!(results, r)
    end

    return results
end