include("special.jl")

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
function batch_update_regularization!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        update_regularization!(solver, solver.opt.regularization)
    end
    return
end
function batch_factorize_regularized_system!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        factorize_regularized_system!(solver)
    end
    return
end
function batch_factorize_system!(batch_solver)
    batch_update_regularization!(batch_solver)
    batch_factorize_regularized_system!(batch_solver)
end

function batch_set_predictive_rhs!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        set_predictive_rhs!(solver, solver.kkt)
    end
end
function batch_solve_system!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        solve_system!(solver)
    end
    return
end
function batch_affine_direction!(batch_solver)
    batch_set_predictive_rhs!(batch_solver)
    batch_solve_system!(batch_solver)
    return
end
function batch_prediction_step_size!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        prediction_step_size!(solver)
    end
    return
end
function batch_set_correction_rhs!(batch_solver)
    for solver in batch_solver
        is_done(solver) && continue
        set_correction_rhs!(solver, solver.kkt, solver.mu[], solver.correction_lb, solver.correction_ub, solver.ind_lb, solver.ind_ub)
    end
    return
end
function batch_mehrotra_correction_direction!(batch_solver)
    batch_set_correction_rhs!(batch_solver)
    batch_solve_system!(batch_solver)
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

        # Affine direction
        batch_affine_direction!(batch_solver)

        # Prediction step size
        batch_prediction_step_size!(batch_solver)

        # Mehrotra's Correction direction
        batch_mehrotra_correction_direction!(batch_solver)

        # Gondzio's additional correction direction
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
        # FIXME: batch_introduce
        # for solver in batch_solver
        #     MadNLP.@notice(solver.logger,"This is MadIPM, running with $(MadNLP.introduce(solver.kkt.linear_solver))\n")
        # end
        batch_initialize!(batch_solver)
        batch_mpc!(batch_solver)
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

            MadNLP.update!(batch_stats[i],solver)
        end
    end

    return batch_stats
end


function madipm_batch(ms; kwargs...)
    if isdefined(Main, :CUDSS)  # FIXME
        solver = MadIPM.SameStructureBatchMPCSolver(ms; kwargs...)
        return MadIPM.batch_solve!(solver)
    else
        return madipm_foreach(ms; kwargs...)
    end
end

function madipm_foreach(ms; kwargs...)
    results = []
    for m in ms
        r = madipm(m; kwargs...)
        push!(results, r)
    end

    return results
end