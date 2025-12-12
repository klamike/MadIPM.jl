function batch_factorize_system!(batch_solver::BatchMPCSolver)
    for solver in batch_solver
        is_done(solver) && continue
        update_regularization!(solver, solver.opt.regularization)
    end

    max_trials = 3
    for ntrial in 1:max_trials
        for solver in batch_solver
            is_done(solver) && continue
            if !isone(ntrial)
                is_factorized(solver.kkt.linear_solver) && continue
                solver.del_w *= 100.0
                solver.del_c *= 100.0
            end
            set_aug_diagonal_reg!(solver.kkt, solver)  # reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower
        end

        for solver in batch_solver
            is_done(solver) && continue
            if !isone(ntrial)
                is_factorized(solver.kkt.linear_solver) && continue
            end
            MadNLP.factorize_wrapper!(solver)
        end
    end
    return
end

function batch_affine_direction!(batch_solver::BatchMPCSolver)
    for solver in batch_solver
        is_done(solver) && continue
        set_predictive_rhs!(solver, solver.kkt)
    end

    for solver in batch_solver
        is_done(solver) && continue
        solve_system!(solver.d, solver, solver.p)
    end
    return
end

function batch_mehrotra_correction_direction!(batch_solver::BatchMPCSolver)
    for solver in batch_solver
        is_done(solver) && continue
        set_correction_rhs!(solver, solver.kkt, solver.mu, solver.correction_lb, solver.correction_ub, solver.ind_lb, solver.ind_ub)
    end

    for solver in batch_solver
        is_done(solver) && continue
        solve_system!(solver.d, solver, solver.p)
    end
    return
end

function batch_evaluate_model!(batch_solver::BatchMPCSolver)
    for solver in batch_solver
        is_done(solver) && continue
        solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
    end

    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
    end

    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)
    end

    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)
    end
    return
end