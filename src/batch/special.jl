function batch_affine_direction!(batch_solver::BatchMPCSolver)
    for solver in batch_solver
        is_done(solver) && continue
        set_predictive_rhs!(solver, solver.kkt)
    end

    for solver in batch_solver
        is_done(solver) && continue
        _presolve_system!(solver.d, solver, solver.p)
    end

    # TODO: MadNLP.solve!(::BatchKKTSystem{LS<:CUDSSSolver}, ::CuMatrix)
    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.solve!(solver.kkt, solver.d)
    end

    for solver in batch_solver
        is_done(solver) && continue
        _postsolve_system!(solver.d, solver, solver.p)
    end
    return
end

function batch_mehrotra_correction_direction!(batch_solver::BatchMPCSolver)
    for solver in batch_solver
        is_done(solver) && continue
        set_correction_rhs!(solver, solver.kkt, solver.mu[], solver.correction_lb, solver.correction_ub, solver.ind_lb, solver.ind_ub)
    end

    for solver in batch_solver
        is_done(solver) && continue
        _presolve_system!(solver.d, solver, solver.p)
    end

    # TODO: MadNLP.solve!(::BatchKKTSystem{LS<:CUDSSSolver}, ::CuMatrix)
    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.solve!(solver.kkt, solver.d)
    end

    for solver in batch_solver
        is_done(solver) && continue
        _postsolve_system!(solver.d, solver, solver.p)
    end
    return
end

function batch_evaluate_model!(batch_solver::BatchMPCSolver)
    # TODO: use NLPModels.batch_*
    for solver in batch_solver
        is_done(solver) && continue
        solver.obj_val[] = MadNLP.eval_f_wrapper(solver, solver.x)
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