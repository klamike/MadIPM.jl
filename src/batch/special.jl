function batch_evaluate_model!(batch_solver::SameStructureBatchMPCSolver)
    # TODO: use NLPModels.batch_*
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