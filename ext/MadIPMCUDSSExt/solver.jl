import MadIPM: is_done, is_factorized

function MadIPM.batch_factorize_regularized_system!(
    batch_solver::MadIPM.SparseSameStructureBatchMPCSolver
)
    # NOTE: no trials since is_factorized(::CUDSS) = true
    for solver in batch_solver
        is_done(solver) && continue
        MadIPM.set_aug_diagonal_reg!(solver.kkt, solver)
        MadNLP.build_kkt!(solver.kkt)
    end
    MadNLP.factorize!(batch_solver.kkts.batch_solver)
    return
end

function MadIPM.batch_solve_system!(
    batch_solver::MadIPM.SparseSameStructureBatchMPCSolver
)
    for solver in batch_solver
        is_done(solver) && continue
        MadIPM._presolve_system!(solver)  # from MadIPM.solve_system!
        MadNLP.reduce_rhs!(solver.kkt, solver.d)  # from solve! for SparseKKTSystem
    end

    d = batch_solver.d
    copyto!(d.primal_dual_buffer, MadNLP.primal_dual(d))
    MadNLP.solve!(batch_solver.kkts.batch_solver, d.primal_dual_buffer)
    copyto!(MadNLP.primal_dual(d), d.primal_dual_buffer)

    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.finish_aug_solve!(solver.kkt, solver.d)  # from solve! for SparseKKTSystem
        MadIPM._postsolve_system!(solver)  # from MadIPM.solve_system!
    end
    return
end

function MadIPM.batch_init_starting_point_solve!(
        batch_solver::MadIPM.SparseSameStructureBatchMPCSolver
)
    for solver in batch_solver
        solver.kkt.reg .= solver.del_w
        solver.kkt.pr_diag .= solver.del_w
        solver.kkt.du_diag .= solver.del_c
        MadNLP.build_kkt!(solver.kkt)
    end
    MadNLP.factorize!(batch_solver.kkts.batch_solver)
    for solver in batch_solver
        MadIPM.set_initial_primal_rhs!(solver)
    end
    MadIPM.batch_solve_system!(batch_solver)
    for solver in batch_solver
        axpy!(1.0, MadNLP.primal(solver.d), MadNLP.primal(solver.x))
        MadIPM.set_initial_dual_rhs!(solver)
    end
    MadIPM.batch_solve_system!(batch_solver)
    return
end