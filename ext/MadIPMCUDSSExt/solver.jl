import MadIPM: is_done, is_factorized

function MadIPM.batch_factorize_regularized_system!(
    batch_solver::MadIPM.SparseSameStructureBatchMPCSolver{T, KKTSystem, BK}
) where {
    T,
    KKTSystem <:MadNLP.SparseKKTSystem,
    BK <: MadIPM.SparseSameStructureBatchKKTSystem{T, KKTSystem,
        <:MadNLPGPU.CUDSSSolver{T}},
}
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
    batch_solver::MadIPM.SparseSameStructureBatchMPCSolver{T, KKTSystem, BK}
) where {
    T,
    KKTSystem <: MadNLP.SparseKKTSystem,
    BK <: MadIPM.SparseSameStructureBatchKKTSystem{T, KKTSystem,
        <:MadNLPGPU.CUDSSSolver{T}},
}
    for solver in batch_solver
        is_done(solver) && continue
        MadIPM._presolve_system!(solver)  # from MadIPM.solve_system!
        MadNLP.reduce_rhs!(solver.kkt, solver.d)  # from solve! for SparseKKTSystem
    end
    
    copyto!(d.primal_dual_buffer, MadNLP.primal_dual(batch_solver.d))
    MadNLP.solve!(batch_solver.kkts.batch_solver, d.primal_dual_buffer)
    copyto!(MadNLP.primal_dual(batch_solver.d), d.primal_dual_buffer)

    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.finish_aug_solve!(solver.kkt, solver.d)  # from solve! for SparseKKTSystem
        MadIPM._postsolve_system!(solver)  # from MadIPM.solve_system!
    end
    return
end