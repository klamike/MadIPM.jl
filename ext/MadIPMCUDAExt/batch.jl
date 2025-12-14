import MadIPM: is_done, is_factorized



function batch_affine_direction!(
    batch_solver::MadIPM.SameStructureBatchMPCSolver{T, Ts, MT, VT, VI, KKTSystem, BK}
) where {
    T, Ts, MT, VT, VI,
    KKTSystem <: MadNLP.SparseKKTSystem,
    BK <: MadIPM.SparseSameStructureBatchKKTSystem{T, VT, VI, KKTSystem, <:CUDSSUniformBatchSolver{T}},
}
    for solver in batch_solver
        is_done(solver) && continue
        set_predictive_rhs!(solver, solver.kkt)
        MadNLP.reduce_rhs!(solver.kkt, solver.d)
    end

    # TODO: MadNLP.solve!(::BatchKKTSystem, ::BatchUnreducedKKTVector)
    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.solve!(solver.kkt.linear_solver, MadNLP.primal_dual(solver.d))
    end

    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.finish_aug_solve!(solver.kkt, solver.d)
    end
    return
end