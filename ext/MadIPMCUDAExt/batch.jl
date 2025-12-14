import MadIPM: is_done, is_factorized



function MadIPM.batch_affine_direction!(
    batch_solver::MadIPM.SameStructureBatchMPCSolver{T, Ts, MT, VT, VI, KKTSystem, BK}
) where {
    T, Ts, MT, VT, VI,
    KKTSystem <: MadNLP.SparseKKTSystem,
    BK <: MadIPM.SparseSameStructureBatchKKTSystem{T, <:AbstractVector{T}, <:AbstractVector{Int32}, KKTSystem, <:CUDSSUniformBatchSolver{T}},
}
    for solver in batch_solver
        is_done(solver) && continue
        MadIPM.set_predictive_rhs!(solver, solver.kkt)
        MadNLP.reduce_rhs!(solver.kkt, solver.d)
    end

    d = batch_solver.d
    copyto!(d.primal_dual_buffer, vec(d.primal_dual_view))
    MadNLP.solve!(batch_solver.kkts.batch_solver, d.primal_dual_buffer)
    copyto!(vec(d.primal_dual_view), d.primal_dual_buffer)

    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.finish_aug_solve!(solver.kkt, solver.d)
    end
    return
end