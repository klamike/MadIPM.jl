import MadIPM: is_done, is_factorized

function MadIPM.batch_factorize_regularized_system!(
    batch_solver::MadIPM.SameStructureBatchMPCSolver{T, Ts, MT, VT, VI, KKTSystem, BK}
) where {
    T, Ts, MT, VT, VI,
    KKTSystem <: MadNLP.SparseKKTSystem,
    BK <: MadIPM.SparseSameStructureBatchKKTSystem{T, <:AbstractVector{T}, <:AbstractVector{Int32}, KKTSystem, <:CUDSSUniformBatchSolver{T}},
}
    for solver in batch_solver
        is_done(solver) && continue
        MadIPM.set_aug_diagonal_reg!(solver.kkt, solver)
        # MadNLP.@trace(solver.logger,"Factorization started.")
        MadNLP.build_kkt!(solver.kkt)
        # solver.cnt.linear_solver_time += @elapsed begin
        #     MadNLP.factorize!(solver.kkt.linear_solver)
        # end
    end
    MadNLP.factorize!(batch_solver.kkts.batch_solver)
    return
end

function MadIPM.batch_solve_system!(
    batch_solver::MadIPM.SameStructureBatchMPCSolver{T, Ts, MT, VT, VI, KKTSystem, BK}
) where {
    T, Ts, MT, VT, VI,
    KKTSystem <: MadNLP.SparseKKTSystem,
    BK <: MadIPM.SparseSameStructureBatchKKTSystem{T, <:AbstractVector{T}, <:AbstractVector{Int32}, KKTSystem, <:CUDSSUniformBatchSolver{T}},
}   
    @error "Batched solve"
    for solver in batch_solver
        is_done(solver) && continue
        MadIPM._presolve_system!(solver)
        MadNLP.reduce_rhs!(solver.kkt, solver.d)
    end

    d = batch_solver.d
    copyto!(d.primal_dual_buffer, vec(d.primal_dual_view))
    MadNLP.solve!(batch_solver.kkts.batch_solver, d.primal_dual_buffer)
    copyto!(vec(d.primal_dual_view), d.primal_dual_buffer)

    for solver in batch_solver
        is_done(solver) && continue
        MadNLP.finish_aug_solve!(solver.kkt, solver.d)
        MadIPM._postsolve_system!(solver)
    end
    return
end