import MadIPM: is_done, is_factorized

function batch_factorize_system!(
    batch_solver::MadIPM.BatchMPCSolver{T, MT, VT, VI, KKTSystem}
) where {
    T, MT, VT, VI,
    QN, VI32,
    LS <: MadNLPGPU.CUDSSSolver,
    KKTSystem <: Union{
        MadNLP.SparseKKTSystem{T, VT, MT, QN, LS},
        MadIPM.NormalKKTSystem{T, VT, MT, VI, VI32, LS}
    },
}
    for solver in batch_solver
        is_done(solver) && continue
        MadIPM.update_regularization!(solver, solver.opt.regularization)
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
            MadIPM.set_aug_diagonal_reg!(solver.kkt, solver)
        end

        # TODO: build batch for cudss

        for solver in batch_solver
            is_done(solver) && continue
            if !isone(ntrial)
                is_factorized(solver.kkt.linear_solver) && continue
            end

            MadNLP.@trace(solver.logger,"Factorization started.")
            MadNLP.build_kkt!(solver.kkt)
        end
        for solver in batch_solver
            is_done(solver) && continue
            if !isone(ntrial)
                is_factorized(solver.kkt.linear_solver) && continue
            end

            solver.cnt.linear_solver_time += @elapsed begin
                MadNLP.factorize!(solver.kkt.linear_solver)
            end
        end
    end
    return
end