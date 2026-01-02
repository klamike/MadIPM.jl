
#=
    Interface to direct solver for solving KKT system
=#

MadNLP.build_kkt!(solver::MadIPM.MPCSolver) = MadNLP.build_kkt!(solver.kkt)
set_aug_diagonal_reg!(solver) = set_aug_diagonal_reg!(solver.kkt, solver)
NVTX.@annotate function factorize_regularized_system!(solver)
    max_trials = 3
    for ntrial in 1:max_trials
        set_aug_diagonal_reg!(solver)
        MadNLP.factorize_wrapper!(solver)
        if is_factorized(solver.kkt.linear_solver)
            break
        end
        solver.del_w *= 100.0
        solver.del_c *= 100.0
    end
end

NVTX.@annotate function solve_system!(
    d::MadNLP.UnreducedKKTVector{T},
    solver::MadNLP.AbstractMadNLPSolver{T},
    p::MadNLP.UnreducedKKTVector{T},
) where T
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve!(solver.kkt, d)
    # check_residual!(d, solver, p)
    return d
end

NVTX.@annotate function check_residual!(d::MadNLP.UnreducedKKTVector{T}, solver, p) where T
    opt = solver.opt

    # Check residual
    w = solver._w1
    NVTX.@range "copyto" begin
        copyto!(MadNLP.full(w), MadNLP.full(p))
        CUDA.synchronize()
    end
    NVTX.@range "mul" begin
        mul!(w, solver.kkt, d, -one(T), one(T))
        CUDA.synchronize()
    end
    NVTX.@range "norms" begin
        norm_w = norm(MadNLP.full(w), Inf)
        norm_p = norm(MadNLP.full(p), Inf)
        CUDA.synchronize()
    end
    NVTX.@range "ratio" begin
        residual_ratio = norm_w / max(one(T), norm_p)
        CUDA.synchronize()
    end
    NVTX.@range "log" begin
        MadNLP.@debug(
            solver.logger,
            @sprintf("Residual after linear solve: %6.2e", residual_ratio),
            )
        CUDA.synchronize()
    end
    NVTX.@range "check" begin
        if isnan(residual_ratio) || (opt.check_residual && (residual_ratio > opt.tol_linear_solve))
            throw(MadNLP.SolveException)
        end
        CUDA.synchronize()
    end
    return d
end
