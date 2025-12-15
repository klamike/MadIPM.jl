
#=
    Interface to direct solver for solving KKT system
=#

function factorize_regularized_system!(solver)
    max_trials = 3
    for ntrial in 1:max_trials
        set_aug_diagonal_reg!(solver.kkt, solver)
        MadNLP.factorize_wrapper!(solver)
        if is_factorized(solver.kkt.linear_solver)
            break
        end
        solver.del_w[] *= 100.0
        solver.del_c[] *= 100.0
    end
end

function _presolve_system!(
    solver::MadNLP.AbstractMadNLPSolver{T},
) where T
    copyto!(MadNLP.full(solver.d), MadNLP.full(solver.p))
end

function _postsolve_system!(
    solver::MadNLP.AbstractMadNLPSolver{T},
) where T
    d = solver.d
    p = solver.p
    # Check residual
    w = solver._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, solver.kkt, d, -one(T), one(T))
    norm_w = norm(MadNLP.full(w), Inf)
    norm_p = norm(MadNLP.full(p), Inf)

    residual_ratio = norm_w / max(one(T), norm_p)
    MadNLP.@debug(
        solver.logger,
        @sprintf("Residual after linear solve: %6.2e", residual_ratio),
    )
    if isnan(residual_ratio) || (solver.opt.check_residual && (residual_ratio > solver.opt.tol_linear_solve))
        @error "SolveException" residual_ratio
        throw(MadNLP.SolveException)
    end
end

function solve_system!(
    solver::MadNLP.AbstractMadNLPSolver{T},
) where T
    @error "Unbatched solve"
    _presolve_system!(solver)
    MadNLP.solve!(solver.kkt, solver.d)
    _postsolve_system!(solver)
    return solver.d
end
