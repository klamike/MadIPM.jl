
#=
    Interface to direct solver for solving KKT system
=#

function factorize_regularized_system!(solver)
    max_trials = 3
    problem = solver.problem
    state = solver.state
    for ntrial in 1:max_trials
        set_aug_diagonal_reg!(problem.kkt, solver)
        MadNLP.factorize_wrapper!(solver)
        if is_factorized(problem.kkt.linear_solver)
            break
        end
        state.del_w *= 100.0
        state.del_c *= 100.0
    end
end

function solve_system!(
    d::MadNLP.UnreducedKKTVector{T},
    solver::MadNLP.AbstractMadNLPSolver{T},
    p::MadNLP.UnreducedKKTVector{T},
) where T
    problem = solver.problem
    state = solver.state
    opt = problem.opt
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(problem.kkt, d)

    # Check residual
    w = state._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, problem.kkt, d, -one(T), one(T))
    norm_w = norm(MadNLP.full(w), Inf)
    norm_p = norm(MadNLP.full(p), Inf)

    residual_ratio = norm_w / max(one(T), norm_p)
    MadNLP.@debug(
        problem.logger,
        @sprintf("Residual after linear solve: %6.2e", residual_ratio),
    )
    if isnan(residual_ratio) || (opt.check_residual && (residual_ratio > opt.tol_linear_solve))
        throw(MadNLP.SolveException)
    end
    return d
end
function MadNLP.factorize_wrapper!(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    MadNLP.@trace(problem.logger, "Factorization started.")
    MadNLP.build_kkt!(problem.kkt)
    state.cnt.linear_solver_time += @elapsed MadNLP.factorize_kkt!(problem.kkt)
    state.cnt.factorization_cnt += 1
    return
end
