
#=
    Interface to direct solver for solving KKT system. The IPM-side
    try-bump-retry driver (`factorize_regularized_system!`) lives in
    src/solver/factorize.jl alongside its batch counterpart.
=#

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

function solve_system!(
    d::BatchUnreducedKKTVector{T},
    batch_solver::AbstractBatchMPCSolver{T},
    p::BatchUnreducedKKTVector{T},
) where T
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(batch_solver.kkt, batch_solver)

    w = batch_solver._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, batch_solver.kkt, d, -one(T), one(T))

    ws = batch_solver.workspace
    MadNLP.full(w) .*= ws.active_mask
    MadNLP.full(p) .*= ws.active_mask

    opt = batch_solver.opt
    check_res = opt.check_residual
    tol_ls = T(opt.tol_linear_solve)
    _fw = MadNLP.full(w)
    _fw .= abs.(_fw)
    batch_maximum!(ws._norm_gpu_w, _fw)                  # (1,bs) per-instance norm_w
    _fw .= abs.(MadNLP.full(p))
    batch_maximum!(ws._norm_gpu_p, _fw)                  # (1,bs) per-instance norm_p
    @. ws._norm_gpu_w /= max(one(T), ws._norm_gpu_p)    # ratio in-place
    @. ws._ls_error |= isnan(ws._norm_gpu_w) | (check_res & (ws._norm_gpu_w > tol_ls))
    return d
end
