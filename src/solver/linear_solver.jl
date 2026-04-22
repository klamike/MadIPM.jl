# ============================================================================
# KKT-system solves and factorizations.
#
# `solve_system!`:
#   - copies the rhs, solves, then recomputes the residual `w = p - K·d`;
#   - scalar path throws `MadNLP.SolveException` on divergence or NaN (only
#     when `opt.check_residual` is set);
#   - batch path flags failed instances in `ws._ls_error[j]` so converged
#     peers keep iterating.
#
# `factorize_wrapper!` rebuilds the KKT matrix, factors, accumulates time.
# ============================================================================

# ---------- scalar ----------

function solve_system!(
    d::MadNLP.UnreducedKKTVector{T},
    solver::MadNLP.AbstractMadNLPSolver{T},
    p::MadNLP.UnreducedKKTVector{T},
) where {T}
    problem, state = solver.problem, solver.state
    opt            = problem.opt

    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(problem.kkt, d)

    w = state._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, problem.kkt, d, -one(T), one(T))

    residual_ratio = norm(MadNLP.full(w), Inf) / max(one(T), norm(MadNLP.full(p), Inf))
    MadNLP.@debug(problem.logger,
        @sprintf("Residual after linear solve: %6.2e", residual_ratio))

    (isnan(residual_ratio) ||
     (opt.check_residual && residual_ratio > opt.tol_linear_solve)) &&
        throw(MadNLP.SolveException)
    return d
end

function MadNLP.factorize_wrapper!(s::MaybeBatchMPCSolver)
    MadNLP.@trace(_logger(s), "Factorization started.")
    MadNLP.build_kkt!(_kkt(s))
    cnt = s.state.cnt
    cnt.linear_solver_time += @elapsed MadNLP.factorize_kkt!(_kkt(s))
    cnt.factorization_cnt  += 1
    return nothing
end

# ---------- batch ----------

function solve_system!(
    d::BatchUnreducedKKTVector{T},
    batch_solver::UniformBatchMPCSolver{T},
    p::BatchUnreducedKKTVector{T},
) where {T}
    problem, state = batch_solver.problem, batch_solver.state
    opt            = problem.opt
    ws             = state.workspace

    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(problem.kkt, batch_solver)

    w = state._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, problem.kkt, d, -one(T), one(T))

    MadNLP.full(w) .*= ws.active_mask
    MadNLP.full(p) .*= ws.active_mask

    tol_ls    = T(opt.tol_linear_solve)
    check_res = opt.check_residual
    fw        = MadNLP.full(w)

    fw .= abs.(fw)
    batch_maximum!(ws._norm_gpu_w, fw)
    fw .= abs.(MadNLP.full(p))
    batch_maximum!(ws._norm_gpu_p, fw)
    @. ws._norm_gpu_w /= max(one(T), ws._norm_gpu_p)
    @. ws._ls_error |= isnan(ws._norm_gpu_w) | (check_res & (ws._norm_gpu_w > tol_ls))
    return d
end
