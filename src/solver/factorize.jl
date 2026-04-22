# Primal-dual regularization + KKT factorization loop. Try up to 3 times:
# shrink regularization via the schedule, set the augmented diagonal, factor,
# and if factorization fails bump the failed instances' regularization by
# 100× and retry. The `_check_factorization_status!` /
# `_bump_failed_regularization!` hooks below specialize for scalar vs batch.

function factorize_regularized_system!(s::MaybeBatchMPCSolver)
    update_regularization!(s, _regularization(s))
    max_trials = 3
    for _ in 1:max_trials
        set_aug_diagonal_reg!(_kkt(s), s)
        MadNLP.factorize_wrapper!(s)
        nfailed = _check_factorization_status!(s)
        nfailed == 0 && break
        _bump_failed_regularization!(s, nfailed)
    end
    return
end

# ---------- scalar: 0/1 failure count, bump uniformly ----------
@inline function _check_factorization_status!(solver::MPCSolver)
    return is_factorized(solver.problem.kkt.linear_solver) ? 0 : 1
end

@inline function _bump_failed_regularization!(solver::MPCSolver, _nfailed)
    state = solver.state
    state.del_w *= 100.0
    state.del_c *= 100.0
    return
end



# ---------- batch: count failures + mask ----------
# `is_factorized!` reports which local instances failed; the mask built
# from `failed_locals` tells `_bump_failed_regularization!` which columns
# get their regularization scaled so the next factorization attempt isn't
# affected by peers that factored fine.

@inline function _check_factorization_status!(batch_solver::UniformBatchMPCSolver)
    problem = batch_solver.problem
    failed_locals = problem.batch_views.selected_local_buffer
    return is_factorized!(failed_locals, problem.kkt.batch_solver, active_view(problem.batch_views))
end

function _bump_failed_regularization!(batch_solver::UniformBatchMPCSolver{T}, nfailed::Int) where T
    problem = batch_solver.problem
    state = batch_solver.state
    factor_view = active_view(problem.batch_views)
    failed_locals = problem.batch_views.selected_local_buffer
    ws = state.workspace
    # build root-level mask from local failed idx
    fill!(ws.active_mask_cpu, zero(T))
    @inbounds for k in 1:nfailed
        j = factor_view.local_to_root[failed_locals[k]]
        ws.active_mask_cpu[1, j] = one(T)
    end
    copyto!(ws.active_mask, ws.active_mask_cpu)
    mask = ws.active_mask
    @. state.del_w = ifelse(mask == one(T), T(100) * state.del_w, state.del_w)
    @. state.del_c = ifelse(mask == one(T), T(100) * state.del_c, state.del_c)
    # restore active mask: required so subsequent steps don't throw away any
    # successful factorization that we still need.
    _update_active_mask!(batch_solver)
    return
end
