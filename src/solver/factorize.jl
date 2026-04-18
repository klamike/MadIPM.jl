# Try-bump-retry factorization driver, unified across scalar and batch.
#
# Both flavours follow the same loop: set the augmented diagonal, factorize,
# check whether anything failed, and bump the corresponding regularization
# entries up by 100. The flavours diverge only in two hooks:
#
#   * `_check_factorization_status!(s)` — returns `nfailed`, the number of
#     factorizations to retry. Scalar treats the whole solve as a single unit
#     (returns 0 / 1); batch returns the number of per-instance failures and
#     records them in the active-set buffer.
#   * `_bump_failed_regularization!(s, nfailed)` — multiplies del_w / del_c
#     by 100 for the failed entries. Scalar bumps the single scalar pair;
#     batch bumps only the masked instances.

function factorize_regularized_system!(s::AnyMPCSolver)
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

# ---------- scalar hooks ----------

@inline function _check_factorization_status!(solver::MPCSolver)
    return is_factorized(solver.problem.kkt.linear_solver) ? 0 : 1
end

@inline function _bump_failed_regularization!(solver::MPCSolver, _nfailed)
    state = solver.state
    state.del_w *= 100.0
    state.del_c *= 100.0
    return
end

# ---------- batch hooks ----------

@inline function _check_factorization_status!(batch_solver::AbstractBatchMPCSolver)
    problem = batch_solver.problem
    failed_locals = problem.batch_views.selected_local_buffer
    return is_factorized!(failed_locals, problem.kkt.batch_solver, active_view(problem.batch_views))
end

function _bump_failed_regularization!(batch_solver::AbstractBatchMPCSolver{T}, nfailed::Int) where T
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
