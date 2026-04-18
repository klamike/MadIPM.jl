# Try-bump-retry factorization driver.
# Stays specialized: scalar bumps a single (del_w, del_c) on failure;
# batch only bumps the per-instance entries whose factorization failed.

# ---------- scalar ----------

function factorize_regularized_system!(solver::MPCSolver)
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

# ---------- batch ----------

function _bump_failed_regularization!(batch_solver::AbstractBatchMPCSolver{T}, failed_locals, nfailed::Int) where T
    factor_view = active_view(batch_solver.batch_views)
    ws = batch_solver.workspace
    # build root-level mask from local failed idx
    fill!(ws.active_mask_cpu, zero(T))
    @inbounds for k in 1:nfailed
        j = factor_view.local_to_root[failed_locals[k]]
        ws.active_mask_cpu[1, j] = one(T)
    end
    copyto!(ws.active_mask, ws.active_mask_cpu)
    mask = ws.active_mask
    @. batch_solver.del_w = ifelse(mask == one(T), T(100) * batch_solver.del_w, batch_solver.del_w)
    @. batch_solver.del_c = ifelse(mask == one(T), T(100) * batch_solver.del_c, batch_solver.del_c)
    # restore active mask
    # this is required to not throw away any successful factorization that we need later
    _update_active_mask!(batch_solver)
    return
end

function factorize_system!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    batch_views = batch_solver.batch_views
    update_regularization!(batch_solver, batch_solver.regularization)
    max_trials = 3
    factor_view = active_view(batch_views)
    failed_locals = batch_views.selected_local_buffer

    for _ in 1:max_trials
        set_aug_diagonal_reg!(batch_solver.kkt, batch_solver)
        MadNLP.factorize_wrapper!(batch_solver)
        nfailed = is_factorized!(
            failed_locals, batch_solver.kkt.batch_solver, factor_view,
        )
        nfailed == 0 && break
        _bump_failed_regularization!(batch_solver, failed_locals, nfailed)
    end
    return
end
