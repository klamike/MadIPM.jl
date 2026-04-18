# IPM iteration: predictor-corrector + step + apply + evaluate, and the
# enclosing `mpc!` loop. Stays specialized: scalar uses BLAS axpy! and
# direct eval wrappers; batch uses broadcast (FMA-matched per CLAUDE.md)
# and active-set bookkeeping.

# ---------- scalar ----------

function prediction_step!(solver::MPCSolver)
    state = solver.state
    affine_direction!(solver)
    alpha_aff_p, alpha_aff_d = get_fraction_to_boundary_step(solver, one(eltype(state.y)))
    mu_affine = get_affine_complementarity_measure(solver, alpha_aff_p, alpha_aff_d)
    get_correction!(solver, state.correction_lb)
    state.mu_curr = update_barrier!(_barrier_update(solver), solver, mu_affine)
    return
end

function apply_step!(solver::MPCSolver)
    state = solver.state
    axpy!(state.alpha_p, MadNLP.primal(state.d), MadNLP.primal(state.x))
    axpy!(state.alpha_d, MadNLP.dual(state.d), state.y)
    state.zl_r .+= state.alpha_d .* MadNLP.dual_lb(state.d)
    state.cnt.k += 1
    return
end

function evaluate_model!(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    state.obj_val = MadNLP.eval_f_wrapper(solver, state.x)
    MadNLP.eval_cons_wrapper!(solver, state.c, state.x)
    MadNLP.eval_grad_f_wrapper!(solver, state.f, state.x)
    MadNLP.jtprod!(state.jacl, problem.kkt, state.y)
    return
end

function mpc_step!(solver::MPCSolver)
    update_regularization!(solver, _regularization(solver))
    factorize_regularized_system!(solver)
    prediction_step!(solver)
    mehrotra_correction_direction!(solver)
    update_step!(_step_rule(solver), solver)
    apply_step!(solver)
    evaluate_model!(solver)
    return
end

function mpc!(solver::MPCSolver)
    while true
        MadNLP.print_iter(solver)
        update_termination_criteria!(solver)
        solver.state.status != MadNLP.REGULAR && return
        mpc_step!(solver)
    end
end

# ---------- batch ----------

function prediction_step!(solver::AbstractBatchMPCSolver)
    ws = solver.workspace
    affine_direction!(solver)

    fill!(ws.tau, one(eltype(ws.tau)))
    get_fraction_to_boundary_step!(solver)
    zero_inactive_step!(solver)
    get_affine_complementarity_measure!(solver, ws.alpha_p, ws.alpha_d)
    get_correction!(solver, MadNLP.full(solver.correction_lb))
    update_barrier!(_barrier_update(solver), solver, ws.mu_affine)
    return
end

function apply_step!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, y, xl, xu = batch_solver.x, batch_solver.y, batch_solver.xl, batch_solver.xu
    zl, zu, d = batch_solver.zl, batch_solver.zu, batch_solver.d
    batch_size = batch_solver.batch_size
    nlb, nub = d.nlb, d.nub

    # x += alpha_p * dx
    MadNLP.full(x) .+= ws.alpha_p .* MadNLP.primal(d)

    # y += alpha_d * d_dual
    MadNLP.full(y) .+= ws.alpha_d .* MadNLP.dual(d)

    # zl_r += alpha_d * dzl, zu_r += alpha_d * dzu
    if nlb > 0
        lower(zl) .+= ws.alpha_d .* MadNLP.dual_lb(d)
    end
    if nub > 0
        upper(zu) .+= ws.alpha_d .* MadNLP.dual_ub(d)
    end

    _adjust_boundary_active!(lower(x), lower(xl), upper(x), upper(xu), ws.mu_batch, ws.active_mask)
    increment_k!(batch_solver)  # this is CPU work, ends up overlapped
    return
end

function evaluate_model!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return
end

function mpc_step!(batch_solver::AbstractBatchMPCSolver)
    fill!(batch_solver.workspace._ls_error, zero(Int32))
    factorize_system!(batch_solver)
    prediction_step!(batch_solver)
    mehrotra_correction_direction!(batch_solver)
    update_step!(_step_rule(batch_solver), batch_solver)
    zero_inactive_step!(batch_solver)
    apply_step!(batch_solver)
    evaluate_model!(batch_solver)
end

function _update_active_mask!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.workspace
    buf = ws.active_mask_cpu
    fill_batch_view_mask!(buf, active_view(batch_solver.batch_views))
    copyto!(ws.active_mask, buf)
end

function increment_k!(batch_solver::AbstractBatchMPCSolver)
    bcnt = batch_solver.batch_cnt
    ws = batch_solver.workspace
    for i in 1:batch_solver.batch_size
        if ws.status[i] == MadNLP.REGULAR
            bcnt.k[i] += 1
        end
    end
end

function mpc!(batch_solver::AbstractBatchMPCSolver)
    while true
        MadNLP.print_iter(batch_solver)
        update_termination_criteria!(batch_solver)
        changed = update_termination_status!(batch_solver)
        if changed
            update_active_set!(batch_solver)
            active_batch_size(batch_solver) == 0 && return
            _update_active_mask!(batch_solver)
        end
        mpc_step!(batch_solver)
    end
end
