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

function apply_step!(s::AnyMPCSolver)
    αp = _alpha_p(s)
    αd = _alpha_d(s)
    d  = _d(s)
    MadNLP.primal(_x(s)) .+= αp .* MadNLP.primal(d)
    _y(s) .+= αd .* MadNLP.dual(d)
    _zl_r(s) .+= αd .* _dz_lb(s)
    _apply_step_post!(s)
    return
end

@inline function _apply_step_post!(solver::MPCSolver)
    solver.state.cnt.k += 1
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
    state = solver.state
    ws = state.workspace
    affine_direction!(solver)

    fill!(ws.tau, one(eltype(ws.tau)))
    get_fraction_to_boundary_step!(solver)
    zero_inactive_step!(solver)
    get_affine_complementarity_measure!(solver, ws.alpha_p, ws.alpha_d)
    get_correction!(solver, MadNLP.full(state.correction_lb))
    update_barrier!(_barrier_update(solver), solver, ws.mu_affine)
    return
end

@inline function _apply_step_post!(batch_solver::AbstractBatchMPCSolver)
    state = batch_solver.state
    ws = state.workspace
    d = state.d
    # ub-side dual update (batch only — std-form scalar has nub=0)
    if d.nub > 0
        upper(state.zu) .+= ws.alpha_d .* MadNLP.dual_ub(d)
    end
    _adjust_boundary_active!(lower(state.x), lower(state.xl), upper(state.x), upper(state.xu), ws.mu_batch, ws.active_mask)
    increment_k!(batch_solver)  # this is CPU work, ends up overlapped
    return
end

function evaluate_model!(batch_solver::AbstractBatchMPCSolver)
    state = batch_solver.state
    problem = batch_solver.problem
    ws = state.workspace
    bcb = problem.bcb
    MadNLP.unpack_x!(ws.bx, bcb, state.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.jtprod!(state.jacl, problem.kkt, state.y)
    return
end

function mpc_step!(batch_solver::AbstractBatchMPCSolver)
    fill!(batch_solver.state.workspace._ls_error, zero(Int32))
    factorize_regularized_system!(batch_solver)
    prediction_step!(batch_solver)
    mehrotra_correction_direction!(batch_solver)
    update_step!(_step_rule(batch_solver), batch_solver)
    zero_inactive_step!(batch_solver)
    apply_step!(batch_solver)
    evaluate_model!(batch_solver)
end

function _update_active_mask!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.state.workspace
    buf = ws.active_mask_cpu
    fill_batch_view_mask!(buf, active_view(batch_solver.problem.batch_views))
    copyto!(ws.active_mask, buf)
end

function increment_k!(batch_solver::AbstractBatchMPCSolver)
    state = batch_solver.state
    bcnt = state.cnt
    ws = state.workspace
    for i in 1:batch_solver.problem.batch_size
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
