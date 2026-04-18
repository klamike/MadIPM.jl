function init_starting_point!(batch_solver::AbstractBatchMPCSolver{T}) where {T}
    bkkt = batch_solver.kkt
    ws   = batch_solver.workspace
    nlb  = batch_solver.d.nlb

    bx, bxl, bzl = batch_solver.x, batch_solver.xl, batch_solver.zl
    x = MadNLP.primal(bx)
    xl = lower(bx)
    lb = lower(bxl)
    zl = lower(bzl)
    # use jacl as a scratch buffer for Aᵀ y + c
    res = MadNLP.full(batch_solver.jacl)

    # Add initial primal-dual regularization
    bkkt.reg .= batch_solver.del_w
    pr_diag(bkkt) .= batch_solver.del_w
    du_diag(bkkt) .= batch_solver.del_c

    # Step 0: factorize initial KKT.
    MadNLP.factorize_wrapper!(batch_solver)

    # Step 1: least-squares primal correction. A dx = b - A x.
    set_initial_primal_rhs!(batch_solver)
    solve_system!(batch_solver.d, batch_solver, batch_solver.p)
    x .+= MadNLP.primal(batch_solver.d)

    # Step 2: least-squares dual correction. Aᵀ y = -c.
    set_initial_dual_rhs!(batch_solver)
    solve_system!(batch_solver.d, batch_solver, batch_solver.p)
    MadNLP.full(batch_solver.y) .= MadNLP.dual(batch_solver.d)

    # Step 3: zl = Aᵀ y + c  (since c + Aᵀ y - zl = 0 in std form, no zu).
    MadNLP.jtprod!(res, bkkt, batch_solver.y)
    res .+= MadNLP.primal(batch_solver.f)
    MadNLP.full(bzl) .= res

    _s1 = ws.alpha_xl  # (1,bs) scratch
    # δ_x = max(0, -1.5 * min(xl - lb, 0)); note xl - lb corresponds to x - lvar on std form.
    if nlb > 0
        batch_mapreduce!(-, min, T(Inf), _s1, xl, lb)
        @. _s1 = min(_s1, zero(T))
    else
        fill!(_s1, zero(T))
    end
    delta_x = ws.mu_batch
    @. delta_x = max(zero(T), T(-1.5) * _s1)

    # δ_z = max(0, -1.5 * min(zl, 0))
    if nlb > 0
        batch_mapreduce!(identity, min, T(Inf), _s1, zl)
        @. _s1 = min(_s1, zero(T))
    else
        fill!(_s1, zero(T))
    end
    delta_z = ws.mu_curr
    @. delta_z = max(zero(T), T(-1.5) * _s1)

    xl .+= delta_x
    zl .+= one(T) .+ delta_z

    # μ = sum((xl - lb) * zl)  — LB-only
    μ = ws.mu_affine
    fill!(μ, zero(T))
    if nlb > 0
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_lb, xl, zl)
        μ .+= ws.sum_lb
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_lb, lb, zl)
        μ .-= ws.sum_lb
    end

    # δ_x2 = μ / (2 * sum(zl))
    if nlb > 0
        batch_mapreduce!(identity, +, zero(T), ws.sum_lb, zl)
    else
        fill!(ws.sum_lb, zero(T))
    end
    delta_x2 = _s1
    @. delta_x2 = μ / (2 * ws.sum_lb + eps(T))

    # δ_z2 = μ / (2 * sum(xl - lb))
    if nlb > 0
        batch_mapreduce!(-, +, zero(T), ws.sum_lb, xl, lb)
    else
        fill!(ws.sum_lb, zero(T))
    end
    delta_z2 = ws.alpha_zl
    @. delta_z2 = μ / (2 * ws.sum_lb + eps(T))

    xl .+= delta_x2
    zl .+= delta_z2
    return
end

function initialize!(batch_solver::AbstractBatchMPCSolver{T}) where T
    opt = batch_solver.opt
    bcb = batch_solver.bcb
    ws = batch_solver.workspace

    MadNLP.initialize!(
        bcb,
        batch_solver.x,
        batch_solver.xl,
        MadNLP.full(batch_solver.y),
        MadNLP.full(batch_solver.rhs),
        bcb.ind_ineq,
        ws.bx;
        tol=opt.bound_relax_factor,
        bound_push=opt.bound_push,
        bound_fac=opt.bound_fac,
    )
    fill!(MadNLP.full(batch_solver.jacl), zero(T))

    if opt.scaling
        MadNLP.set_scaling!(
            bcb,
            batch_solver.x,
            batch_solver.xl,
            MadNLP.full(batch_solver.y),
            MadNLP.full(batch_solver.rhs),
            bcb.ind_ineq,
            T(opt.nlp_scaling_max_gradient),
            ws.bx,
        )
    end

    MadNLP.initialize!(batch_solver.kkt)
    init_regularization!(batch_solver, batch_solver.regularization)

    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_jac_wrapper!(batch_solver, batch_solver.kkt)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, batch_solver.kkt)

    batch_mapreduce!(abs, max, typemin(T), ws.norm_b, MadNLP.full(batch_solver.rhs))
    batch_mapreduce!(abs, max, typemin(T), ws.norm_c, MadNLP.full(batch_solver.f))

    init_starting_point!(batch_solver)
    initialize_solver_state!(batch_solver)

    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return
end

function initialize_solver_state!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.workspace
    opt = batch_solver.opt
    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(T))
    fill!(ws.status, MadNLP.REGULAR)
    reset_active_view!(batch_solver.batch_views)
    _update_active_mask!(batch_solver)
    fill!(ws.inf_pr, zero(T))
    fill!(ws.inf_du, zero(T))
    fill!(ws.inf_compl, zero(T))
    fill!(ws.dual_obj, zero(T))
    fill!(ws.alpha_p, zero(T))
    fill!(ws.alpha_d, zero(T))
    t_now = time()
    batch_solver.batch_cnt.start_time[] = t_now
    fill!(batch_solver.batch_cnt.k, 0)
    batch_solver.batch_cnt.linear_solver_time[] = 0.0
    batch_solver.batch_cnt.eval_function_time[] = 0.0
    batch_solver.batch_cnt.obj_cnt[] = 0
    batch_solver.batch_cnt.obj_grad_cnt[] = 0
    batch_solver.batch_cnt.con_cnt[] = 0
    return
end

function compute_term_gpu!(ws::UniformBatchWorkspace{T}, opt) where T
    ds = T(opt.divergence_scale)
    tol = T(opt.tol)
    div_tol = T(opt.divergence_tol)
    Int_ERROR = Int(MadNLP.INTERNAL_ERROR)
    Int_SOLVED = Int(MadNLP.SOLVE_SUCCEEDED)
    Int_INFEASIBLE = Int(MadNLP.INFEASIBLE_PROBLEM_DETECTED)
    Int_DIVERGING = Int(MadNLP.DIVERGING_ITERATES)
    Int_REGULAR = Int(MadNLP.REGULAR)
    @. ws._term_gpu = ifelse(
        ws._ls_error > zero(Int32),
        Int_ERROR,
        ifelse(
            max(ws.inf_pr, ws.inf_du, ws.inf_compl) <= tol,
            Int_SOLVED,
            ifelse(
                (ws.inf_compl > div_tol * ws.best_complementarity) &
                (ws.dual_obj > max(ds * abs(ws.obj_val), one(T))),
                Int_INFEASIBLE,
                ifelse(
                    ws.obj_val < -(div_tol * max(ds * abs(ws.dual_obj), one(T))),
                    Int_DIVERGING,
                    Int_REGULAR,
                ),
            ),
        ),
    )
    minimum!(ws._any_nonregular_gpu, ws._term_gpu)
end

function update_termination_criteria!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.workspace
    opt = batch_solver.opt
    x, xl = batch_solver.x, batch_solver.xl
    zl = batch_solver.zl
    nlb = batch_solver.d.nlb

    get_inf_pr!(ws.inf_pr, MadNLP.full(batch_solver.c))
    @. ws.inf_pr /= max(one(T), ws.norm_b)

    get_inf_du!(ws.inf_du, MadNLP.full(batch_solver.f), MadNLP.full(zl),
                MadNLP.full(batch_solver.jacl))
    @. ws.inf_du /= max(one(T), ws.norm_c)

    get_inf_compl_lb!(ws.inf_compl, x, xl, zl, ws.sum_lb, nlb)
    @. ws.inf_compl /= max(one(T), ws.norm_c)
    @. ws.best_complementarity = min(ws.best_complementarity, ws.inf_compl)

    dual_objective!(ws.dual_obj, MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs),
        lower(zl), lower(xl), ws.sum_lb, nlb)

    compute_term_gpu!(ws, opt)
    return
end

function update_termination_status!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    opt = batch_solver.opt
    bcnt = batch_solver.batch_cnt
    bs = batch_solver.batch_size
    Int_REGULAR = Int64(Int(MadNLP.REGULAR))

    walltime_hit = time() - bcnt.start_time[] >= opt.max_wall_time
    max_iter_hit = walltime_hit ? false :
        any(ws.status[i] == MadNLP.REGULAR && bcnt.k[i] >= opt.max_iter for i in 1:bs)

    if !walltime_hit && !max_iter_hit
        copyto!(ws._any_nonregular_cpu, ws._any_nonregular_gpu)
        ws._any_nonregular_cpu[1] == Int_REGULAR && return false
    end

    copyto!(ws._term_cpu, ws._term_gpu)
    @inbounds for i in 1:bs
        ws.status[i] != MadNLP.REGULAR && continue
        code = MadNLP.Status(ws._term_cpu[i])
        if code != MadNLP.REGULAR
            ws.status[i] = code
        elseif bcnt.k[i] >= opt.max_iter
            ws.status[i] = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        elseif walltime_hit
            ws.status[i] = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
        end
    end
    return true
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

function increment_k!(batch_solver::AbstractBatchMPCSolver)
    bcnt = batch_solver.batch_cnt
    ws = batch_solver.workspace
    for i in 1:batch_solver.batch_size
        if ws.status[i] == MadNLP.REGULAR
            bcnt.k[i] += 1
        end
    end
end

function update_solution!(stats::BatchExecutionStats, batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    x, zl = batch_solver.x, batch_solver.zl

    stats.status .= ws.status
    stats.iter .= batch_solver.batch_cnt.k

    MadNLP.unpack_x!(stats.solution, bcb, x)
    MadNLP.unpack_y!(stats.multipliers, bcb, MadNLP.full(batch_solver.y))
    MadNLP.unpack_z!(stats.multipliers_L, bcb, MadNLP.variable(zl))
    fill!(stats.multipliers_U, zero(eltype(stats.multipliers_U)))
    unpack_obj!(stats.objective, bcb, ws.obj_val)
    MadNLP.unpack_cons!(stats.constraints, bcb, MadNLP.full(batch_solver.c), MadNLP.full(batch_solver.rhs), bcb.ind_ineq, MadNLP.slack(x))

    stats.dual_feas .= vec(ws.inf_du)
    stats.primal_feas .= vec(ws.inf_pr)
    stats.total_time .= batch_solver.batch_cnt.total_time
    return stats
end

function affine_direction!(solver::AbstractBatchMPCSolver)
    set_predictive_rhs!(solver, solver.kkt)
    solve_system!(solver.d, solver, solver.p)
end

function prediction_step!(solver::AbstractBatchMPCSolver)
    ws = solver.workspace
    affine_direction!(solver)

    fill!(ws.tau, one(eltype(ws.tau)))
    get_fraction_to_boundary_step!(solver)
    zero_inactive_step!(solver)
    get_affine_complementarity_measure!(solver, ws.alpha_p, ws.alpha_d)
    get_correction!(solver, MadNLP.full(solver.correction_lb))
    update_barrier!(solver.barrier_update, solver, ws.mu_affine)
    return
end

function mehrotra_correction_direction!(solver::AbstractBatchMPCSolver)
    set_correction_rhs!(solver, solver.kkt, solver.workspace.mu_batch, MadNLP.full(solver.correction_lb), nothing)
    solve_system!(solver.d, solver, solver.p)
    return
end

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


function apply_step!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, y, xl = batch_solver.x, batch_solver.y, batch_solver.xl
    zl, d = batch_solver.zl, batch_solver.d
    nlb = d.nlb

    MadNLP.full(x) .+= ws.alpha_p .* MadNLP.primal(d)
    MadNLP.full(y) .+= ws.alpha_d .* MadNLP.dual(d)
    if nlb > 0
        lower(zl) .+= ws.alpha_d .* MadNLP.dual_lb(d)
    end
    _adjust_boundary_active!(lower(x), lower(xl), ws.mu_batch, ws.active_mask)
    increment_k!(batch_solver)
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
    update_step!(batch_solver.step_rule, batch_solver)
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

function solve!(batch_solver::AbstractBatchMPCSolver{T, MT, VT}) where {T, MT, VT}
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    bs = batch_solver.batch_size

    nvar_nlp = bcb.nlp.meta.nvar
    ncon = bcb.ncon
    stats = BatchExecutionStats(MT, VT, nvar_nlp, ncon, bs)

    try
        MadNLP.@notice(batch_solver.logger, "MadIPM batch solve ($bs problems)\n")
        initialize!(batch_solver)
        mpc!(batch_solver)
    catch e
        for i in 1:bs
            if ws.status[i] == MadNLP.REGULAR
                ws.status[i] = MadNLP.INTERNAL_ERROR
            end
        end
        batch_solver.opt.rethrow_error && rethrow(e)
    finally
        bcnt = batch_solver.batch_cnt
        t_end = time()
        bcnt.total_time .= t_end .- bcnt.start_time[]
        update_solution!(stats, batch_solver)
        status_counts = Dict{MadNLP.Status, Int}()
        for i in 1:bs
            s = ws.status[i]
            status_counts[s] = get(status_counts, s, 0) + 1
        end
        for (s, cnt) in status_counts
            MadNLP.@notice(batch_solver.logger, "$(MadNLP.get_status_output(s, batch_solver.opt)): $cnt/$bs")
        end
    end

    return stats
end

"""
    stats = madipm_batch(bnlp::ObjRHSBatchQuadraticModel; kwargs...)

Solve a batch of LP/QP instances by reformulating each into standard form
(`Ax = b, z ≥ 0`) via [`standard_form`](@ref), running the batch IPM over the
shared std-form KKT, and recovering each primal/dual in the original space.

The input batch must share the Jacobian/Hessian sparsity and bound kinds
across instances (enforced by `standard_form`). Keyword arguments (other
than `regularization`, `step_rule`, `barrier_update`, `print_level`, etc.)
are forwarded to [`IPMOptions`](@ref).
"""
function madipm_batch(bnlp::ObjRHSBatchQuadraticModel; kwargs...)
    std_bnlp, ws_batch = standard_form(bnlp)
    batch_solver = UniformBatchMPCSolver(std_bnlp; kwargs...)
    std_stats = solve!(batch_solver)
    # Recover solution / multipliers in original space.
    nbatch = std_stats.solution |> size |> last
    orig_stats = BatchExecutionStats(typeof(bnlp.c_batch), typeof(bnlp.data.c), NLPModels.get_nvar(bnlp), NLPModels.get_ncon(bnlp), nbatch)
    copyto!(orig_stats.status, std_stats.status)
    recover_primal!(orig_stats.solution, ws_batch, std_stats.solution)
    recover_variable_multipliers!(orig_stats.multipliers_L, orig_stats.multipliers_U, ws_batch, std_stats.multipliers_L)
    BatchQuadraticModels._batch_gather_dual!(orig_stats.multipliers, ws_batch.con_start.row, std_stats.multipliers)
    # Per-instance objective in original space: f(x) = c0_batch + c' x_std + 1/2 x_std' Q x_std
    # But we can also just evaluate NLPModels.obj per column against the orig model.
    @inbounds for j in axes(orig_stats.solution, 2)
        orig_stats.dual_feas[j]   = std_stats.dual_feas[j]
        orig_stats.primal_feas[j] = std_stats.primal_feas[j]
        orig_stats.iter[j]        = std_stats.iter[j]
        orig_stats.total_time[j]  = std_stats.total_time[j]
    end
    # Objective in orig space: use std obj + workspace c0 offset removal.
    # std.obj = c_std' z + (1/2) z' Q_std z, with c0_std per instance in ws.c0_batch.
    # Orig.obj = std.obj shifted by -ws.c0_batch + orig.c0.
    copyto!(orig_stats.objective, std_stats.objective)
    # std_stats.objective already includes the std c0 (which we never set on std model).
    # Add the presolve shift so we get the true original objective.
    orig_stats.objective .+= ws_batch.c0_batch
    # Constraints in original space: cᵢ = A xᵢ (per instance). A is shared.
    mul!(orig_stats.constraints, bnlp.data.A, orig_stats.solution)
    return orig_stats
end

# Fallback for other batch NLP types — no std-form wrapping.
function madipm_batch(bnlp::NLPModels.AbstractBatchNLPModel; kwargs...)
    batch_solver = UniformBatchMPCSolver(bnlp; kwargs...)
    return solve!(batch_solver)
end

function IPMOptions(
    bnlp::NLPModels.AbstractBatchNLPModel{T};
    linear_solver = MadNLP.LDLSolver,
    kwargs...,
) where T
    return IPMOptions(; linear_solver = linear_solver, kwargs...)
end

function MadNLP.print_iter(batch_solver::AbstractBatchMPCSolver)
    logger = batch_solver.logger
    MadNLP.get_level(logger) > MadNLP.INFO && return
    ws = batch_solver.workspace
    bcnt = batch_solver.batch_cnt
    na = active_batch_size(batch_solver)
    bs = batch_solver.batch_size
    k = maximum(bcnt.k)

    active_str = "$na/$bs"
    mod(k, 10) == 0 && MadNLP.@info(logger, @sprintf(
        " iter  active  max_inf_pr  max_inf_du  max_inf_compl  max_alpha_p"))
    MadNLP.@info(logger, @sprintf(
        "%4i  ", k) * lpad(active_str, 6) * @sprintf(
        "   %6.2e     %6.2e      %7.2e      %6.2e",
        maximum(ws.inf_pr), maximum(ws.inf_du),
        maximum(ws.inf_compl), maximum(ws.alpha_p),
    ))
    return
end
