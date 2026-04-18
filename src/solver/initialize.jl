# Solver initialization: warm-start primal/dual variables and prepare
# scratch state. Stays specialized: scalar uses `axpy!`/scalar arithmetic
# and reads single buffers; batch uses broadcast / `batch_mapreduce!` and
# manages active-set bookkeeping.

# ---------- scalar ----------

function init_starting_point!(solver::MPCSolver)
    problem = solver.problem
    state = solver.state
    T = eltype(state.y)
    x = MadNLP.primal(state.x)
    z = state.zl_r
    res = state.jacl

    problem.kkt.reg .= state.del_w
    problem.kkt.pr_diag .= state.del_w
    problem.kkt.du_diag .= state.del_c

    MadNLP.factorize_wrapper!(solver)

    set_initial_primal_rhs!(solver)
    solve_system!(state.d, solver, state.p)
    axpy!(one(T), MadNLP.primal(state.d), x)

    set_initial_dual_rhs!(solver)
    solve_system!(state.d, solver, state.p)
    state.y .= MadNLP.dual(state.d)

    MadNLP.jtprod!(res, problem.kkt, state.y)
    axpy!(one(T), MadNLP.primal(state.f), res)
    copyto!(state.zl.values, res)

    delta_x = max(zero(T), -T(1.5) * minimum(x; init = zero(T)))
    delta_z = max(zero(T), -T(1.5) * minimum(z; init = zero(T)))

    x .+= delta_x
    z .+= one(T) + delta_z

    μ = isempty(z) ? zero(eltype(z)) : dot(x, z)
    sumz = sum(z)
    sumx = sum(x)
    delta_x2 = iszero(sumz) ? zero(eltype(z)) : μ / (T(2) * sumz)
    delta_z2 = iszero(sumx) ? zero(eltype(z)) : μ / (T(2) * sumx)

    x .+= delta_x2
    z .+= delta_z2
    return
end

function initialize!(solver::MPCSolver{T}) where {T}
    problem = solver.problem
    state = solver.state
    x = MadNLP.variable(state.x)
    x .= max.(NLPModels.get_x0(problem.nlp), T(problem.opt.bound_push))
    state.y .= NLPModels.get_y0(problem.nlp)
    state.rhs .= NLPModels.get_lcon(problem.nlp)
    fill!(state.jacl, zero(T))

    MadNLP.initialize!(problem.kkt)
    init_regularization!(solver, problem.regularization)

    state.obj_val = MadNLP.eval_f_wrapper(solver, state.x)
    MadNLP.eval_jac_wrapper!(solver, problem.kkt, state.x)
    MadNLP.eval_grad_f_wrapper!(solver, state.f, state.x)
    MadNLP.eval_cons_wrapper!(solver, state.c, state.x)
    MadNLP.eval_lag_hess_wrapper!(solver, problem.kkt, state.x, state.y)

    state.norm_b = norm(state.rhs, Inf)
    state.norm_c = norm(MadNLP.primal(state.f), Inf)

    init_starting_point!(solver)

    state.mu = T(problem.opt.mu_init)
    state.best_complementarity = typemax(typeof(state.best_complementarity))
    state.status = MadNLP.REGULAR
    MadNLP.jtprod!(state.jacl, problem.kkt, state.y)
    return
end

# ---------- batch ----------

function init_starting_point!(batch_solver::AbstractBatchMPCSolver{T}) where T
    bkkt = batch_solver.kkt
    bs = batch_solver.batch_size
    n = batch_solver.d.n
    m = batch_solver.d.m

    bx, bxl, bxu = batch_solver.x, batch_solver.xl, batch_solver.xu
    bzl, bzu = batch_solver.zl, batch_solver.zu
    x = MadNLP.primal(bx)
    l, u = MadNLP.full(bxl), MadNLP.full(bxu)
    lb, ub = lower(bxl), upper(bxu)
    zl, zu = lower(bzl), upper(bzu)
    xl, xu = lower(bx), upper(bx)
    # use jacl as a buffer
    res = MadNLP.full(batch_solver.jacl)

    # Add initial primal-dual regularization
    bkkt.reg .= batch_solver.del_w
    pr_diag(bkkt) .= batch_solver.del_w
    du_diag(bkkt) .= batch_solver.del_c

    # Step 0: factorize initial KKT system
    MadNLP.factorize_wrapper!(batch_solver)

    # Step 1: Compute initial primal variable as x0 = x + dx, with dx the
    #         least square solution of the system A * dx = (b - A*x)
    set_initial_primal_rhs!(batch_solver)
    solve_system!(batch_solver.d, batch_solver, batch_solver.p)
    # x0 = x + dx
    x .+= MadNLP.primal(batch_solver.d)

    # Step 2: Compute initial dual variable as the least square solution of A' * y = -f
    set_initial_dual_rhs!(batch_solver)
    solve_system!(batch_solver.d, batch_solver, batch_solver.p)
    MadNLP.full(batch_solver.y) .= MadNLP.dual(batch_solver.d)

    # Step 3: init bounds multipliers using c + A' * y - zl + zu = 0
    # A' * y
    MadNLP.jtprod!(res, bkkt, batch_solver.y)
    # A'*y + c
    res .+= MadNLP.primal(batch_solver.f)
    # Initialize bounds multipliers
    map!(
        (r_, l_, u_, zl_) -> begin
            val = if isfinite(l_) && isfinite(u_)
                0.5 * r_
            elseif isfinite(l_)
                r_
            else
                zl_
            end
            val
        end,
        MadNLP.full(batch_solver.zl), res, l, u, MadNLP.full(batch_solver.zl),
    )
    map!(
        (r_, l_, u_, zu_) -> begin
            val = if isfinite(l_) && isfinite(u_)
                -0.5 * r_
            elseif isfinite(u_)
                -r_
            else
                zu_
            end
            val
        end,
        MadNLP.full(batch_solver.zu), res, l, u, MadNLP.full(batch_solver.zu),
    )

    ws = batch_solver.workspace
    nlb_init, nub_init = batch_solver.d.nlb, batch_solver.d.nub
    bs = batch_solver.batch_size
    _s1 = ws.alpha_xl  # (1,bs) scratch
    _s2 = ws.alpha_xu  # (1,bs) scratch

    # delta_x = max(0, -1.5 * min(xl-lb, 0), -1.5 * min(ub-xu, 0))
    if nlb_init > 0
        batch_mapreduce!(-, min, T(Inf), _s1, xl, lb)
        @. _s1 = min(_s1, zero(T))
    else
        fill!(_s1, zero(T))
    end
    if nub_init > 0
        batch_mapreduce!(-, min, T(Inf), _s2, ub, xu)
        @. _s2 = min(_s2, zero(T))
    else
        fill!(_s2, zero(T))
    end
    delta_x = ws.mu_batch  # (1,bs) scratch
    @. delta_x = max(zero(T), T(-1.5) * _s1, T(-1.5) * _s2)

    # delta_s = max(0, -1.5 * min(zl, 0), -1.5 * min(zu, 0))
    if nlb_init > 0
        batch_mapreduce!(identity, min, T(Inf), _s1, zl)
        @. _s1 = min(_s1, zero(T))
    else
        fill!(_s1, zero(T))
    end
    if nub_init > 0
        batch_mapreduce!(identity, min, T(Inf), _s2, zu)
        @. _s2 = min(_s2, zero(T))
    else
        fill!(_s2, zero(T))
    end
    delta_s = ws.mu_curr  # (1,bs) scratch
    @. delta_s = max(zero(T), T(-1.5) * _s1, T(-1.5) * _s2)

    xl .+= delta_x
    xu .-= delta_x
    zl .+= 1.0 .+ delta_s
    zu .+= 1.0 .+ delta_s

    # μ = sum((xl-lb)*zl) + sum((ub-xu)*zu)
    μ = ws.mu_affine  # (1,bs) scratch
    fill!(μ, zero(T))
    if nlb_init > 0
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_lb, xl, zl)
        μ .+= ws.sum_lb
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_lb, lb, zl)
        μ .-= ws.sum_lb
    end
    if nub_init > 0
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_ub, ub, zu)
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_lb, xu, zu)
        ws.sum_ub .-= ws.sum_lb
        μ .+= ws.sum_ub
    end

    # delta_x2 = μ / (2 * (sum(zl) + sum(zu)))
    if nlb_init > 0
        batch_mapreduce!(identity, +, zero(T), ws.sum_lb, zl)
    else
        fill!(ws.sum_lb, zero(T))
    end
    if nub_init > 0
        batch_mapreduce!(identity, +, zero(T), ws.sum_ub, zu)
    else
        fill!(ws.sum_ub, zero(T))
    end
    delta_x2 = _s1  # reuse (1,bs) scratch
    @. delta_x2 = μ / (2 * (ws.sum_lb + ws.sum_ub))

    # delta_s2 = μ / (2 * (sum(xl-lb) + sum(ub-xu)))
    if nlb_init > 0
        batch_mapreduce!(-, +, zero(T), ws.sum_lb, xl, lb)
    else
        fill!(ws.sum_lb, zero(T))
    end
    if nub_init > 0
        batch_mapreduce!(-, +, zero(T), ws.sum_ub, ub, xu)
    else
        fill!(ws.sum_ub, zero(T))
    end
    delta_s2 = _s2  # reuse (1,bs) scratch
    @. delta_s2 = μ / (2 * (ws.sum_lb + ws.sum_ub))

    xl .+= delta_x2
    xu .-= delta_x2
    zl .+= delta_s2
    zu .+= delta_s2

    # Use Ipopt's heuristic to project x back on the interval [l, u]
    kappa = batch_solver.opt.bound_fac
    map!(
        (l_, u_, x_) -> begin
            out = if x_ < l_
                pl = min(kappa * max(1.0, l_), kappa * (u_ - l_))
                l_ + pl
            elseif u_ < x_
                pu = min(kappa * max(1.0, u_), kappa * (u_ - l_))
                u_ - pu
            else
                x_
            end
            out
        end,
        x, l, u, x,
    )
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
        batch_solver.xu,
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
            batch_solver.xu,
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
