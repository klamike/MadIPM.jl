mutable struct StandardMPCSolver{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: MadNLP.AbstractKKTSystem{T},
    StdModel <: NLPModels.AbstractNLPModel{T,VT},
    OrigModel <: NLPModels.AbstractNLPModel,
    CB <: MadNLP.AbstractCallback{T},
} <: MadNLP.AbstractMadNLPSolver{T}
    original_nlp::OrigModel
    nlp::StdModel
    map::StandardFormMap{T}
    class::AbstractConicProblem
    cb::CB
    kkt::KKTSystem

    opt::IPMOptions
    cnt::MadNLP.MadNLPCounters
    logger::MadNLP.MadNLPLogger

    n::Int
    m::Int
    nlb::Int

    x::MadNLP.PrimalVector{T, VT, VI}
    y::VT
    zl::MadNLP.PrimalVector{T, VT, VI}

    obj_val::T
    f::MadNLP.PrimalVector{T, VT, VI}
    c::VT
    jacl::VT

    d::MadNLP.UnreducedKKTVector{T, VT}
    p::MadNLP.UnreducedKKTVector{T, VT}

    _w1::MadNLP.UnreducedKKTVector{T, VT}
    _w2::MadNLP.UnreducedKKTVector{T, VT}

    correction_lb::VT
    rhs::VT
    ind_ineq::VI
    ind_lb::VI

    x_lr::MadNLP.SubVector{T,VT,VI}
    zl_r::MadNLP.SubVector{T,VT,VI}
    dx_lr::MadNLP.SubVector{T,VT,VI}

    inf_pr::T
    inf_du::T
    inf_compl::T
    norm_b::T
    norm_c::T

    mu::T

    alpha_p::T
    alpha_d::T
    del_w::T
    del_c::T
    best_complementarity::T
    mu_curr::T
    status::MadNLP.Status
end

_identity_standard_map(::Type{T}, n::Int, m::Int) where {T} = StandardFormMap(
    zeros(T, n),
    SparseArrays.sparse(1:n, 1:n, ones(T, n), n, n),
    collect(1:m),
    collect(1:n),
    zeros(Int, n),
)

function _standardize_model(nlp::StandardLinearModel{T,VT}) where {T,VT}
    return nlp, _identity_standard_map(T, NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)), nlp
end

function _standardize_model(nlp::StandardQuadraticModel{T,VT}) where {T,VT}
    return nlp, _identity_standard_map(T, NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)), nlp
end

function _standardize_model(nlp::LinearModel)
    std, map = standard_form(nlp)
    return std, map, nlp
end

function _standardize_model(nlp::QuadraticModel)
    std, map = standard_form(nlp)
    return std, map, nlp
end

function _load_standard_options(nlp; kwargs...)
    if any(first(pair) == :kkt_system for pair in kwargs)
        return load_options(nlp; kwargs...)
    elseif nlp isa StandardLinearModel
        return load_options(nlp; kkt_system = NormalKKTSystem, kwargs...)
    else
        return load_options(nlp; kwargs...)
    end
end

function StandardMPCSolver(nlp::Union{LinearModel, QuadraticModel, StandardLinearModel, StandardQuadraticModel}; kwargs...)
    std_nlp, map, original_nlp = _standardize_model(nlp)
    options = _load_standard_options(std_nlp; kwargs...)
    VT = typeof(NLPModels.get_x0(std_nlp))

    ipm_opt = options.interior_point
    logger = options.logger
    cnt = MadNLP.MadNLPCounters(start_time=time())

    cb = MadNLP.create_callback(
        MadNLP.SparseCallback,
        std_nlp;
        fixed_variable_treatment = ipm_opt.fixed_variable_treatment,
        equality_treatment = ipm_opt.equality_treatment,
    )

    ind_lb = cb.ind_lb
    empty_ind = similar(ind_lb, 0)
    ns = 0
    nx = MadNLP.n_variables(cb)
    n = nx + ns
    m = NLPModels.get_ncon(std_nlp)
    nlb = length(ind_lb)

    kkt = MadNLP.create_kkt_system(
        ipm_opt.kkt_system,
        cb,
        ipm_opt.linear_solver;
        opt_linear_solver = options.linear_solver,
    )

    x = MadNLP.PrimalVector(VT, nx, ns, ind_lb, empty_ind)
    zl = MadNLP.PrimalVector(VT, nx, ns, ind_lb, empty_ind)
    f = MadNLP.PrimalVector(VT, nx, ns, ind_lb, empty_ind)

    d = MadNLP.UnreducedKKTVector(VT, n, m, nlb, 0, ind_lb, empty_ind)
    p = MadNLP.UnreducedKKTVector(VT, n, m, nlb, 0, ind_lb, empty_ind)
    _w1 = MadNLP.UnreducedKKTVector(VT, n, m, nlb, 0, ind_lb, empty_ind)
    _w2 = MadNLP.UnreducedKKTVector(VT, n, m, nlb, 0, ind_lb, empty_ind)

    correction_lb = VT(undef, nlb)
    jacl = VT(undef, n)
    y = VT(undef, m)
    c = VT(undef, m)
    rhs = VT(undef, m)

    x_lr = view(full(x), ind_lb)
    zl_r = view(full(zl), ind_lb)
    dx_lr = view(d.xp, ind_lb)

    cnt.init_time = time() - cnt.start_time
    class = iszero(MadNLP.get_nnzh(std_nlp)) ? LinearProgram() : QuadraticProgram()

    return StandardMPCSolver(
        original_nlp,
        std_nlp,
        map,
        class,
        cb,
        kkt,
        ipm_opt,
        cnt,
        logger,
        n,
        m,
        nlb,
        x,
        y,
        zl,
        zero(eltype(y)),
        f,
        c,
        jacl,
        d,
        p,
        _w1,
        _w2,
        correction_lb,
        rhs,
        empty_ind,
        ind_lb,
        x_lr,
        zl_r,
        dx_lr,
        zero(eltype(y)),
        zero(eltype(y)),
        zero(eltype(y)),
        zero(eltype(y)),
        zero(eltype(y)),
        zero(eltype(y)),
        zero(eltype(y)),
        zero(eltype(y)),
        zero(eltype(y)),
        zero(eltype(y)),
        typemax(eltype(y)),
        zero(eltype(y)),
        MadNLP.INITIAL,
    )
end

MPCSolver(nlp::Union{LinearModel, QuadraticModel, StandardLinearModel, StandardQuadraticModel}; kwargs...) = StandardMPCSolver(nlp; kwargs...)

function MadNLP.print_iter(solver::StandardMPCSolver; options...)
    obj_scale = solver.cb.obj_scale[]
    mod(solver.cnt.k, 10) == 0 && MadNLP.@info(solver.logger, @sprintf(
        "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr"))
    MadNLP.@info(solver.logger, Printf.@sprintf(
        "%4i%s% 10.7e %6.2e %6.2e %5.1f %6.2e %s %6.2e %6.2e",
        solver.cnt.k,
        " ",
        solver.obj_val / obj_scale,
        solver.inf_pr, solver.inf_du, log10(solver.mu),
        solver.cnt.k == 0 ? 0.0 : norm(MadNLP.primal(solver.d), Inf),
        solver.del_w == 0 ? "   - " : @sprintf("%5.1f", log(10, solver.del_w)),
        solver.alpha_d, solver.alpha_p))
    return
end

function set_initial_primal_rhs!(solver::StandardMPCSolver)
    fill!(full(solver.p), 0.0)
    MadNLP.dual(solver.p) .= .-solver.c
    return
end

function set_initial_dual_rhs!(solver::StandardMPCSolver)
    fill!(full(solver.p), 0.0)
    MadNLP.primal(solver.p) .= .-MadNLP.primal(solver.f)
    return
end

function init_starting_point!(solver::StandardMPCSolver)
    x = MadNLP.primal(solver.x)
    z = solver.zl_r
    res = solver.jacl

    solver.kkt.reg .= solver.del_w
    solver.kkt.pr_diag .= solver.del_w
    solver.kkt.du_diag .= solver.del_c

    MadNLP.factorize_wrapper!(solver)

    set_initial_primal_rhs!(solver)
    solve_system!(solver.d, solver, solver.p)
    axpy!(1.0, MadNLP.primal(solver.d), x)

    set_initial_dual_rhs!(solver)
    solve_system!(solver.d, solver, solver.p)
    solver.y .= MadNLP.dual(solver.d)

    MadNLP.jtprod!(res, solver.kkt, solver.y)
    axpy!(1.0, MadNLP.primal(solver.f), res)
    copyto!(solver.zl.values, res)

    delta_x = max(0.0, -1.5 * minimum(x; init = 0.0))
    delta_z = max(0.0, -1.5 * minimum(z; init = 0.0))

    x .+= delta_x
    z .+= 1.0 + delta_z

    μ = isempty(z) ? zero(eltype(z)) : dot(x, z)
    sumz = sum(z)
    sumx = sum(x)
    delta_x2 = iszero(sumz) ? zero(eltype(z)) : μ / (2 * sumz)
    delta_z2 = iszero(sumx) ? zero(eltype(z)) : μ / (2 * sumx)

    x .+= delta_x2
    z .+= delta_z2
    return
end

function initialize!(solver::StandardMPCSolver{T}) where {T}
    x = MadNLP.variable(solver.x)
    x .= max.(NLPModels.get_x0(solver.nlp), T(solver.opt.bound_push))
    solver.y .= NLPModels.get_y0(solver.nlp)
    solver.rhs .= NLPModels.get_lcon(solver.nlp)
    fill!(solver.jacl, zero(T))

    MadNLP.initialize!(solver.kkt)
    init_regularization!(solver, solver.opt.regularization)

    solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
    MadNLP.eval_jac_wrapper!(solver, solver.kkt, solver.x)
    MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)
    MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
    MadNLP.eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    solver.norm_b = norm(solver.rhs, Inf)
    solver.norm_c = norm(MadNLP.primal(solver.f), Inf)

    init_starting_point!(solver)

    solver.mu = solver.opt.mu_init
    solver.cnt.start_time = time()
    solver.best_complementarity = typemax(typeof(solver.best_complementarity))
    solver.status = MadNLP.REGULAR
    MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)
    return
end

function set_predictive_rhs!(solver::StandardMPCSolver, kkt::MadNLP.AbstractKKTSystem)
    px = MadNLP.primal(solver.p)
    py = MadNLP.dual(solver.p)
    pzl = MadNLP.dual_lb(solver.p)
    f = MadNLP.primal(solver.f)
    fill!(MadNLP.full(solver.p), 0.0)
    px .= .-f .+ MadNLP.full(solver.zl) .- solver.jacl
    py .= .-solver.c
    pzl .= .-solver.x_lr .* solver.zl_r
    return
end

function set_correction_rhs!(solver::StandardMPCSolver, kkt::MadNLP.AbstractKKTSystem, mu::Float64, correction_lb::AbstractVector{Float64})
    px = MadNLP.primal(solver.p)
    py = MadNLP.dual(solver.p)
    pzl = MadNLP.dual_lb(solver.p)
    px .= .-MadNLP.primal(solver.f) .+ MadNLP.full(solver.zl) .- solver.jacl
    py .= .-solver.c
    pzl .= .-solver.x_lr .* solver.zl_r .+ mu .- correction_lb
    return
end

function get_correction!(solver::StandardMPCSolver, correction_lb)
    correction_lb .= solver.dx_lr .* MadNLP.dual_lb(solver.d)
    return
end

function set_extra_correction!(
    solver::StandardMPCSolver,
    correction_lb,
    alpha_p,
    alpha_d,
    βmin,
    βmax,
    μ,
)
    dlb = MadNLP.dual_lb(solver.d)
    tmin, tmax = βmin * μ, βmax * μ
    map!(
        (x, dx, z, dz, corr) -> begin
            xv = x + alpha_p * dx
            zv = z + alpha_d * dz
            v = xv * zv
            δ = if v < tmin
                tmin - v
            elseif v > tmax
                tmax - v
            else
                0.0
            end
            corr - δ
        end,
        correction_lb,
        solver.x_lr, solver.dx_lr, solver.zl_r, dlb, correction_lb,
    )
    return
end

function set_aug_diagonal_reg!(kkt::MadNLP.AbstractKKTSystem{T}, solver::StandardMPCSolver{T}) where {T}
    fill!(kkt.reg, solver.del_w)
    fill!(kkt.du_diag, solver.del_c)
    kkt.l_diag .= .-solver.x_lr
    copyto!(kkt.l_lower, solver.zl_r)
    copyto!(kkt.pr_diag, kkt.reg)
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    return
end

function set_aug_diagonal_reg!(kkt::MadNLP.ScaledSparseKKTSystem{T}, solver::StandardMPCSolver{T}) where {T}
    fill!(kkt.reg, solver.del_w)
    fill!(kkt.du_diag, solver.del_c)
    kkt.l_diag .= solver.x_lr
    copyto!(kkt.l_lower, solver.zl_r)
    MadNLP._set_aug_diagonal!(kkt)
    return
end

function get_complementarity_measure(solver::StandardMPCSolver)
    isempty(solver.x_lr) && return 0.0
    return mapreduce(*, +, solver.x_lr, solver.zl_r; init = zero(eltype(solver.x_lr))) / length(solver.x_lr)
end

function update_barrier!(rule::Mehrotra, solver::StandardMPCSolver, mu_affine)
    mu_curr = get_complementarity_measure(solver)
    sigma = if solver.nlb > 0
        clamp((mu_affine / mu_curr)^3, 1e-6, 10.0)
    else
        1.0
    end
    solver.mu = max(solver.opt.mu_min, sigma * mu_curr)
    return mu_curr
end

function get_affine_complementarity_measure(solver::StandardMPCSolver, alpha_p, alpha_d)
    isempty(solver.x_lr) && return 0.0
    return mapreduce(
        (x, dx, z, dz) -> (x + alpha_p * dx) * (z + alpha_d * dz),
        +,
        solver.x_lr, solver.dx_lr, solver.zl_r, MadNLP.dual_lb(solver.d);
        init = zero(eltype(solver.x_lr)),
    ) / length(solver.x_lr)
end

function get_alpha_max_primal_std(x, dx, tau)
    return mapreduce(
        (dxi, xi, i) -> begin
            val = (dxi < 0) ? (-xi) * tau / dxi : Inf
            (val, i)
        end,
        (a, b) -> a[1] < b[1] ? a : b,
        dx, x, eachindex(x);
        init = (1.0, 0),
    )
end

function get_alpha_max_dual_std(z, dz, tau)
    return mapreduce(
        (dzi, zi, i) -> begin
            val = (dzi < 0) ? (-zi) * tau / dzi : Inf
            (val, i)
        end,
        (a, b) -> a[1] < b[1] ? a : b,
        dz, z, eachindex(z);
        init = (1.0, 0),
    )
end

function get_fraction_to_boundary_step(solver::StandardMPCSolver, tau)
    alpha_x, _ = get_alpha_max_primal_std(solver.x_lr, solver.dx_lr, tau)
    alpha_z, _ = get_alpha_max_dual_std(solver.zl_r, MadNLP.dual_lb(solver.d), tau)
    return alpha_x, alpha_z
end

function update_step!(rule::Union{ConservativeStep, AdaptiveStep}, solver::StandardMPCSolver)
    tau = get_tau(rule, solver)
    solver.alpha_p, solver.alpha_d = get_fraction_to_boundary_step(solver, tau)
    return
end

function update_step!(rule::MehrotraAdaptiveStep, solver::StandardMPCSolver)
    gamma_a = 1.0 / (1.0 - rule.gamma_f)
    d_zl = MadNLP.dual_lb(solver.d)
    alpha_x, i_x = get_alpha_max_primal_std(solver.x_lr, solver.dx_lr, 1.0)
    alpha_z, i_z = get_alpha_max_dual_std(solver.zl_r, d_zl, 1.0)

    mu_full = get_affine_complementarity_measure(solver, alpha_x, alpha_z) / gamma_a
    alpha_p = 1.0
    alpha_d = 1.0

    if alpha_x < 1.0
        tmp = mu_full / (solver.zl_r[i_x] + alpha_z * d_zl[i_x])
        alpha_p = (solver.x_lr[i_x] - tmp) / (-solver.dx_lr[i_x])
    end
    if alpha_z < 1.0
        tmp = mu_full / (solver.x_lr[i_z] + alpha_x * solver.dx_lr[i_z])
        alpha_d = -(solver.zl_r[i_z] - tmp) / d_zl[i_z]
    end

    solver.alpha_p = max(alpha_p, rule.gamma_f * alpha_x)
    solver.alpha_d = max(alpha_d, rule.gamma_f * alpha_z)
    return
end

function mehrotra_correction_direction!(solver::StandardMPCSolver)
    set_correction_rhs!(solver, solver.kkt, solver.mu, solver.correction_lb)
    solve_system!(solver.d, solver, solver.p)
    return
end

function prediction_step!(solver::StandardMPCSolver)
    affine_direction!(solver)
    alpha_aff_p, alpha_aff_d = get_fraction_to_boundary_step(solver, 1.0)
    mu_affine = get_affine_complementarity_measure(solver, alpha_aff_p, alpha_aff_d)
    get_correction!(solver, solver.correction_lb)
    solver.mu_curr = update_barrier!(solver.opt.barrier_update, solver, mu_affine)
    return
end

function gondzio_correction_direction!(solver::StandardMPCSolver)
    solver.opt.max_ncorr ≤ 0 && return

    δ = 0.1
    βmin = 0.1
    βmax = 10.0
    tau = 0.995
    Δp = solver._w2.values
    alpha_p, alpha_d = get_fraction_to_boundary_step(solver, tau)

    for _ in 1:solver.opt.max_ncorr
        tilde_alpha_p = min(alpha_p + δ, 1.0)
        tilde_alpha_d = min(alpha_d + δ, 1.0)
        ga = get_affine_complementarity_measure(solver, tilde_alpha_p, tilde_alpha_d)
        g = solver.mu_curr
        mu = (ga / g)^2 * ga
        set_extra_correction!(solver, solver.correction_lb, tilde_alpha_p, tilde_alpha_d, βmin, βmax, mu)
        set_correction_rhs!(solver, solver.kkt, mu, solver.correction_lb)
        copyto!(Δp, solver.d.values)
        solve_system!(solver.d, solver, solver.p)
        hat_alpha_p, hat_alpha_d = get_fraction_to_boundary_step(solver, tau)
        if (hat_alpha_p < 1.005 * alpha_p) || (hat_alpha_d < 1.005 * alpha_d)
            copyto!(solver.d.values, Δp)
            break
        else
            alpha_p = hat_alpha_p
            alpha_d = hat_alpha_d
        end
    end
    return alpha_p, alpha_d
end

function apply_step!(solver::StandardMPCSolver)
    axpy!(solver.alpha_p, MadNLP.primal(solver.d), MadNLP.primal(solver.x))
    axpy!(solver.alpha_d, MadNLP.dual(solver.d), solver.y)
    solver.zl_r .+= solver.alpha_d .* MadNLP.dual_lb(solver.d)
    solver.cnt.k += 1
    return
end

function dual_objective(solver::StandardMPCSolver)
    return -dot(solver.y, solver.rhs)
end

function get_optimality_gap(solver::StandardMPCSolver)
    isempty(solver.x_lr) && return 0.0
    return mapreduce(*, +, solver.x_lr, solver.zl_r; init = zero(eltype(solver.x_lr)))
end

function update_termination_criteria!(solver::StandardMPCSolver)
    dobj = dual_objective(solver)
    solver.inf_pr = MadNLP.get_inf_pr(solver.c) / max(1.0, solver.norm_b)
    solver.inf_du = norm(MadNLP.primal(solver.f) .+ solver.jacl .- MadNLP.full(solver.zl), Inf) / max(1.0, solver.norm_c)
    solver.inf_compl = get_optimality_gap(solver) / max(1.0, solver.norm_c)
    solver.best_complementarity = min(solver.best_complementarity, solver.inf_compl)

    if max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= solver.opt.tol
        solver.status = MadNLP.SOLVE_SUCCEEDED
    elseif ((solver.inf_compl > solver.opt.divergence_tol * solver.best_complementarity) &&
            (dobj > max(solver.opt.divergence_scale * abs(solver.obj_val), 1.0)))
        solver.status = MadNLP.INFEASIBLE_PROBLEM_DETECTED
    elseif solver.obj_val < -solver.opt.divergence_tol * max(solver.opt.divergence_scale * abs(dobj), 1.0)
        solver.status = MadNLP.DIVERGING_ITERATES
    elseif solver.cnt.k >= solver.opt.max_iter
        solver.status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    elseif time() - solver.cnt.start_time >= solver.opt.max_wall_time
        solver.status = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
    end
    return
end

function evaluate_model!(solver::StandardMPCSolver)
    solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
    MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
    MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)
    MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)
    return
end

function init_regularization!(solver::StandardMPCSolver, ::NoRegularization)
    solver.del_w = 1.0
    solver.del_c = 0.0
    return
end

function update_regularization!(solver::StandardMPCSolver, ::NoRegularization)
    solver.del_w = 0.0
    solver.del_c = 0.0
    return
end

function init_regularization!(solver::StandardMPCSolver, reg::FixedRegularization)
    solver.del_w = 1.0
    solver.del_c = reg.delta_d
    return
end

function update_regularization!(solver::StandardMPCSolver, reg::FixedRegularization)
    solver.del_w = reg.delta_p
    solver.del_c = reg.delta_d
    return
end

function init_regularization!(solver::StandardMPCSolver, reg::AdaptiveRegularization)
    solver.del_w = 1.0
    solver.del_c = reg.delta_d
    return
end

function update_regularization!(solver::StandardMPCSolver, reg::AdaptiveRegularization)
    reg.delta_p = max(reg.delta_p / 10.0, reg.delta_min)
    reg.delta_d = min(reg.delta_d / 10.0, -reg.delta_min)
    solver.del_w = reg.delta_p
    solver.del_c = reg.delta_d
    return
end

function MadNLP.MadNLPExecutionStats(solver::StandardMPCSolver{T}) where {T}
    n = NLPModels.get_nvar(solver.original_nlp)
    m = NLPModels.get_ncon(solver.original_nlp)
    x_template = NLPModels.get_x0(solver.original_nlp)
    y_template = NLPModels.get_y0(solver.original_nlp)
    solution = similar(x_template, n)
    constraints = similar(y_template, m)
    multipliers = similar(y_template, m)
    multipliers_L = similar(x_template, n)
    multipliers_U = similar(x_template, n)
    return MadNLP.MadNLPExecutionStats(
        solver.opt,
        solver.status,
        solution,
        zero(T),
        constraints,
        zero(T),
        zero(T),
        multipliers,
        multipliers_L,
        multipliers_U,
        0,
        solver.cnt,
    )
end

function update_solution!(stats::MadNLP.MadNLPExecutionStats, solver::StandardMPCSolver)
    stats.status = solver.status
    recover_primal!(stats.solution, solver.map, MadNLP.variable(solver.x))
    fill!(stats.multipliers, zero(eltype(stats.multipliers)))
    @inbounds for i in eachindex(stats.multipliers)
        row = solver.map.constraint_rows[i]
        row > 0 && (stats.multipliers[i] = solver.y[row])
    end
    recover_variable_multipliers!(stats.multipliers_L, stats.multipliers_U, solver.map, MadNLP.variable(solver.zl))
    NLPModels.cons!(solver.original_nlp, stats.solution, stats.constraints)
    stats.objective = NLPModels.obj(solver.original_nlp, stats.solution)
    stats.dual_feas = solver.inf_du
    stats.primal_feas = solver.inf_pr
    stats.iter = solver.cnt.k
    return stats
end
