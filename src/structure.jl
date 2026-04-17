mutable struct MPCProblem{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: MadNLP.AbstractKKTSystem{T},
    StdModel <: NLPModels.AbstractNLPModel{T,VT},
    OrigModel <: NLPModels.AbstractNLPModel,
    CB <: MadNLP.AbstractCallback{T},
    WS,
    OPT <: IPMOptions,
    REG <: AbstractRegularization,
    STEP <: AbstractStepRule,
    BARR <: AbstractBarrierUpdate,
}
    original_nlp::OrigModel
    nlp::StdModel
    workspace::WS
    cb::CB
    kkt::KKTSystem
    opt::OPT
    regularization::REG
    step_rule::STEP
    barrier_update::BARR
    logger::MadNLP.MadNLPLogger
    n::Int
    m::Int
    nlb::Int
    ind_lb::VI
end

mutable struct MPCState{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
} 
    cnt::MadNLP.MadNLPCounters

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

mutable struct MPCSolver{T, P <: MPCProblem{T}, S <: MPCState{T}} <: MadNLP.AbstractMadNLPSolver{T}
    problem::P
    state::S
end

function MPCSolver(nlp::Union{LinearModel, QuadraticModel}; kwargs...)
    std_nlp, workspace = standard_form(nlp)
    options = load_options(std_nlp; kwargs...)
    VT = typeof(NLPModels.get_x0(std_nlp))

    ipm_opt = options.interior_point
    logger = options.logger
    regularization = options.regularization
    step_rule = options.step_rule
    barrier_update = options.barrier_update
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
    problem = MPCProblem(
        nlp,
        std_nlp,
        workspace,
        cb,
        kkt,
        ipm_opt,
        regularization,
        step_rule,
        barrier_update,
        logger,
        n,
        m,
        nlb,
        ind_lb,
    )
    T = eltype(y)
    z = zero(T)
    state = MPCState(
        cnt,
        x, y, zl,
        z,
        f, c, jacl,
        d, p, _w1, _w2, correction_lb, rhs, empty_ind,
        x_lr, zl_r, dx_lr,
        z, z, z, z, z,
        z,
        z, z, z, z,
        typemax(T),
        z,
        MadNLP.INITIAL,
    )
    return MPCSolver(problem, state)
end

"""
    update!(solver::MPCSolver; c, c0, A, Q, lvar, uvar, lcon, ucon, x0, y0)

Mutate the original model held by `solver` with any provided fields (all in
the original variable/constraint space) and propagate the minimal set of
updates to the standard-form model. Call `MadIPM.solve!(solver)` afterwards
to solve the updated problem without rebuilding.

Sparsity patterns and bound kinds (finite ↔ infinite, `l == u`) must be
unchanged; for structural changes construct a new `MPCSolver`.
"""
function update!(solver::MPCSolver; kwargs...)
    problem = solver.problem
    update_standard_form!(problem.original_nlp, problem.nlp, problem.workspace; kwargs...)
    return solver
end
