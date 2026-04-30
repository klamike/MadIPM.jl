"""
    MPCProblem{...}

Static problem data for one scalar MPC solve: the original NLP, the
standard-form NLP actually being solved, the presolve workspace linking
them, the KKT system, logger, and the three policy objects (regularization,
step rule, barrier update). Mutated only by `update!(solver; ...)`.
"""
mutable struct MPCProblem{
    T,
    KKTSystem <: MadNLP.AbstractKKTSystem{T},
    StdModel <: NLPModels.AbstractNLPModel{T},
    OrigModel <: NLPModels.AbstractNLPModel,
    CB <: MadNLP.AbstractCallback{T},
    WS,
    OPT <: IPMOptions,
    REG <: AbstractRegularization,
    STEP <: AbstractStepRule,
    BARR <: AbstractBarrierUpdate,
    SCL <: AbstractScaler,
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
    scaler::SCL
    logger::MadNLP.MadNLPLogger
    nlb::Int
end

"""
    MPCState{T, VT, VI}

Mutable iterate state for the scalar MPC solve: primal/dual iterates
(`x`, `y`, `zl`), search direction (`d`), scratch RHS (`p`, `_w1`, `rhs`,
`correction_lb`, `f`), index-range subviews for the lower-bound slice, and
the summary scalars (`mu`, `alpha_*`, infeasibilities, regularization
magnitudes, best complementarity, overall status).
"""
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

    correction_lb::VT
    rhs::VT

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
    status::MadNLP.Status
end

"""
    MPCSolver(nlp::LinearModel | ::QuadraticModel; kwargs...)

Scalar Mehrotra predictor-corrector IPM solver. Standardizes the input NLP
via `standard_form`, allocates the iterate state, and returns a ready-to-run
solver. Call `MadIPM.solve!(solver)` to run the IPM loop.

Kwargs are forwarded to [`IPMOptions`](@ref) and [`load_options`](@ref).
"""
mutable struct MPCSolver{T, P <: MPCProblem{T}, S <: MPCState{T}} <: MadNLP.AbstractMadNLPSolver{T}
    problem::P
    state::S
end

# ---------- unified IPM kernel accessors (scalar) ----------
# The kernels in `src/kernels/` and the solver loop call these `_foo(solver)`
# accessors so the exact same code drives both `MPCSolver` and
# `UniformBatchMPCSolver`. The batch counterparts live in
# `src/batch/structure.jl` with identical names — keep the two halves in
# sync when adding new accessors.

@inline _opt(s::MPCSolver)            = s.problem.opt
@inline _logger(s::MPCSolver)         = s.problem.logger
@inline _kkt(s::MPCSolver)            = s.problem.kkt
@inline _step_rule(s::MPCSolver)      = s.problem.step_rule
@inline _regularization(s::MPCSolver) = s.problem.regularization
@inline _barrier_update(s::MPCSolver) = s.problem.barrier_update

@inline _x(s::MPCSolver)    = s.state.x
@inline _zl(s::MPCSolver)   = s.state.zl
@inline _f(s::MPCSolver)    = s.state.f
@inline _y(s::MPCSolver)    = s.state.y
@inline _c(s::MPCSolver)    = s.state.c
@inline _jacl(s::MPCSolver) = s.state.jacl
@inline _p(s::MPCSolver)    = s.state.p
@inline _d(s::MPCSolver)    = s.state.d

# Lower-bound slice: restricted-range views into the iterate/step.
@inline _x_lr(s::MPCSolver)  = s.state.x_lr
@inline _xl_r(s::MPCSolver{T}) where {T} = zero(T)   # std form: lvar = 0 is a scalar
@inline _zl_r(s::MPCSolver)  = s.state.zl_r
@inline _dx_lr(s::MPCSolver) = s.state.dx_lr
@inline _dz_lb(s::MPCSolver) = MadNLP.dual_lb(s.state.d)

# Upper-bound slice: std form has no u-side multipliers/variables on the
# scalar path. Return empty views so u-side broadcasts compile down to no-ops.
@inline _x_ur(s::MPCSolver)  = view(MadNLP.full(s.state.x),  Int[])
@inline _xu_r(s::MPCSolver)  = view(MadNLP.full(s.state.x),  Int[])
@inline _zu_r(s::MPCSolver)  = view(MadNLP.full(s.state.zl), Int[])
@inline _dz_ub(s::MPCSolver) = view(MadNLP.full(s.state.d),  Int[])

@inline _correction_lb(s::MPCSolver) = s.state.correction_lb

@inline _alpha_p(s::MPCSolver) = s.state.alpha_p
@inline _alpha_d(s::MPCSolver) = s.state.alpha_d
@inline _del_w(s::MPCSolver)   = s.state.del_w
@inline _del_c(s::MPCSolver)   = s.state.del_c
@inline _mu(s::MPCSolver)      = s.state.mu

function MPCSolver(nlp::Union{LinearModel, QuadraticModel}; kwargs...)
    std_nlp, workspace = standard_form(nlp)
    options            = load_options(std_nlp; kwargs...)
    ipm_opt            = options.interior_point
    cnt                = MadNLP.MadNLPCounters(start_time = time())

    # Ruiz equilibration (optional; `NoScaling` short-circuits). Done BEFORE
    # the callback/KKT are built from `std_nlp`, so MadNLP sees the scaled
    # data from the first factorization onward.
    scaler = make_scaler(options.scaling, std_nlp)
    refresh_scaling!(scaler, std_nlp)

    cb = MadNLP.create_callback(
        MadNLP.SparseCallback, std_nlp;
        fixed_variable_treatment = ipm_opt.fixed_variable_treatment,
        equality_treatment       = ipm_opt.equality_treatment,
    )
    kkt = MadNLP.create_kkt_system(
        ipm_opt.kkt_system, cb, ipm_opt.linear_solver;
        opt_linear_solver = options.linear_solver,
    )

    VT        = typeof(NLPModels.get_x0(std_nlp))
    T         = eltype(VT)
    ind_lb    = cb.ind_lb
    empty_ind = similar(ind_lb, 0)
    nx        = MadNLP.n_variables(cb)
    ns        = 0           # std form has no inequality slacks
    n         = nx + ns
    m         = NLPModels.get_ncon(std_nlp)
    nlb       = length(ind_lb)

    _pv()  = MadNLP.PrimalVector(VT, nx, ns, ind_lb, empty_ind)
    _ukv() = MadNLP.UnreducedKKTVector(VT, n, m, nlb, 0, ind_lb, empty_ind)

    x, zl, f         = _pv(), _pv(), _pv()
    d, p, _w1        = _ukv(), _ukv(), _ukv()
    y, c, rhs, jacl  = VT(undef, m), VT(undef, m), VT(undef, m), VT(undef, n)
    correction_lb    = VT(undef, nlb)

    x_lr, zl_r, dx_lr = view(full(x), ind_lb), view(full(zl), ind_lb), view(d.xp, ind_lb)

    cnt.init_time = time() - cnt.start_time
    problem = MPCProblem(
        nlp, std_nlp, workspace, cb, kkt, ipm_opt,
        options.regularization, options.step_rule, options.barrier_update,
        scaler, options.logger, nlb,
    )
    z = zero(T)
    state = MPCState(
        cnt,
        x, y, zl,
        z,                                  # obj_val
        f, c, jacl,
        d, p, _w1, correction_lb, rhs,
        x_lr, zl_r, dx_lr,
        z, z, z, z, z,                      # inf_pr, inf_du, inf_compl, norm_b, norm_c
        z,                                  # mu
        z, z, z, z,                         # alpha_p, alpha_d, del_w, del_c
        typemax(T),                         # best_complementarity
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
    unapply_scaling!(problem.scaler, problem.nlp)
    try
        update_standard_form!(problem.original_nlp, problem.nlp, problem.workspace; kwargs...)
    catch
        refresh_scaling!(problem.scaler, problem.nlp)
        rethrow()
    end
    # `refresh_scaling!` re-runs Ruiz only when A / Q structurally changed
    # (signature mismatch); otherwise it reapplies the cached scales to the
    # freshly-updated c / b / bounds / x0.
    refresh_scaling!(problem.scaler, problem.nlp; force = true)
    return solver
end
