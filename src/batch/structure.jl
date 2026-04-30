"""
    UniformBatchWorkspace{T, VT, MT, MI, MI64}

Scratch buffers shared across the batch IPM iteration: step-length / μ
accumulators (all `(1, nbatch)` matrices so broadcasts stay on-backend),
per-instance termination status, active-set mask, and full-size workspace
vectors `bx`/`bf`/`bg`/`bv`. `_term_*` and `_any_nonregular_*` mirror GPU
reductions into CPU counters without blocking the solver loop.
"""
struct UniformBatchWorkspace{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}, MI<:AbstractMatrix{Int32}, MI64<:AbstractMatrix{Int64}}
    alpha_xl::MT
    alpha_xu::MT
    alpha_zl::MT
    alpha_zu::MT
    alpha_p::MT
    alpha_d::MT
    tau::MT

    mu_batch::MT
    mu_curr::MT
    mu_affine::MT
    sum_lb::MT
    sum_ub::MT

    obj_val::MT
    norm_b::MT
    norm_c::MT
    inf_pr::MT
    inf_du::MT
    inf_compl::MT
    best_complementarity::MT
    dual_obj::MT
    status::Vector{MadNLP.Status}

    _term_gpu::MI64
    _term_cpu::Matrix{Int64}
    _any_nonregular_gpu::MI64
    _any_nonregular_cpu::Matrix{Int64}
    _norm_gpu_w::MT
    _norm_gpu_p::MT
    _ls_error::MI

    active_mask::MT
    active_mask_cpu::Matrix{T}

    bx::MT
    bf::VT
    bg::MT
    bv::MT
end

function UniformBatchWorkspace(
    ::Type{MT}, ::Type{VT},
    n::Int, m::Int, nlb::Int, nub::Int, batch_size::Int;
    nvar_nlp::Int = 0,
) where {T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}}
    proto = MT(undef, 1, batch_size)
    MI    = typeof(similar(proto, Int32))
    MI64  = typeof(similar(proto, Int64))
    row() = MT(undef, 1, batch_size)  # (1, bs) scratch, matches proto's backend

    return UniformBatchWorkspace{T, VT, MT, MI, MI64}(
        row(), row(), row(), row(),                                 # alpha_xl..alpha_zu
        row(), row(), row(),                                        # alpha_p, alpha_d, tau
        row(), row(), row(), row(), row(),                          # mu_batch, mu_curr, mu_affine, sum_lb, sum_ub
        row(), row(), row(), row(), row(), row(), row(), row(),     # obj..dual_obj
        fill(MadNLP.INITIAL, batch_size),                           # status
        similar(proto, Int64, 1, batch_size),                       # _term_gpu
        zeros(Int64, 1, batch_size),                                # _term_cpu
        fill!(similar(proto, Int64, 1, 1), Int64(MadNLP.REGULAR)),  # _any_nonregular_gpu
        fill(Int64(MadNLP.REGULAR), 1, 1),                          # _any_nonregular_cpu
        row(), row(),                                               # _norm_gpu_w/_p
        fill!(similar(proto, Int32), zero(Int32)),                  # _ls_error
        fill!(row(), one(T)),                                       # active_mask
        ones(T, 1, batch_size),                                     # active_mask_cpu
        MT(undef, nvar_nlp, batch_size),                            # bx
        VT(undef, batch_size),                                      # bf
        MT(undef, nvar_nlp, batch_size),                            # bg
        MT(undef, m, batch_size),                                   # bv
    )
end

"""
    BatchMPCProblem{...}

Static problem data for a batched solve. Mirrors [`MPCProblem`](@ref) but
adds `bcb` (batch callback), `batch_views` (active-set bookkeeping), and
`batch_size`. `original_nlp`/`workspace` are `nothing` when the solver is
constructed directly from a standard-form batch NLP; otherwise they hold the
original batch model and the BQM presolve workspace used to recover
solutions in the original variable space.
"""
mutable struct BatchMPCProblem{BM, BCB, BVS, KKT<:AbstractBatchKKTSystem, REG<:AbstractRegularization, STEP<:AbstractStepRule, BARR<:AbstractBarrierUpdate, SCL<:AbstractScaler}
    original_nlp::Any           # original-space batch model (nothing when input is already std-form)
    nlp::BM                     # std-form batch model actually solved
    workspace::Any              # StandardFormBatchWorkspace, or nothing
    bcb::BCB
    kkt::KKT
    opt::IPMOptions
    regularization::REG
    step_rule::STEP
    barrier_update::BARR
    scaler::SCL
    logger::MadNLP.MadNLPLogger
    batch_views::BVS
    batch_size::Int
end

"""
    BatchMPCState{T, MT, VT}

Mutable batched iterate state: per-instance primal/dual iterates as
`(dim, nbatch)` matrices, search direction, scratch vectors, and the
`UniformBatchWorkspace` with the aggregate scalars.
"""
mutable struct BatchMPCState{T, MT, VT}
    cnt::BatchCounters

    x::BatchPrimalVector{T, MT}
    xl::BatchPrimalVector{T, MT}
    xu::BatchPrimalVector{T, MT}
    zl::BatchPrimalVector{T, MT}
    zu::BatchPrimalVector{T, MT}
    f::BatchPrimalVector{T, MT}

    y::BatchVector{T, MT}
    c::BatchVector{T, MT}
    jacl::BatchVector{T, MT}
    rhs::BatchVector{T, MT}
    correction_lb::BatchVector{T, MT}

    d::BatchUnreducedKKTVector{T, MT}
    p::BatchUnreducedKKTVector{T, MT}
    _w1::BatchUnreducedKKTVector{T, MT}

    workspace::UniformBatchWorkspace{T, VT, MT}

    del_w::MT
    del_c::MT
end

"""
    UniformBatchMPCSolver{...}

Batched Mehrotra predictor-corrector IPM solver. Input is a batched NLP
(already standard-form, or an `ObjRHSBatchQuadraticModel`/`BatchQuadraticModel`
that we standardize internally). Call `MadIPM.solve!(solver)` to run; each
batch instance is solved through the shared solver loop with an active-set
mask that prunes converged instances.
"""
mutable struct UniformBatchMPCSolver{T, MT, VT, P<:BatchMPCProblem, S<:BatchMPCState{T, MT, VT}}
    problem::P
    state::S
end

_get_ind_lb(bs::UniformBatchMPCSolver) = bs.problem.bcb.ind_lb
_get_ind_ub(bs::UniformBatchMPCSolver) = bs.problem.bcb.ind_ub

# Assert that `bnlp` is already in standard form: lvar == 0, uvar == +Inf,
# lcon == ucon (all equalities). Called by the inner `UniformBatchMPCSolver`
# ctor to catch users who pass a non-standardized model — they should go
# through the `ObjRHSBatchQuadraticModel`/`BatchQuadraticModel` ctor instead,
# which standardizes automatically.
function _assert_standard_form(bnlp::NLPModels.AbstractBatchNLPModel{T}) where {T}
    bm = bnlp.meta
    # `all(==(·), ...)` runs on the host or device depending on where the
    # arrays live — avoid materialising GPU matrices to CPU when there's
    # nothing to gain.
    all(==(zero(T)), bm.lvar) || throw(ArgumentError(
        "UniformBatchMPCSolver expects a standard-form batch (lvar = 0); construct from an original-space ObjRHSBatchQuadraticModel/BatchQuadraticModel to standardize."))
    all(==(T(Inf)), bm.uvar) || throw(ArgumentError(
        "UniformBatchMPCSolver expects a standard-form batch (uvar = +Inf); construct from an original-space batch model to standardize."))
    bm.lcon == bm.ucon || throw(ArgumentError(
        "UniformBatchMPCSolver expects a standard-form batch (lcon == ucon, all equalities); construct from an original-space batch model to standardize."))
    return nothing
end

function zero_inactive_step!(batch_solver::UniformBatchMPCSolver{T}) where T
    ws = batch_solver.state.workspace
    ws.alpha_p .*= ws.active_mask
    ws.alpha_d .*= ws.active_mask
end

# ---------- unified IPM kernel accessors (batch) ----------
# Batched counterparts of the `_foo(solver)` accessors in `src/structure.jl`.
# Keep names aligned so the kernels in `src/kernels/` and the solver loop
# dispatch identically on `MPCSolver` and `UniformBatchMPCSolver`.

@inline _opt(s::UniformBatchMPCSolver)            = s.problem.opt
@inline _logger(s::UniformBatchMPCSolver)         = s.problem.logger
@inline _kkt(s::UniformBatchMPCSolver)            = s.problem.kkt
@inline _step_rule(s::UniformBatchMPCSolver)      = s.problem.step_rule
@inline _regularization(s::UniformBatchMPCSolver) = s.problem.regularization
@inline _barrier_update(s::UniformBatchMPCSolver) = s.problem.barrier_update

@inline _x(s::UniformBatchMPCSolver)    = s.state.x
@inline _zl(s::UniformBatchMPCSolver)   = s.state.zl
@inline _f(s::UniformBatchMPCSolver)    = s.state.f
@inline _y(s::UniformBatchMPCSolver)    = MadNLP.full(s.state.y)
@inline _c(s::UniformBatchMPCSolver)    = MadNLP.full(s.state.c)
@inline _jacl(s::UniformBatchMPCSolver) = MadNLP.full(s.state.jacl)
@inline _p(s::UniformBatchMPCSolver)    = s.state.p
@inline _d(s::UniformBatchMPCSolver)    = s.state.d

# Lower-bound slice.
@inline _x_lr(s::UniformBatchMPCSolver)  = lower(s.state.x)
@inline _xl_r(s::UniformBatchMPCSolver)  = lower(s.state.xl)
@inline _zl_r(s::UniformBatchMPCSolver)  = lower(s.state.zl)
@inline _dx_lr(s::UniformBatchMPCSolver) = xp_lr(s.state.d)
@inline _dz_lb(s::UniformBatchMPCSolver) = MadNLP.dual_lb(s.state.d)
# Upper-bound slice — unlike the scalar path, the batch std form *does* have
# u-side multipliers (for var upper bounds mapped to equality rows).
@inline _x_ur(s::UniformBatchMPCSolver)  = upper(s.state.x)
@inline _xu_r(s::UniformBatchMPCSolver)  = upper(s.state.xu)
@inline _zu_r(s::UniformBatchMPCSolver)  = upper(s.state.zu)
@inline _dz_ub(s::UniformBatchMPCSolver) = MadNLP.dual_ub(s.state.d)

@inline _correction_lb(s::UniformBatchMPCSolver) = MadNLP.full(s.state.correction_lb)

@inline _alpha_p(s::UniformBatchMPCSolver) = s.state.workspace.alpha_p
@inline _alpha_d(s::UniformBatchMPCSolver) = s.state.workspace.alpha_d
@inline _del_w(s::UniformBatchMPCSolver)   = s.state.del_w
@inline _del_c(s::UniformBatchMPCSolver)   = s.state.del_c
@inline _mu(s::UniformBatchMPCSolver)      = s.state.workspace.mu_batch

const MaybeBatchMPCSolver{T} = Union{MPCSolver{T}, UniformBatchMPCSolver{T}}
active_batch_size(bs::UniformBatchMPCSolver) = local_batch_size(active_view(bs.problem.batch_views))

function update_active_set!(state::BatchViewState, status::Vector{MadNLP.Status})
    nselected = 0
    @inbounds for i in eachindex(status)
        if status[i] == MadNLP.REGULAR
            nselected += 1
            state.selected_local_buffer[nselected] = i
        end
    end
    if nselected == batch_size_root(root_view(state))
        return reset_active_view!(state)
    end
    reset_active_view!(state)
    return select_local!(state, state.selected_local_buffer, nselected; reset_slots=true)
end

update_active_set!(bs::UniformBatchMPCSolver) = update_active_set!(bs.problem.batch_views, bs.state.workspace.status)


"""
    UniformBatchMPCSolver(bnlp::AbstractBatchNLPModel; linear_solver, kwargs...)

Construct a batch solver from a `AbstractBatchNLPModel`.
"""
function UniformBatchMPCSolver(
    bnlp::NLPModels.AbstractBatchNLPModel{T, MT};
    VT = typeof(similar(NLPModels.get_x0(bnlp), T, 0)),
    VI = typeof(similar(NLPModels.get_x0(bnlp), Int, 0)),
    uniformbatch_linear_solver = LoopedBatchLinearSolver,
    check_batch_structure::Bool = true,
    check_standard_form::Bool = true,
    kwargs...,
) where {T, MT}
    bmeta = bnlp.meta
    batch_size = bmeta.nbatch
    @assert batch_size > 0 "Need at least one instance in batch"
    check_standard_form && _assert_standard_form(bnlp)

    nvar_nlp = bmeta.nvar

    opt_batch_ls = MadNLP.default_options(uniformbatch_linear_solver)
    remaining_kwargs = MadNLP.set_options!(opt_batch_ls, kwargs)

    options = load_options(bnlp; remaining_kwargs...)
    ipm_opt = options.interior_point
    logger = options.logger
    regularization = options.regularization
    step_rule = options.step_rule
    barrier_update = options.barrier_update

    # `linear_solver` is the IPMOptions kwarg the user actually wires; mirror
    # it onto the per-instance looped solver so e.g. `linear_solver=CUDSSSolver`
    # on GPU is respected (otherwise opt_batch_ls keeps its CPU default).
    opt_batch_ls.looped_linear_solver = ipm_opt.linear_solver

    # Ruiz equilibration of the std-form batch: std form is already built, so
    # we can scale `A`, `Q`, `c_batch`, `lcon/ucon`, `x0` in-place before
    # constructing the callback/KKT (which cache a scaled view of A).
    scaler = make_scaler(options.scaling, bnlp)
    refresh_scaling!(scaler, bnlp)

    cnt = BatchCounters(batch_size)
    bcb = MadNLP.create_callback(
        UniformBatchCallback{T, VT, MT, VI}, bnlp;
        fixed_variable_treatment = ipm_opt.fixed_variable_treatment,
        equality_treatment       = ipm_opt.equality_treatment,
        check_batch_structure    = check_batch_structure,
    )

    ind_lb, ind_ub = bcb.ind_lb, bcb.ind_ub
    nx, ns, m      = bcb.nvar, length(bcb.ind_ineq), bcb.ncon
    n              = nx + ns
    nlb, nub       = length(ind_lb), length(ind_ub)
    batch_views    = BatchViewState(bcb, batch_size)

    batch_kkts = MadNLP.create_kkt_system(
        ipm_opt.kkt_system, bcb, uniformbatch_linear_solver;
        opt_linear_solver = opt_batch_ls, batch_views = batch_views,
    )

    _pv()  = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    _ukv() = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)

    x, xl, xu, zl, zu, f = _pv(), _pv(), _pv(), _pv(), _pv(), _pv()
    d, p, w1             = _ukv(), _ukv(), _ukv()

    correction_lb = BatchVector(MT, VT, nlb, batch_size)
    jacl          = BatchVector(MT, VT, n,   batch_size)
    y             = BatchVector(MT, VT, m,   batch_size)
    c             = BatchVector(MT, VT, m,   batch_size)
    rhs           = BatchVector(MT, VT, m,   batch_size)

    workspace = UniformBatchWorkspace(MT, VT, n, m, nlb, nub, batch_size;
                                       nvar_nlp = nvar_nlp)
    del_w = fill!(MT(undef, 1, batch_size), zero(T))
    del_c = fill!(MT(undef, 1, batch_size), zero(T))

    problem = BatchMPCProblem(
        nothing, bnlp, nothing, bcb, batch_kkts, ipm_opt,
        regularization, step_rule, barrier_update, scaler,
        logger, batch_views, batch_size,
    )
    state = BatchMPCState(
        cnt, x, xl, xu, zl, zu, f,
        y, c, jacl, rhs, correction_lb,
        d, p, w1, workspace, del_w, del_c,
    )
    return UniformBatchMPCSolver{T, MT, VT, typeof(problem), typeof(state)}(problem, state)
end

"""
    UniformBatchMPCSolver(bnlp::ObjRHSBatchQuadraticModel; kwargs...)

Construct a batch solver from an original-space batch LP/QP. Standardizes
each instance via [`standard_form`](@ref), builds the std-form solver, and
keeps the original model + workspace on the problem so `solve!` recovers
solutions/multipliers in the original space.
"""
function UniformBatchMPCSolver(bnlp::BatchQuadraticModel; kwargs...)
    std_bnlp, ws_batch = standard_form(bnlp)
    # Delegate to the generic ctor; `invoke` avoids re-dispatching to this
    # method. `standard_form` guarantees the std batch already satisfies the
    # standard-form invariants, so skip the (GPU-unfriendly) re-check.
    solver = invoke(UniformBatchMPCSolver,
                    Tuple{NLPModels.AbstractBatchNLPModel{eltype(std_bnlp.c_batch),typeof(std_bnlp.c_batch)}},
                    std_bnlp; check_standard_form = false, kwargs...)
    solver.problem.original_nlp = bnlp
    solver.problem.workspace = ws_batch
    return solver
end

"""
    update!(solver::UniformBatchMPCSolver; c_batch, c0_batch, A, Q, lvar_batch, uvar_batch, lcon_batch, ucon_batch, x0_batch, y0_batch)

Mutate the original batch model held by `solver` and propagate to the
std-form batch model. Sparsity patterns and bound kinds must be unchanged;
construct a new solver for structural changes.
"""
function update!(solver::UniformBatchMPCSolver; kwargs...)
    problem = solver.problem
    problem.original_nlp === nothing && error(
        "update! requires a solver built from an original-space batch model " *
        "(e.g. ObjRHSBatchQuadraticModel); this solver was constructed from " *
        "an already-standardized batch NLP.")
    unapply_scaling!(problem.scaler, problem.nlp)
    try
        update_standard_form!(problem.original_nlp, problem.nlp, problem.workspace; kwargs...)
    catch
        refresh_scaling!(problem.scaler, problem.nlp)
        rethrow()
    end
    refresh_scaling!(problem.scaler, problem.nlp; force = true)
    return solver
end
