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

function UniformBatchWorkspace(::Type{MT}, ::Type{VT}, n::Int, m::Int, nlb::Int, nub::Int, batch_size::Int;
                        nvar_nlp::Int=0) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    _proto = MT(undef, 1, batch_size)
    MI = typeof(similar(_proto, Int32))
    MI64 = typeof(similar(_proto, Int64))
    return UniformBatchWorkspace{T, VT, MT, MI, MI64}(
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_xl, alpha_xu
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_zl, alpha_zu
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_p, alpha_d
        MT(undef, 1, batch_size),  # tau
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # mu_batch, mu_curr
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # mu_affine, sum_lb
        MT(undef, 1, batch_size),  # sum_ub
        MT(undef, 1, batch_size),  # obj_val
        MT(undef, 1, batch_size),  # norm_b
        MT(undef, 1, batch_size),  # norm_c
        MT(undef, 1, batch_size),  # inf_pr
        MT(undef, 1, batch_size),  # inf_du
        MT(undef, 1, batch_size),  # inf_compl
        MT(undef, 1, batch_size),  # best_complementarity
        MT(undef, 1, batch_size),  # dual_obj
        fill(MadNLP.INITIAL, batch_size),  # status
        similar(_proto, Int64, 1, batch_size),    # _term_gpu
        zeros(Int64, 1, batch_size),  # _term_cpu
        fill!(similar(_proto, Int64, 1, 1), Int64(MadNLP.REGULAR)),  # _any_nonregular_gpu
        fill(Int64(MadNLP.REGULAR), 1, 1),  # _any_nonregular_cpu
        MT(undef, 1, batch_size),  # _norm_gpu_w
        MT(undef, 1, batch_size),  # _norm_gpu_p
        fill!(similar(_proto, Int32), zero(Int32)),  # _ls_error
        fill!(MT(undef, 1, batch_size), one(T)),  # active_mask
        ones(T, 1, batch_size),                    # active_mask_cpu
        MT(undef, nvar_nlp, batch_size),   # bx
        VT(undef, batch_size),  # bf
        MT(undef, nvar_nlp, batch_size),   # bg
        MT(undef, m, batch_size),          # bv
    )
end

mutable struct BatchMPCProblem{BM, BCB, BVS, KKT<:AbstractBatchKKTSystem, REG<:AbstractRegularization, STEP<:AbstractStepRule, BARR<:AbstractBarrierUpdate}
    nlp::BM
    bcb::BCB
    kkt::KKT
    opt::IPMOptions
    regularization::REG
    step_rule::STEP
    barrier_update::BARR
    logger::MadNLP.MadNLPLogger
    batch_views::BVS
    batch_size::Int
end

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

mutable struct UniformBatchMPCSolver{T, MT, VT, P<:BatchMPCProblem, S<:BatchMPCState{T, MT, VT}} <: AbstractBatchMPCSolver{T, MT, VT}
    problem::P
    state::S
end

_get_ind_lb(bs::AbstractBatchMPCSolver) = bs.problem.bcb.ind_lb
_get_ind_ub(bs::AbstractBatchMPCSolver) = bs.problem.bcb.ind_ub

# ---------- accessors for unified IPM kernels (batch half) ----------
# Mirror the scalar accessors in src/structure.jl. Each batch accessor
# returns a `(dim, bs)` matrix view (or `(1, bs)` for per-instance scalars
# such as `α_p`, `δ_w`, `μ`) so the same broadcasted expressions in the
# unified kernels work identically on scalar (`Vector`/`T`) and batch
# (`Matrix`/`Matrix(1,bs)`) storage.

@inline _opt(s::AbstractBatchMPCSolver)            = s.problem.opt
@inline _logger(s::AbstractBatchMPCSolver)         = s.problem.logger
@inline _kkt(s::AbstractBatchMPCSolver)            = s.problem.kkt
@inline _step_rule(s::AbstractBatchMPCSolver)      = s.problem.step_rule
@inline _regularization(s::AbstractBatchMPCSolver) = s.problem.regularization
@inline _barrier_update(s::AbstractBatchMPCSolver) = s.problem.barrier_update

@inline _x(s::AbstractBatchMPCSolver)      = s.state.x
@inline _zl(s::AbstractBatchMPCSolver)     = s.state.zl
@inline _f(s::AbstractBatchMPCSolver)      = s.state.f
@inline _y(s::AbstractBatchMPCSolver)      = MadNLP.full(s.state.y)
@inline _c(s::AbstractBatchMPCSolver)      = MadNLP.full(s.state.c)
@inline _jacl(s::AbstractBatchMPCSolver)   = MadNLP.full(s.state.jacl)
@inline _p(s::AbstractBatchMPCSolver)      = s.state.p
@inline _d(s::AbstractBatchMPCSolver)      = s.state.d

@inline _x_lr(s::AbstractBatchMPCSolver)   = lower(s.state.x)
@inline _xl_r(s::AbstractBatchMPCSolver)   = lower(s.state.xl)
@inline _zl_r(s::AbstractBatchMPCSolver)   = lower(s.state.zl)
@inline _dx_lr(s::AbstractBatchMPCSolver)  = xp_lr(s.state.d)
@inline _dz_lb(s::AbstractBatchMPCSolver)  = MadNLP.dual_lb(s.state.d)
@inline _x_ur(s::AbstractBatchMPCSolver)   = upper(s.state.x)
@inline _xu_r(s::AbstractBatchMPCSolver)   = upper(s.state.xu)
@inline _zu_r(s::AbstractBatchMPCSolver)   = upper(s.state.zu)
@inline _dz_ub(s::AbstractBatchMPCSolver)  = MadNLP.dual_ub(s.state.d)

@inline _correction_lb(s::AbstractBatchMPCSolver) = MadNLP.full(s.state.correction_lb)

@inline _alpha_p(s::AbstractBatchMPCSolver) = s.state.workspace.alpha_p
@inline _alpha_d(s::AbstractBatchMPCSolver) = s.state.workspace.alpha_d
@inline _del_w(s::AbstractBatchMPCSolver)   = s.state.del_w
@inline _del_c(s::AbstractBatchMPCSolver)   = s.state.del_c
@inline _mu(s::AbstractBatchMPCSolver)      = s.state.workspace.mu_batch

# Union over both solver flavours used by the unified IPM kernels.
const AnyMPCSolver{T} = Union{MPCSolver{T}, AbstractBatchMPCSolver{T}}
active_batch_size(bs::AbstractBatchMPCSolver) = local_batch_size(active_view(bs.problem.batch_views))

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

update_active_set!(bs::AbstractBatchMPCSolver) = update_active_set!(bs.problem.batch_views, bs.state.workspace.status)


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
    kwargs...,
) where {T, MT}
    bmeta = bnlp.meta
    batch_size = bmeta.nbatch
    @assert batch_size > 0 "Need at least one instance in batch"

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

    cnt = BatchCounters(batch_size)
    bcb = MadNLP.create_callback(
        UniformBatchCallback{T,VT,MT,VI},
        bnlp;
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment,
        check_batch_structure=check_batch_structure,
    )

    ind_lb = bcb.ind_lb
    ind_ub = bcb.ind_ub

    ns = length(bcb.ind_ineq)
    nx = bcb.nvar
    n = nx + ns
    m = bcb.ncon
    nlb = length(ind_lb)
    nub = length(ind_ub)

    batch_views = BatchViewState(bcb, batch_size)

    batch_kkts = MadNLP.create_kkt_system(
        ipm_opt.kkt_system,
        bcb,
        uniformbatch_linear_solver;
        opt_linear_solver = opt_batch_ls,
        batch_views = batch_views,
    )

    batch_x  = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_xl = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_xu = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_zl = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_zu = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_f  = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)

    batch_d  = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    batch_p  = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    batch_w1 = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)

    batch_correction_lb = BatchVector(MT, VT, nlb, batch_size)
    batch_jacl          = BatchVector(MT, VT, n, batch_size)
    batch_y             = BatchVector(MT, VT, m, batch_size)
    batch_c             = BatchVector(MT, VT, m, batch_size)
    batch_rhs           = BatchVector(MT, VT, m, batch_size)

    workspace = UniformBatchWorkspace(MT, VT, n, m, nlb, nub, batch_size;
                               nvar_nlp=nvar_nlp)

    batch_del_w = fill!(MT(undef, 1, batch_size), zero(T))
    batch_del_c = fill!(MT(undef, 1, batch_size), zero(T))

    problem = BatchMPCProblem(
        bnlp, bcb, batch_kkts,
        ipm_opt, regularization, step_rule, barrier_update,
        logger, batch_views, batch_size,
    )
    state = BatchMPCState(
        cnt,
        batch_x, batch_xl, batch_xu, batch_zl, batch_zu, batch_f,
        batch_y, batch_c, batch_jacl, batch_rhs,
        batch_correction_lb,
        batch_d, batch_p, batch_w1,
        workspace,
        batch_del_w, batch_del_c,
    )
    return UniformBatchMPCSolver{T, MT, VT, typeof(problem), typeof(state)}(problem, state)
end
