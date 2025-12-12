include("madnlp.jl")

mutable struct ViewBasedMPCSolver{
    # NOTE: this is only here because I don't want to mess with the non-batched version.
    # eventually we can consider merging them into a single type.
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: MadNLP.AbstractKKTSystem{T},
    Model <: NLPModels.AbstractNLPModel{T,VT},
    CB <: MadNLP.AbstractCallback{T},
} <: AbstractMPCSolver{T}
    nlp::Model
    class::AbstractConicProblem
    cb::CB
    kkt::KKTSystem

    opt::IPMOptions
    cnt::MadNLP.MadNLPCounters
    logger::MadNLP.MadNLPLogger

    n::Int
    m::Int
    nlb::Int
    nub::Int

    x::ViewBasedPrimalVector{T}
    y::AbstractVector{T}
    zl::ViewBasedPrimalVector{T}
    zu::ViewBasedPrimalVector{T}
    xl::ViewBasedPrimalVector{T}
    xu::ViewBasedPrimalVector{T}

    f::ViewBasedPrimalVector{T}
    c::AbstractVector{T}

    jacl::AbstractVector{T}

    d::ViewBasedUnreducedKKTVector{T}
    p::ViewBasedUnreducedKKTVector{T}

    _w1::ViewBasedUnreducedKKTVector{T}
    _w2::ViewBasedUnreducedKKTVector{T}

    correction_lb::AbstractVector{T}
    correction_ub::AbstractVector{T}
    rhs::AbstractVector{T}
    ind_ineq::VI
    ind_fixed::VI
    ind_lb::VI
    ind_ub::VI
    ind_llb::VI
    ind_uub::VI

    x_lr
    x_ur
    xl_r
    xu_r
    zl_r
    zu_r
    dx_lr
    dx_ur

    status::MadNLP.Status

    _batch_solver
    _batch_index::Int
end

############################################################################################
#                                                                                          #
#        very dumb {get/set}property override                                              #
#   this is for any `T` fields. probably you can use Ref{T} instead,                       #
#   but that would require modifying solver code.                                          #
#                                                                                          #
############################################################################################

const _batch_delegated_fields = (:obj_val, :inf_pr, :inf_du, :inf_compl, :norm_b, :norm_c, 
                                 :mu, :alpha_p, :alpha_d, :del_w, :del_c,
                                 :best_complementarity, :mu_curr)
for field in _batch_delegated_fields
    @eval function Base.getproperty(solver::ViewBasedMPCSolver, ::Val{$(Meta.quot(field))})
        batch_solver = getfield(solver, :_batch_solver)
        if batch_solver !== nothing
            i = getfield(solver, :_batch_index)
            return getfield(batch_solver, $(Meta.quot(field)))[i]
        end
        return getfield(solver, $(Meta.quot(field)))
    end
end
function Base.getproperty(solver::ViewBasedMPCSolver, name::Symbol)
    if name in _batch_delegated_fields
        return getproperty(solver, Val(name))
    end
    return getfield(solver, name)
end
for field in _batch_delegated_fields
    @eval function Base.setproperty!(solver::ViewBasedMPCSolver, ::Val{$(Meta.quot(field))}, val)
        batch_solver = getfield(solver, :_batch_solver)
        if batch_solver !== nothing
            i = getfield(solver, :_batch_index)
            getfield(batch_solver, $(Meta.quot(field)))[i] = val
            return val
        end
        return setfield!(solver, $(Meta.quot(field)), val)
    end
end
function Base.setproperty!(solver::ViewBasedMPCSolver, name::Symbol, val)
    if name in _batch_delegated_fields
        return setproperty!(solver, Val(name), val)
    end
    return setfield!(solver, name, val)
end

############################################################################################
#                                                                                          #
#   end   very dumb {get/set}property !!!                                                  #
#                                                                                          #
############################################################################################

mutable struct BatchMPCSolver{
    T,
    MT <: AbstractMatrix{T},
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: MadNLP.AbstractKKTSystem{T},
    Model <: NLPModels.AbstractNLPModel{T,VT},
    CB <: MadNLP.AbstractCallback{T},
} <: MadNLP.AbstractMadNLPSolver{T}
    nlps::Vector{Model}
    batch_size::Int
    solvers::Vector{ViewBasedMPCSolver{T, VT, VI, KKTSystem, Model, CB}}
    
    # Batch vectors
    x::BatchPrimalVector{T, MT, VT, VI}
    zl::BatchPrimalVector{T, MT, VT, VI}
    zu::BatchPrimalVector{T, MT, VT, VI}
    xl::BatchPrimalVector{T, MT, VT, VI}
    xu::BatchPrimalVector{T, MT, VT, VI}
    f::BatchPrimalVector{T, MT, VT, VI}
    
    d::BatchUnreducedKKTVector{T, MT, VT, VI}
    p::BatchUnreducedKKTVector{T, MT, VT, VI}
    _w1::BatchUnreducedKKTVector{T, MT, VT, VI}
    _w2::BatchUnreducedKKTVector{T, MT, VT, VI}
    
    # Batch regular vectors (matrices, each column is one instance)
    y::MT  # size (m, batch_size)
    c::MT  # size (m, batch_size)
    jacl::MT  # size (n, batch_size)
    correction_lb::MT  # size (nlb, batch_size)
    correction_ub::MT  # size (nub, batch_size)
    rhs::MT  # size (m, batch_size)
    
    # Batch scalar vectors (one value per batch instance)
    obj_val::VT
    dobj_val::VT
    inf_pr::VT
    inf_du::VT
    inf_compl::VT
    norm_b::VT
    norm_c::VT
    mu::VT
    alpha_p::VT
    alpha_d::VT
    del_w::VT
    del_c::VT
    best_complementarity::VT
    mu_affine::VT
    mu_curr::VT
    
    # Shared across batch
    opt::IPMOptions
    cnt::Vector{MadNLP.MadNLPCounters}
    logger::Vector{MadNLP.MadNLPLogger}
    kkt::Vector{KKTSystem}
    cb::Vector{CB}
    class::AbstractConicProblem
    
    # Dimensions
    n::Int
    m::Int
    nlb::Int
    nub::Int
    nx::Int
    ns::Int
    
    # Index sets (shared)
    ind_ineq::VI
    ind_fixed::VI
    ind_lb::VI
    ind_ub::VI
    ind_llb::VI
    ind_uub::VI
    
    x_lr::SubArray{T, 2, MT, <:Tuple}
    x_ur::SubArray{T, 2, MT, <:Tuple}
    xl_r::SubArray{T, 2, MT, <:Tuple}
    xu_r::SubArray{T, 2, MT, <:Tuple}
    zl_r::SubArray{T, 2, MT, <:Tuple}
    zu_r::SubArray{T, 2, MT, <:Tuple}
    dx_lr::SubArray{T, 2, MT, <:Tuple}
    dx_ur::SubArray{T, 2, MT, <:Tuple}
end

Base.length(batch::BatchMPCSolver) = batch.batch_size
Base.iterate(batch::BatchMPCSolver, i=1) = i > length(batch) ? nothing : (batch.solvers[i], i+1)
Base.getindex(batch::BatchMPCSolver, i::Int) = batch.solvers[i]

# FIXME vector type from matrix type?
_infer_vector_type(::Type{Matrix{T}}) where T = Vector{T}
_infer_vector_type(::Type{MT}) where {T, MT <: AbstractMatrix{T}} = Vector{T}  # Default fallback

function BatchMPCSolver(nlps::Vector{Model}; kwargs...) where {T, Model <: NLPModels.AbstractNLPModel{T}}
    batch_size = length(nlps)
    batch_size == 0 && error("BatchMPCSolver requires at least one model")
    
    # Validate all models have same dimensions
    nvar = NLPModels.get_nvar(nlps[1])
    ncon = NLPModels.get_ncon(nlps[1])
    for (i, nlp) in enumerate(nlps)
        @assert NLPModels.get_nvar(nlp) == nvar "All models must have same number of variables (model $i differs)"
        @assert NLPModels.get_ncon(nlp) == ncon "All models must have same number of constraints (model $i differs)"
    end
    
    # FIXME
    MT = Matrix{T}
    VT = _infer_vector_type(MT)
    VI = Vector{Int}
    
    # shared
    options = load_options(nlps[1]; kwargs...)
    ipm_opt = options.interior_point
    ind_cons = MadNLP.get_index_constraints(nlps[1];
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment
    )
    for (i, nlp) in enumerate(nlps)
        _ind_cons = MadNLP.get_index_constraints(nlp;
            fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
            equality_treatment=ipm_opt.equality_treatment
        )
        @assert _ind_cons == ind_cons "All models must have same ind_cons"
    end
    
    ind_lb = ind_cons.ind_lb
    ind_ub = ind_cons.ind_ub
    ns = length(ind_cons.ind_ineq)
    nx = nvar
    n = nx + ns
    m = ncon
    nlb = length(ind_lb)
    nub = length(ind_ub)

    nnzh = MadNLP.get_nnzh(nlps[1])
    class = iszero(nnzh) ? LinearProgram() : QuadraticProgram()
    
    # batched vectors
    x_batch = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    zl_batch = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    zu_batch = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    xl_batch = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    xu_batch = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    f_batch = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    
    d_batch = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    p_batch = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    _w1_batch = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    _w2_batch = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    
    # batched buffers
    y_batch = MT(undef, m, batch_size)
    c_batch = MT(undef, m, batch_size)
    jacl_batch = MT(undef, n, batch_size)
    correction_lb_batch = MT(undef, nlb, batch_size)
    correction_ub_batch = MT(undef, nub, batch_size)
    rhs_batch = MT(undef, m, batch_size)
    
    # batched scalar vectors
    obj_val_batch = VT(undef, batch_size)
    dobj_val_batch = VT(undef, batch_size)
    inf_pr_batch = VT(undef, batch_size)
    inf_du_batch = VT(undef, batch_size)
    inf_compl_batch = VT(undef, batch_size)
    norm_b_batch = VT(undef, batch_size)
    norm_c_batch = VT(undef, batch_size)
    mu_batch = VT(undef, batch_size)
    alpha_p_batch = VT(undef, batch_size)
    alpha_d_batch = VT(undef, batch_size)
    del_w_batch = VT(undef, batch_size)
    del_c_batch = VT(undef, batch_size)
    best_complementarity_batch = VT(undef, batch_size)
    mu_affine_batch = VT(undef, batch_size)
    mu_curr_batch = VT(undef, batch_size)
    
    cbs = [MadNLP.create_callback(MadNLP.SparseCallback, nlp; 
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment) for nlp in nlps]
    kkts = [MadNLP.create_kkt_system(  # FIXME: detect shared structure here?
        ipm_opt.kkt_system,
        cbs[i],
        ind_cons,
        ipm_opt.linear_solver;
        opt_linear_solver=options.linear_solver,
    ) for i in 1:batch_size]
    cnts = [MadNLP.MadNLPCounters(start_time=time()) for _ in 1:batch_size]
    loggers = [options.logger for _ in 1:batch_size]  # Could create separate loggers
    
    solvers = Vector{ViewBasedMPCSolver{T, VT, VI, typeof(kkts[1]), Model, typeof(cbs[1])}}(undef, batch_size)
    for i in 1:batch_size
        x_i = ViewBasedPrimalVector(_madnlp_unsafe_column_wrap(
            x_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        zl_i = ViewBasedPrimalVector(_madnlp_unsafe_column_wrap(
            zl_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        zu_i = ViewBasedPrimalVector(_madnlp_unsafe_column_wrap(
            zu_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        xl_i = ViewBasedPrimalVector(_madnlp_unsafe_column_wrap(
            xl_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        xu_i = ViewBasedPrimalVector(_madnlp_unsafe_column_wrap(
            xu_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        f_i = ViewBasedPrimalVector(_madnlp_unsafe_column_wrap(
            f_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        d_i = ViewBasedUnreducedKKTVector(_madnlp_unsafe_column_wrap(
            d_batch.values, n+m+nlb+nub, (i-1)*(n+m+nlb+nub)+1, VT), n, m, nlb, nub, ind_lb, ind_ub)
        p_i = ViewBasedUnreducedKKTVector(_madnlp_unsafe_column_wrap(
            p_batch.values, n+m+nlb+nub, (i-1)*(n+m+nlb+nub)+1, VT), n, m, nlb, nub, ind_lb, ind_ub)
        _w1_i = ViewBasedUnreducedKKTVector(_madnlp_unsafe_column_wrap(
            _w1_batch.values, n+m+nlb+nub, (i-1)*(n+m+nlb+nub)+1, VT), n, m, nlb, nub, ind_lb, ind_ub)
        _w2_i = ViewBasedUnreducedKKTVector(_madnlp_unsafe_column_wrap(
            _w2_batch.values, n+m+nlb+nub, (i-1)*(n+m+nlb+nub)+1, VT), n, m, nlb, nub, ind_lb, ind_ub)

        y_i = _madnlp_unsafe_column_wrap(y_batch, m, (i-1)*m + 1, VT)
        c_i = _madnlp_unsafe_column_wrap(c_batch, m, (i-1)*m + 1, VT)
        jacl_i = _madnlp_unsafe_column_wrap(jacl_batch, n, (i-1)*n + 1, VT)
        correction_lb_i = _madnlp_unsafe_column_wrap(correction_lb_batch, nlb, (i-1)*nlb + 1, VT)
        correction_ub_i = _madnlp_unsafe_column_wrap(correction_ub_batch, nub, (i-1)*nub + 1, VT)
        rhs_i = _madnlp_unsafe_column_wrap(rhs_batch, m, (i-1)*m + 1, VT)
        x_lr_i = view(full(x_i), ind_cons.ind_lb)
        x_ur_i = view(full(x_i), ind_cons.ind_ub)
        xl_r_i = view(full(xl_i), ind_cons.ind_lb)
        xu_r_i = view(full(xu_i), ind_cons.ind_ub)
        zl_r_i = view(full(zl_i), ind_cons.ind_lb)
        zu_r_i = view(full(zu_i), ind_cons.ind_ub)
        dx_lr_i = view(d_i.xp, ind_cons.ind_lb)
        dx_ur_i = view(d_i.xp, ind_cons.ind_ub)
        
        solvers[i] = ViewBasedMPCSolver(
            nlps[i], class, cbs[i], kkts[i],
            ipm_opt, cnts[i], loggers[i],
            n, m, nlb, nub,
            x_i, y_i, zl_i, zu_i, xl_i, xu_i,
            f_i, c_i,
            jacl_i,
            d_i, p_i,
            _w1_i, _w2_i,
            correction_lb_i, correction_ub_i,
            rhs_i,
            ind_cons.ind_ineq, ind_cons.ind_fixed, ind_cons.ind_llb, ind_cons.ind_uub,
            ind_cons.ind_lb, ind_cons.ind_ub,
            x_lr_i, x_ur_i, xl_r_i, xu_r_i, zl_r_i, zu_r_i, dx_lr_i, dx_ur_i,
            MadNLP.INITIAL,
            nothing, 0,  # Will set batch reference after construction
        )
    end
    
    x_lr_batch = view(x_batch.values, ind_lb, :)
    x_ur_batch = view(x_batch.values, ind_ub, :)
    xl_r_batch = view(xl_batch.values, ind_lb, :)
    xu_r_batch = view(xu_batch.values, ind_ub, :)
    zl_r_batch = view(zl_batch.values, ind_lb, :)
    zu_r_batch = view(zu_batch.values, ind_ub, :)
    dx_lr_batch = view(d_batch.values, ind_lb, :)
    dx_ur_batch = view(d_batch.values, ind_ub, :)
    
    batch = BatchMPCSolver(
        nlps, batch_size, solvers,
        x_batch, zl_batch, zu_batch, xl_batch, xu_batch, f_batch,
        d_batch, p_batch, _w1_batch, _w2_batch,
        y_batch, c_batch, jacl_batch, correction_lb_batch, correction_ub_batch, rhs_batch,
        obj_val_batch, dobj_val_batch, inf_pr_batch, inf_du_batch, inf_compl_batch, norm_b_batch, norm_c_batch,
        mu_batch, alpha_p_batch, alpha_d_batch, del_w_batch, del_c_batch,
        best_complementarity_batch, mu_affine_batch, mu_curr_batch,
        ipm_opt, cnts, loggers, kkts, cbs, class,
        n, m, nlb, nub, nx, ns,
        ind_cons.ind_ineq, ind_cons.ind_fixed, ind_cons.ind_lb, ind_cons.ind_ub,
        ind_cons.ind_llb, ind_cons.ind_uub,
        x_lr_batch, x_ur_batch, xl_r_batch, xu_r_batch, zl_r_batch, zu_r_batch, dx_lr_batch, dx_ur_batch,
    )
    for i in 1:batch_size
        setfield!(solvers[i], :_batch_solver, batch)
        setfield!(solvers[i], :_batch_index, i)
    end
    
    return batch
end
