include("utils.jl")
include("rhs.jl")
include("callback.jl")
include("kkt.jl")

abstract type AbstractBatchMPCSolver{T} end

mutable struct SparseSameStructureBatchMPCSolver{
    T,
    KKT <: MadNLP.SparseKKTSystem,
    MT <: AbstractMatrix{T},
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    Model <: NLPModels.AbstractNLPModel{T,VT},
    CB <: MadNLP.AbstractCallback{T},
} <: AbstractBatchMPCSolver{T}
    nlps::Vector{Model}
    batch_size::Int
    solvers::Vector{MPCSolver{T, VT, VI, KKT, Model, CB}}

    d::BatchUnreducedKKTVector{T, MT, VT}

    opt::IPMOptions
    cnt::MadNLP.MadNLPCounters  # FIXME
    logger::MadNLP.MadNLPLogger
    kkts::AbstractBatchKKTSystem{T}
end

Base.length(batch_solver::SparseSameStructureBatchMPCSolver) = length(batch_solver.solvers)
Base.iterate(batch_solver::SparseSameStructureBatchMPCSolver, i=1) = iterate(batch_solver.solvers, i)
Base.getindex(batch_solver::SparseSameStructureBatchMPCSolver, i::Int) = batch_solver.solvers[i]

function SparseSameStructureBatchMPCSolver(nlps::Vector{Model}; kwargs...) where {T, VT0 <: AbstractVector{T}, Model <: NLPModels.AbstractNLPModel{T, VT0}}
    batch_size = length(nlps)
    batch_size == 0 && error("BatchMPCSolver requires at least one model")

    nlp1 = nlps[1]
    nvar = NLPModels.get_nvar(nlp1)
    ncon = NLPModels.get_ncon(nlp1)
    for (i, nlp) in enumerate(nlps)
        @assert NLPModels.get_nvar(nlp) == nvar "All models must have same number of variables (model $i differs)"
        @assert NLPModels.get_ncon(nlp) == ncon "All models must have same number of constraints (model $i differs)"
    end

    x0 = NLPModels.get_x0(nlp1)
    VT = typeof(x0)
    MT = typeof(similar(x0, T, 0, 0))
    VI = typeof(similar(x0, Int, 0))
    
    # shared
    options = load_options(nlp1; kwargs...)
    ipm_opt = options.interior_point
    ind_cons = MadNLP.get_index_constraints(nlp1;
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment
    )

    for nlp in nlps
        _ind_cons = MadNLP.get_index_constraints(nlp;
            fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
            equality_treatment=ipm_opt.equality_treatment
        )
        @assert _ind_cons == ind_cons "All models must have same ind_cons"
    end

    # TODO: check same structure
    
    ind_lb = ind_cons.ind_lb
    ind_ub = ind_cons.ind_ub
    ns = length(ind_cons.ind_ineq)
    nx = nvar
    n = nx + ns
    m = ncon
    nlb = length(ind_lb)
    nub = length(ind_ub)

    d_batch = init_batchunreduced_kktvector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)

    bcnt = MadNLP.MadNLPCounters(start_time=time())
    batch_cb = init_samestructure_sparsecallback(MT, VT, VI, nlps;
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment,
    )
    batch_kkt = init_samestructure_kktsystem(
        batch_cb, ind_cons, ipm_opt.linear_solver;
        opt_linear_solver=options.linear_solver,
    )

    solvers = Vector{MPCSolver{T, VT, VI, typeof(batch_kkt.kkts[1]), Model, typeof(batch_cb.callbacks[1])}}(undef, batch_size)
    for i in 1:batch_size
        solvers[i] = MPCSolver(nlps[i];
            d=_unreduced_kkt_vector_view(d_batch, i, ind_lb, ind_ub),
            cb=batch_cb.callbacks[i], kkt=batch_kkt.kkts[i], kwargs...
        )
    end

    batch = SparseSameStructureBatchMPCSolver(
        nlps, batch_size, solvers, d_batch,
        ipm_opt, bcnt, options.logger, batch_kkt,
    )

    bcnt.init_time = time() - bcnt.start_time

    return batch
end
