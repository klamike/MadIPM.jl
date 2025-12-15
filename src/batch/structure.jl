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

    d::BatchUnreducedKKTVector{T, MT, VT, VI}

    opt::IPMOptions
    cnt::MadNLP.MadNLPCounters  # FIXME
    logger::MadNLP.MadNLPLogger
    kkts::AbstractBatchKKTSystem{T}
end

Base.length(batch_solver::SparseSameStructureBatchMPCSolver) = length(batch_solver.solvers)
Base.iterate(batch_solver::SparseSameStructureBatchMPCSolver, i=1) = iterate(batch_solver.solvers)
Base.getindex(batch_solver::SparseSameStructureBatchMPCSolver, i::Int) = batch_solver.solvers[i]

function SparseSameStructureBatchMPCSolver(nlps::Vector{Model}; kwargs...) where {T, VT0 <: AbstractVector{T}, Model <: NLPModels.AbstractNLPModel{T, VT0}}
    batch_size = length(nlps)
    batch_size == 0 && error("BatchMPCSolver requires at least one model")

    nvar = NLPModels.get_nvar(nlps[1])
    ncon = NLPModels.get_ncon(nlps[1])
    for (i, nlp) in enumerate(nlps)
        @assert NLPModels.get_nvar(nlp) == nvar "All models must have same number of variables (model $i differs)"
        @assert NLPModels.get_ncon(nlp) == ncon "All models must have same number of constraints (model $i differs)"
    end

    x0 = NLPModels.get_x0(nlps[1])
    VT = typeof(x0)
    MT = typeof(similar(x0, T, 0, 0))
    VI = typeof(similar(x0, Int, 0))
    
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

    # TODO: check same structure
    
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

    correction_lb_batch = MT(undef, nlb, batch_size)
    correction_ub_batch = MT(undef, nub, batch_size)
    jacl_batch = MT(undef, n, batch_size)
    y_batch = MT(undef, m, batch_size)
    c_batch = MT(undef, m, batch_size)
    rhs_batch = MT(undef, m, batch_size)
    
    bcnt = MadNLP.MadNLPCounters(start_time=time())
    batch_cb = SparseBatchCallback(MT, VT, VI, nlps;
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment,
        shared=(:jac_I, :jac_J, :hess_I, :hess_J),
    )
    batch_kkt = SparseSameStructureBatchKKTSystem(
        batch_cb,
        ind_cons,
        ipm_opt.linear_solver;
        opt_linear_solver=options.linear_solver,
    )

    solvers = Vector{MPCSolver{T, VT, VI, typeof(batch_kkt.kkts[1]), Model, typeof(batch_cb.callbacks[1])}}(undef, batch_size)
    for i in 1:batch_size
        cnt = MadNLP.MadNLPCounters(start_time=time())

        x_i  = _primal_vector_view(x_batch,  i, ind_lb, ind_ub)
        zl_i = _primal_vector_view(zl_batch, i, ind_lb, ind_ub)
        zu_i = _primal_vector_view(zu_batch, i, ind_lb, ind_ub)
        xl_i = _primal_vector_view(xl_batch, i, ind_lb, ind_ub)
        xu_i = _primal_vector_view(xu_batch, i, ind_lb, ind_ub)
        f_i  = _primal_vector_view(f_batch,  i, ind_lb, ind_ub)

        d_i   = _unreduced_kkt_vector_view(d_batch,   i, ind_lb, ind_ub)
        p_i   = _unreduced_kkt_vector_view(p_batch,   i, ind_lb, ind_ub)
        _w1_i = _unreduced_kkt_vector_view(_w1_batch, i, ind_lb, ind_ub)
        _w2_i = _unreduced_kkt_vector_view(_w2_batch, i, ind_lb, ind_ub)

        correction_lb_i = _matrix_column_view(correction_lb_batch, nlb, i, VT)
        correction_ub_i = _matrix_column_view(correction_ub_batch, nub, i, VT)
        jacl_i          = _matrix_column_view(jacl_batch,          n,   i, VT)
        y_i             = _matrix_column_view(y_batch,             m,   i, VT)
        c_i             = _matrix_column_view(c_batch,             m,   i, VT)
        rhs_i           = _matrix_column_view(rhs_batch,           m,   i, VT)

        x_lr_i  = view(full(x_i),  ind_lb)
        x_ur_i  = view(full(x_i),  ind_ub)
        xl_r_i  = view(full(xl_i), ind_lb)
        xu_r_i  = view(full(xu_i), ind_ub)
        zl_r_i  = view(full(zl_i), ind_lb)
        zu_r_i  = view(full(zu_i), ind_ub)
        dx_lr_i = view(d_i.xp, ind_lb)
        dx_ur_i = view(d_i.xp, ind_ub)

        cnt.init_time = time() - cnt.start_time

        solvers[i] = MPCSolver(
            nlps[i], class, batch_cb.callbacks[i], batch_kkt.kkts[i],
            ipm_opt, cnt, options.logger,
            n, m, nlb, nub,
            x_i, y_i, zl_i, zu_i, xl_i, xu_i,
            zero(T), f_i, c_i,
            jacl_i,
            d_i, p_i, _w1_i, _w2_i,
            correction_lb_i, correction_ub_i, rhs_i,
            ind_cons.ind_ineq, ind_cons.ind_fixed, ind_cons.ind_llb, ind_cons.ind_uub, ind_lb, ind_ub,
            x_lr_i, x_ur_i, xl_r_i, xu_r_i, zl_r_i, zu_r_i, dx_lr_i, dx_ur_i,
            zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), typemax(T), zero(T),
            MadNLP.INITIAL,
        )
    end

    batch = SparseSameStructureBatchMPCSolver(
        nlps, batch_size, solvers,
        d_batch,
        ipm_opt, bcnt, options.logger,
        batch_kkt,
    )

    bcnt.init_time = time() - bcnt.start_time

    return batch
end
