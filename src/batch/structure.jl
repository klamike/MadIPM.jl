include("madnlp_rhs.jl")
include("madnlp_cb.jl")
include("madnlp_kkt.jl")

abstract type AbstractBatchMPCSolver{T} end

mutable struct SameStructureBatchMPCSolver{
    T, Ts,
    MT <: AbstractMatrix{T},
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: MadNLP.AbstractKKTSystem,
    BK <: AbstractBatchKKTSystem{T, KKTSystem},
    Model <: NLPModels.AbstractNLPModel{T,VT},
    CB <: MadNLP.AbstractCallback{T},
} <: AbstractBatchMPCSolver{T}
    nlps::Vector{Model}
    batch_size::Int
    solvers::Vector{MPCSolver{T, Ts, VT, VI, KKTSystem, Model, CB}}

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

    y::MT
    c::MT
    jacl::MT
    correction_lb::MT
    correction_ub::MT
    rhs::MT

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

    opt::IPMOptions
    cnt::MadNLP.MadNLPCounters  # FIXME
    logger::MadNLP.MadNLPLogger
    kkts::BK
    cbs::AbstractBatchCallback{CB}
    class::AbstractConicProblem

    n::Int
    m::Int
    nlb::Int
    nub::Int
    nx::Int
    ns::Int

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

Base.length(batch::SameStructureBatchMPCSolver) = batch.batch_size
Base.iterate(batch::SameStructureBatchMPCSolver, i=1) = i > length(batch) ? nothing : (batch.solvers[i], i+1)
Base.getindex(batch::SameStructureBatchMPCSolver, i::Int) = batch.solvers[i]

function SameStructureBatchMPCSolver(nlps::Vector{Model}; kwargs...) where {T, VT0 <: AbstractVector{T}, Model <: NLPModels.AbstractNLPModel{T, VT0}}
    batch_size = length(nlps)
    batch_size == 0 && error("BatchMPCSolver requires at least one model")
    
    # Validate all models have same dimensions
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
    
    bcnt = MadNLP.MadNLPCounters(start_time=time())
    batch_cb = SparseBatchCallback(MT, VT, VI, nlps;
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment,
        shared=(:jac_I, :jac_J, :hess_I, :hess_J),
    )
    batch_kkt = SparseSameStructureBatchKKTSystem(
        ipm_opt.kkt_system,
        batch_cb,
        ind_cons,
        ipm_opt.linear_solver;
        opt_linear_solver=options.linear_solver,
    )
    
    Ts = typeof(MadNLP._madnlp_unsafe_wrap(obj_val_batch, 1, 1))  # FIXME
    solvers = Vector{MPCSolver{T, Ts, VT, VI, typeof(batch_kkt.kkts[1]), Model, typeof(batch_cb.callbacks[1])}}(undef, batch_size)
    for i in 1:batch_size
        cnt = MadNLP.MadNLPCounters(start_time=time())
        x_i = MadNLP.PrimalVector(_madnlp_unsafe_column_wrap(x_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        zl_i = MadNLP.PrimalVector(_madnlp_unsafe_column_wrap(zl_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        zu_i = MadNLP.PrimalVector(_madnlp_unsafe_column_wrap(zu_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        xl_i = MadNLP.PrimalVector(_madnlp_unsafe_column_wrap(xl_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        xu_i = MadNLP.PrimalVector(_madnlp_unsafe_column_wrap(xu_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)
        f_i = MadNLP.PrimalVector(_madnlp_unsafe_column_wrap(f_batch.values, nx+ns, (i-1)*(nx+ns)+1, VT), nx, ns, ind_lb, ind_ub)

        d_i = MadNLP.UnreducedKKTVector(_madnlp_unsafe_column_wrap(d_batch.values, n+m+nlb+nub, (i-1)*(n+m+nlb+nub)+1, VT), n, m, nlb, nub, ind_lb, ind_ub)
        p_i = MadNLP.UnreducedKKTVector(_madnlp_unsafe_column_wrap(p_batch.values, n+m+nlb+nub, (i-1)*(n+m+nlb+nub)+1, VT), n, m, nlb, nub, ind_lb, ind_ub)
        _w1_i = MadNLP.UnreducedKKTVector(_madnlp_unsafe_column_wrap(_w1_batch.values, n+m+nlb+nub, (i-1)*(n+m+nlb+nub)+1, VT), n, m, nlb, nub, ind_lb, ind_ub)
        _w2_i = MadNLP.UnreducedKKTVector(_madnlp_unsafe_column_wrap(_w2_batch.values, n+m+nlb+nub, (i-1)*(n+m+nlb+nub)+1, VT), n, m, nlb, nub, ind_lb, ind_ub)

        correction_lb_i = _madnlp_unsafe_column_wrap(correction_lb_batch, nlb, (i-1)*nlb + 1, VT)
        correction_ub_i = _madnlp_unsafe_column_wrap(correction_ub_batch, nub, (i-1)*nub + 1, VT)
        y_i = _madnlp_unsafe_column_wrap(y_batch, m, (i-1)*m + 1, VT)
        c_i = _madnlp_unsafe_column_wrap(c_batch, m, (i-1)*m + 1, VT)
        jacl_i = _madnlp_unsafe_column_wrap(jacl_batch, n, (i-1)*n + 1, VT)
        rhs_i = _madnlp_unsafe_column_wrap(rhs_batch, m, (i-1)*m + 1, VT)

        x_lr_i = view(full(x_i), ind_cons.ind_lb)
        x_ur_i = view(full(x_i), ind_cons.ind_ub)
        xl_r_i = view(full(xl_i), ind_cons.ind_lb)
        xu_r_i = view(full(xu_i), ind_cons.ind_ub)
        zl_r_i = view(full(zl_i), ind_cons.ind_lb)
        zu_r_i = view(full(zu_i), ind_cons.ind_ub)
        dx_lr_i = view(d_i.xp, ind_cons.ind_lb)
        dx_ur_i = view(d_i.xp, ind_cons.ind_ub)
        
        obj_val_i = MadNLP._madnlp_unsafe_wrap(obj_val_batch, 1, i)
        inf_pr_i = MadNLP._madnlp_unsafe_wrap(inf_pr_batch, 1, i)
        inf_du_i = MadNLP._madnlp_unsafe_wrap(inf_du_batch, 1, i)
        inf_compl_i = MadNLP._madnlp_unsafe_wrap(inf_compl_batch, 1, i)
        norm_b_i = MadNLP._madnlp_unsafe_wrap(norm_b_batch, 1, i)
        norm_c_i = MadNLP._madnlp_unsafe_wrap(norm_c_batch, 1, i)
        mu_i = MadNLP._madnlp_unsafe_wrap(mu_batch, 1, i)
        alpha_p_i = MadNLP._madnlp_unsafe_wrap(alpha_p_batch, 1, i)
        alpha_d_i = MadNLP._madnlp_unsafe_wrap(alpha_d_batch, 1, i)
        del_w_i = MadNLP._madnlp_unsafe_wrap(del_w_batch, 1, i)
        del_c_i = MadNLP._madnlp_unsafe_wrap(del_c_batch, 1, i)
        best_complementarity_i = MadNLP._madnlp_unsafe_wrap(best_complementarity_batch, 1, i)
        mu_curr_i = MadNLP._madnlp_unsafe_wrap(mu_curr_batch, 1, i)

        cnt.init_time = time() - cnt.start_time

        solvers[i] = MPCSolver(
            nlps[i], class, batch_cb.callbacks[i], batch_kkt.kkts[i],
            ipm_opt, cnt, options.logger,
            n, m, nlb, nub,
            x_i, y_i, zl_i, zu_i, xl_i, xu_i,
            obj_val_i, f_i, c_i,
            jacl_i,
            d_i, p_i,
            _w1_i, _w2_i,
            correction_lb_i, correction_ub_i,
            rhs_i,
            ind_cons.ind_ineq, ind_cons.ind_fixed, ind_cons.ind_llb, ind_cons.ind_uub,
            ind_cons.ind_lb, ind_cons.ind_ub,
            x_lr_i, x_ur_i, xl_r_i, xu_r_i, zl_r_i, zu_r_i, dx_lr_i, dx_ur_i,
            inf_pr_i, inf_du_i, inf_compl_i, norm_b_i, norm_c_i, mu_i, alpha_p_i, alpha_d_i, del_w_i, del_c_i, best_complementarity_i, mu_curr_i,
            MadNLP.INITIAL,
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
    
    batch = SameStructureBatchMPCSolver(
        nlps, batch_size, solvers,
        x_batch, zl_batch, zu_batch, xl_batch, xu_batch, f_batch,
        d_batch, p_batch, _w1_batch, _w2_batch,
        y_batch, c_batch, jacl_batch, correction_lb_batch, correction_ub_batch, rhs_batch,
        obj_val_batch, dobj_val_batch, inf_pr_batch, inf_du_batch, inf_compl_batch, norm_b_batch, norm_c_batch,
        mu_batch, alpha_p_batch, alpha_d_batch, del_w_batch, del_c_batch,
        best_complementarity_batch, mu_affine_batch, mu_curr_batch,
        ipm_opt, bcnt, options.logger, batch_kkt, batch_cb, class,
        n, m, nlb, nub, nx, ns,
        ind_cons.ind_ineq, ind_cons.ind_fixed, ind_cons.ind_lb, ind_cons.ind_ub,
        ind_cons.ind_llb, ind_cons.ind_uub,
        x_lr_batch, x_ur_batch, xl_r_batch, xu_r_batch, zl_r_batch, zu_r_batch, dx_lr_batch, dx_ur_batch,
    )

    bcnt.init_time = time() - bcnt.start_time

    return batch
end
