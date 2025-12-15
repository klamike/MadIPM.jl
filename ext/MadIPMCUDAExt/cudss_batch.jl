mutable struct CUDSSUniformBatchSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, CUDSS.CudssSolver}
    tril::CUSPARSE.CuSparseMatrixCSC{T}
    nzVal::CuVector{T}
    x_gpu::CUDSS.CudssMatrix{T}
    b_gpu::CUDSS.CudssMatrix{T}

    n::Int
    nbatch::Int

    opt::MadNLPGPU.CudssSolverOptions
    logger::MadNLP.MadNLPLogger
end

function CUDSSUniformBatchSolver(
    csc::CUSPARSE.CuSparseMatrixCSC{T,Cint},
    nzVal::CuVector{T},
    nbatch::Int;
    opt=MadNLPGPU.CudssSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
) where T
    n, m = size(csc)
    @assert n == m
    nnz_per_batch = length(csc.nzVal)
    @assert length(nzVal) == nnz_per_batch * nbatch "nzVal length $(length(nzVal)) != nnz*nbatch $(nnz_per_batch * nbatch)"

    view = 'U'
    structure = "G"
    # We need view = 'F' for the sparse LU decomposition
    (opt.cudss_algorithm == MadNLP.LU) && error(logger, "The sparse LU of cuDSS is not supported.")
    (opt.cudss_algorithm == MadNLP.CHOLESKY) && (structure = "SPD")
    (opt.cudss_algorithm == MadNLP.LDL) && (structure = "S")

    solver = CUDSS.CudssSolver(csc.colPtr, csc.rowVal, nzVal, structure, view)
    CUDSS.cudss_set(solver, "ubatch_size", nbatch)
    MadNLPGPU.set_cudss_options!(solver, opt)

    if opt.cudss_ordering != MadNLPGPU.DEFAULT_ORDERING
        error(logger, "Custom ordering is not supported for uniform batch solver.")
    end

    # The phase "analysis" is "reordering" combined with "symbolic_factorization"
    @error "Construction" n nbatch nnz_per_batch length(nzVal)
    x_gpu = CUDSS.CudssMatrix(T, n; nbatch)
    b_gpu = CUDSS.CudssMatrix(T, n; nbatch)
    @error "Batch Analysis call" typeof(solver) typeof(x_gpu) typeof(b_gpu)
    CUDSS.cudss("analysis", solver, x_gpu, b_gpu)

    if opt.cudss_ir > 0
        error(logger, "No IR for batch")
    end
    return CUDSSUniformBatchSolver(
        solver,
        csc, nzVal,
        x_gpu, b_gpu,
        n, nbatch,
        opt, logger,
    )
end

function MadNLP.factorize!(M::CUDSSUniformBatchSolver)
    @error "Batch Factorize update" typeof(M.inner.matrix) typeof(M.nzVal) size(M.nzVal)
    CUDSS.cudss_update(M.inner.matrix, M.nzVal)
    if M.inner.fresh_factorization
        @error "Batch Factorize call" typeof(M.inner) typeof(M.x_gpu) typeof(M.b_gpu)
        CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu)
    else
        @error "Batch Refactorize call" typeof(M.inner) typeof(M.x_gpu) typeof(M.b_gpu)
        CUDSS.cudss("refactorization", M.inner, M.x_gpu, M.b_gpu)
    end
    if !M.opt.cudss_hybrid_memory && !M.opt.cudss_hybrid_execute
        CUDA.synchronize()
    end
    return M
end

function MadNLP.solve!(M::CUDSSUniformBatchSolver{T}, xb) where T
    @error "Batch b update" typeof(M.b_gpu) typeof(xb) size(xb)
    CUDSS.cudss_update(M.b_gpu, xb)
    @error "Batch x update" typeof(M.b_gpu) typeof(xb) size(xb)
    CUDSS.cudss_update(M.x_gpu, xb)
    @error "Batch Solve call" typeof(M.inner) typeof(M.x_gpu) typeof(M.b_gpu)
    CUDSS.cudss("solve", M.inner, M.x_gpu, M.b_gpu, asynchronous=true)
    if !M.opt.cudss_hybrid_memory && !M.opt.cudss_hybrid_execute
        CUDA.synchronize()
    end
    return xb
end

MadNLP.input_type(::Type{CUDSSUniformBatchSolver}) = :csc
MadNLP.default_options(::Type{CUDSSUniformBatchSolver}) = MadNLPGPU.CudssSolverOptions()
MadNLP.is_inertia(M::CUDSSUniformBatchSolver) = false

function MadNLP.inertia(::CUDSSUniformBatchSolver)
    error("No inertia for CUDSS batch")
end

MadNLP.improve!(M::CUDSSUniformBatchSolver) = false
MadNLP.is_supported(::Type{CUDSSUniformBatchSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{CUDSSUniformBatchSolver}, ::Type{Float64}) = true
MadNLP.introduce(M::CUDSSUniformBatchSolver) = "cuDSS Uniform Batch v$(CUDSS.version()) (nbatch=$(M.nbatch))"

function MadIPM.SparseSameStructureBatchKKTSystem(
    ::Type{KKT},
    batch_cb::MadIPM.SparseBatchCallback,
    ind_cons,
    linear_solver::Type{<:MadNLPGPU.CUDSSSolver};
    opt_linear_solver=MadNLP.default_options(linear_solver),
) where {KKT <: MadNLP.SparseKKTSystem}

    batch_size = batch_cb.batch_size
    batch_size == 0 && error("SparseSameStructureBatchKKTSystem requires at least one callback")

    cb1 = batch_cb.callbacks[1]
    T = eltype(cb1.con_buffer)
    VT = typeof(cb1.con_buffer)

    n_slack = length(ind_cons.ind_ineq)
    n = cb1.nvar
    m = cb1.ncon

    jac_sparsity_I = MadNLP.create_array(cb1, Int32, cb1.nnzj)
    jac_sparsity_J = MadNLP.create_array(cb1, Int32, cb1.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb1, jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I, hess_sparsity_J = MadNLP.build_hessian_structure(cb1, MadNLP.ExactHessian)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    MadNLP.force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    ind_ineq = ind_cons.ind_ineq

    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    aug_vec_length = n_tot + m
    aug_mat_length = n_tot + m + n_hess + n_jac + n_slack

    aug_I = MadNLP.create_array(cb1, Int32, aug_mat_length)
    aug_J = MadNLP.create_array(cb1, Int32, aug_mat_length)
    nzVals = VT(undef, aug_mat_length * batch_size)
    fill!(nzVals, zero(T))

    offset = n_tot + n_jac + n_slack + n_hess + m

    aug_I[1:n_tot] .= 1:n_tot
    aug_I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    aug_I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I .+ n_tot)
    aug_I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    aug_I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    aug_J[1:n_tot] .= 1:n_tot
    aug_J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    aug_J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    aug_J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    aug_J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    jac_I = Int32[jac_sparsity_I; ind_ineq]
    jac_J = Int32[jac_sparsity_J; n+1:n+n_slack]


    cb_1 = batch_cb.callbacks[1]
    V_1 = MadNLP._madnlp_unsafe_wrap(nzVals, aug_mat_length, 1)

    pr_diag_1 = MadNLP._madnlp_unsafe_wrap(V_1, n_tot)
    du_diag_1 = MadNLP._madnlp_unsafe_wrap(V_1, m, n_jac + n_slack + n_hess + n_tot + 1)

    reg_1 = VT(undef, n_tot)
    l_diag_1 = VT(undef, nlb)
    u_diag_1 = VT(undef, nub)
    l_lower_1 = VT(undef, nlb)
    u_lower_1 = VT(undef, nub)

    hess_1 = MadNLP._madnlp_unsafe_wrap(V_1, n_hess, n_tot + 1)
    jac_1 = MadNLP._madnlp_unsafe_wrap(V_1, n_jac + n_slack, n_hess + n_tot + 1)
    jac_callback_1 = MadNLP._madnlp_unsafe_wrap(V_1, n_jac, n_hess + n_tot + 1)

    aug_raw_1 = MadNLP.SparseMatrixCOO(aug_vec_length, aug_vec_length, aug_I, aug_J, V_1)
    jac_raw_1 = MadNLP.SparseMatrixCOO(
        m, n_tot,
        jac_I,
        jac_J,
        jac_1
    )

    hess_raw_1 = MadNLP.SparseMatrixCOO(
        n_tot, n_tot,
        hess_sparsity_I,
        hess_sparsity_J,
        hess_1
    )
    
    aug_com_1, aug_csc_map_1 = MadNLP.coo_to_csc(aug_raw_1)
    jac_com_1, jac_csc_map_1 = MadNLP.coo_to_csc(jac_raw_1)
    hess_com_1, hess_csc_map_1 = MadNLP.coo_to_csc(hess_raw_1)

    nnz_csc = length(aug_com_1.nzVal)
    csc_nzVals = VT(undef, nnz_csc * batch_size)
    fill!(csc_nzVals, zero(T))

    aug_com_nzVal_1 = MadNLP._madnlp_unsafe_wrap(csc_nzVals, nnz_csc, 1)
    aug_com_1 = CUSPARSE.CuSparseMatrixCSC(aug_com_1.colPtr, aug_com_1.rowVal, aug_com_nzVal_1, aug_com_1.dims)

    quasi_newton_1 = MadNLP.create_quasi_newton(MadNLP.ExactHessian, cb_1, n)
    linear_solver_1 = linear_solver(aug_com_1; opt=opt_linear_solver)

    kkt_1 = MadNLP.SparseKKTSystem(
        hess_1, jac_callback_1, jac_1, quasi_newton_1, reg_1, pr_diag_1, du_diag_1,
        l_diag_1, u_diag_1, l_lower_1, u_lower_1,
        aug_raw_1, aug_com_1, aug_csc_map_1,
        hess_raw_1, hess_com_1, hess_csc_map_1,
        jac_raw_1, jac_com_1, jac_csc_map_1,
        linear_solver_1,
        ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
    )

    KKTType = typeof(kkt_1)
    kkts = Vector{KKTType}(undef, batch_size)
    kkts[1] = kkt_1

    for i in 2:batch_size
        cb_i = batch_cb.callbacks[i]
        val_offset = (i - 1) * aug_mat_length
        csc_offset = (i - 1) * nnz_csc

        V_i = MadNLP._madnlp_unsafe_wrap(nzVals, aug_mat_length, val_offset + 1)

        pr_diag_i = MadNLP._madnlp_unsafe_wrap(V_i, n_tot)
        du_diag_i = MadNLP._madnlp_unsafe_wrap(V_i, m, n_jac + n_slack + n_hess + n_tot + 1)

        reg_i = VT(undef, n_tot)
        l_diag_i = VT(undef, nlb)
        u_diag_i = VT(undef, nub)
        l_lower_i = VT(undef, nlb)
        u_lower_i = VT(undef, nub)

        hess_i = MadNLP._madnlp_unsafe_wrap(V_i, n_hess, n_tot + 1)
        jac_i = MadNLP._madnlp_unsafe_wrap(V_i, n_jac + n_slack, n_hess + n_tot + 1)
        jac_callback_i = MadNLP._madnlp_unsafe_wrap(V_i, n_jac, n_hess + n_tot + 1)

        aug_raw_i = MadNLP.SparseMatrixCOO(aug_vec_length, aug_vec_length, aug_I, aug_J, V_i)
        jac_raw_i = MadNLP.SparseMatrixCOO(
            m, n_tot,
            jac_I,
            jac_J,
            jac_i
        )
        hess_raw_i = MadNLP.SparseMatrixCOO(
            n_tot, n_tot,
            hess_sparsity_I,
            hess_sparsity_J,
            hess_i
        )

        aug_com_nzVal_i = MadNLP._madnlp_unsafe_wrap(csc_nzVals, nnz_csc, csc_offset + 1)
        aug_com_i = CUSPARSE.CuSparseMatrixCSC(aug_com_1.colPtr, aug_com_1.rowVal, aug_com_nzVal_i, aug_com_1.dims)

        jac_com_i, jac_csc_map_i = MadNLP.coo_to_csc(jac_raw_i)
        hess_com_i, hess_csc_map_i = MadNLP.coo_to_csc(hess_raw_i)

        quasi_newton_i = MadNLP.create_quasi_newton(MadNLP.ExactHessian, cb_i, n)

        linear_solver_i = linear_solver(aug_com_i; opt=opt_linear_solver)

        kkts[i] = MadNLP.SparseKKTSystem(
            hess_i, jac_callback_i, jac_i, quasi_newton_i, reg_i, pr_diag_i, du_diag_i,
            l_diag_i, u_diag_i, l_lower_i, u_lower_i,
            aug_raw_i, aug_com_i, aug_csc_map_1,
            hess_raw_i, hess_com_i, hess_csc_map_i,
            jac_raw_i, jac_com_i, jac_csc_map_i,
            linear_solver_i,
            ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
        )
    end

    batch_solver = CUDSSUniformBatchSolver(
        aug_com_1, csc_nzVals,
        batch_size;
        opt=opt_linear_solver,
    )

    return MadIPM.SparseSameStructureBatchKKTSystem(
        nzVals,
        aug_I, aug_J,
        kkts,
        batch_solver,
        aug_mat_length, batch_size,
    )
end
