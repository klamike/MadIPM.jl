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

function _build_batch_kkt_element(
    cb::MadNLP.SparseCallback,
    ind_cons,
    I::AbstractVector{Int32},
    J::AbstractVector{Int32},
    V::VT,
    structure::MadNLP.SparseKKTStructure,
    aug_com_template::CUSPARSE.CuSparseMatrixCSC,
    aug_csc_map,
    csc_nzVal::VT,
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
) where {VT}

    views = MadNLP._build_sparsekkt_views(VT, I, J, V, structure)

    aug_com = CUSPARSE.CuSparseMatrixCSC(
        aug_com_template.colPtr, aug_com_template.rowVal,
        csc_nzVal, aug_com_template.dims
    )
    jac_com, jac_csc_map = MadNLP.coo_to_csc(views.jac_raw)
    hess_com, hess_csc_map = MadNLP.coo_to_csc(views.hess_raw)

    # FIXME: batch QN?
    quasi_newton = MadNLP.create_quasi_newton(MadNLP.ExactHessian, cb, structure.n)
    return MadNLP.SparseKKTSystem(
        views.hess, views.jac_callback, views.jac, quasi_newton,
        views.reg, views.pr_diag, views.du_diag,
        views.l_diag, views.u_diag, views.l_lower, views.u_lower,
        views.aug_raw, aug_com, aug_csc_map,
        views.hess_raw, hess_com, hess_csc_map,
        views.jac_raw, jac_com, jac_csc_map,
        nothing,
        structure.ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
    )
end

function MadIPM.SparseSameStructureBatchKKTSystem(
    batch_cb::MadIPM.SparseBatchCallback,
    ind_cons,
    linear_solver::Type{<:MadNLPGPU.CUDSSSolver};
    opt_linear_solver=MadNLP.default_options(linear_solver),
)
    batch_size = batch_cb.batch_size
    batch_size == 0 && error("SparseSameStructureBatchKKTSystem requires at least one callback")

    cb1 = batch_cb.callbacks[1]
    T = eltype(cb1.con_buffer)
    VT = typeof(cb1.con_buffer)

    structure = MadNLP._build_sparsekkt_structure(cb1, ind_cons, MadNLP.ExactHessian)
    (; aug_mat_length) = structure

    aug_I = MadNLP.create_array(cb1, Int32, aug_mat_length)
    aug_J = MadNLP.create_array(cb1, Int32, aug_mat_length)
    MadNLP.build_aug_indices!(aug_I, aug_J, structure)

    nzVals = VT(undef, aug_mat_length * batch_size)
    fill!(nzVals, zero(T))

    V_1 = MadNLP._madnlp_unsafe_wrap(nzVals, aug_mat_length, 1)
    kkt_1_temp = MadNLP.build_sparse_kkt_system(
        cb1, ind_cons, aug_I, aug_J, V_1, structure, linear_solver;
        opt_linear_solver,
    )

    nnz_csc = length(kkt_1_temp.aug_com.nzVal)
    csc_nzVals = VT(undef, nnz_csc * batch_size)
    fill!(csc_nzVals, zero(T))

    aug_csc_map_1 = kkt_1_temp.aug_csc_map
    csc_nzVal_1 = MadNLP._madnlp_unsafe_wrap(csc_nzVals, nnz_csc, 1)
    kkt_1 = _build_batch_kkt_element(
        cb1, ind_cons, aug_I, aug_J, V_1, structure,
        kkt_1_temp.aug_com, aug_csc_map_1, csc_nzVal_1, linear_solver;
        opt_linear_solver,
    )

    KKTType = typeof(kkt_1)
    kkts = Vector{KKTType}(undef, batch_size)
    kkts[1] = kkt_1

    for i in 2:batch_size
        cb_i = batch_cb.callbacks[i]
        val_offset = (i - 1) * aug_mat_length
        csc_offset = (i - 1) * nnz_csc

        V_i = MadNLP._madnlp_unsafe_wrap(nzVals, aug_mat_length, val_offset + 1)
        csc_nzVal_i = MadNLP._madnlp_unsafe_wrap(csc_nzVals, nnz_csc, csc_offset + 1)

        kkts[i] = _build_batch_kkt_element(
            cb_i, ind_cons, aug_I, aug_J, V_i, structure,
            kkt_1.aug_com, aug_csc_map_1, csc_nzVal_i, linear_solver;
            opt_linear_solver,
        )
    end

    batch_solver = CUDSSUniformBatchSolver(
        kkt_1.aug_com, csc_nzVals,
        batch_size;
        opt=opt_linear_solver,
    )

    return MadIPM.SparseSameStructureBatchKKTSystem(
        nzVals, aug_I, aug_J,
        kkts, batch_solver,
        aug_mat_length, batch_size,
    )
end
