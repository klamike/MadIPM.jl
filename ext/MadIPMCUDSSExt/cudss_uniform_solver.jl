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
    x_gpu = CUDSS.CudssMatrix(T, n; nbatch)
    b_gpu = CUDSS.CudssMatrix(T, n; nbatch)
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
    CUDSS.cudss_update(M.inner.matrix, M.nzVal)
    if M.inner.fresh_factorization
        CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu)
    else
        CUDSS.cudss("refactorization", M.inner, M.x_gpu, M.b_gpu)
    end
    if !M.opt.cudss_hybrid_memory && !M.opt.cudss_hybrid_execute
        CUDA.synchronize()
    end
    return M
end

function MadNLP.solve!(M::CUDSSUniformBatchSolver{T}, xb) where T
    CUDSS.cudss_update(M.b_gpu, xb)
    CUDSS.cudss_update(M.x_gpu, xb)
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
