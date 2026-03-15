using MadNLPGPU
import LinearAlgebra: BlasFloat

MadIPM._scratch_view(scratch::CuMatrix{T}, k, bs) where T =
    CUDA.unsafe_wrap(CuArray{T, 2}, pointer(scratch), (k, bs))

@kernel function _transfer_to_map!(dest, to_map, src)
    k = @index(Global, Linear)
    @inbounds begin
        Atomix.@atomic dest[to_map[k]] += src[k]
    end
end

function MadNLP.transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC{Tv},
    src::MadNLP.SparseMatrixCOO{Tv},
    map::CuVector{Int},
) where {Tv}
    return MadNLP._transfer!(dest.nzVal, src.V, map)
end

function MadNLP._transfer!(dest::CuVector{T}, src::CuVector{T}, map::CuVector{Int}) where T
    fill!(dest, zero(T))
    if length(map) > 0
        backend = CUDABackend()
        _transfer_to_map!(backend)(dest, map, src; ndrange=length(map))
    end
    return
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.SparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadNLP.compress_jacobian!(
    kkt::MadIPM.NormalKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    n_slack = length(kkt.ind_ineq)
    kkt.A.V[end-n_slack+1:end] .= -1.0
    # Transfer to the matrix A stored in CSC format
    fill!(kkt.AT.nzVal, 0.0)
    kkt.AT.nzVal .= kkt.A.V[kkt.A_csr_map]
    return
end

function MadIPM.coo_to_csr(
    n_rows,
    n_cols,
    Ai::CuVector{Ti},
    Aj::CuVector{Ti},
    Ax::CuVector{Tv},
) where {Tv, Ti}
    @assert length(Ai) == length(Aj) == length(Ax)
    B = sparse(Ai, Aj, Ax, n_rows, n_cols; fmt=:csr)
    return (B.rowPtr, B.colVal, B.nzVal)
end

@kernel function assemble_normal_system_kernel!(@Const(n_rows), @Const(n_cols), @Const(Jtp), @Const(Jtj), @Const(Jtx),
                                                @Const(Cp), @Const(Cj), Cx, @Const(Dx), @Const(Tv))
    i = @index(Global, Linear)

    for c in Cp[i]:Cp[i+1]-1
        j = Cj[c]
        acc = zero(Tv)

        p1 = Jtp[i]
        p2 = Jtp[j]
        p1_end = Jtp[i+1] - 1
        p2_end = Jtp[j+1] - 1

        while p1 <= p1_end && p2 <= p2_end
            k1 = Jtj[p1]
            k2 = Jtj[p2]

            if k1 == k2
                acc += Jtx[p1] * Dx[k1] * Jtx[p2]
                p1 += 1
                p2 += 1
            elseif k1 < k2
                p1 += 1
            else
                p2 += 1
            end
        end

        Cx[c] = acc
    end
    nothing
end

function MadIPM.assemble_normal_system!(
    n_rows,
    n_cols,
    Jtp::CuArray{Ti},
    Jtj::CuArray{Ti},
    Jtx::CuArray{Tv},
    Cp::CuArray{Ti},
    Cj::CuArray{Ti},
    Cx::CuArray{Tv},
    Dx::CuArray{Tv},
) where {Ti, Tv}
    backend = CUDABackend()
    kernel! = assemble_normal_system_kernel!(backend)
    kernel!(n_rows, n_cols, Jtp, Jtj, Jtx, Cp, Cj, Cx, Dx, Tv; ndrange = n_rows)
end

@kernel function count_normal_nnz!(Cp, @Const(Jtp), @Const(Jtj), @Const(n_rows), @Const(n_cols))
    i = @index(Global, Linear)

    # thread-local binary buffer
    xb = @localmem UInt8 n_cols
    for k = 1:n_cols
        xb[k] = 0
    end

    for c = Jtp[i]:Jtp[i+1]-1
        j = Jtj[c]
        xb[j] = 1
    end

    count = 0
    for j = i:n_rows
        for c = Jtp[j]:Jtp[j+1]-1
            k = Jtj[c]
            if xb[k] == 1
                count += 1
                break
            end
        end
    end

    Cp[i+1] = count
    nothing
end

@kernel function fill_normal_indices!(Cj, @Const(Cp), @Const(Jtp), @Const(Jtj), @Const(n_rows), @Const(n_cols))
    i = @index(Global, Linear)

    xb = @localmem UInt8 n_cols
    for k = 1:n_cols
        xb[k] = 0
    end

    for c = Jtp[i]:Jtp[i+1]-1
        j = Jtj[c]
        xb[j] = 1
    end

    pos = Cp[i]
    for j = i:n_rows
        for c = Jtp[j]:Jtp[j+1]-1
            k = Jtj[c]
            if xb[k] == 1
                Cj[pos] = j
                pos += 1
                break
            end
        end
    end
    nothing
end

function MadIPM.build_normal_system(
    n_rows,
    n_cols,
    Jtp::CuVector{Ti},
    Jtj::CuVector{Ti},
) where {Ti}
    backend = CUDABackend()
    Cp = CUDA.ones(Ti, n_rows + 1)
    kernel1! = count_normal_nnz!(backend)
    kernel1!(Cp, Jtp, Jtj, n_rows, n_cols; ndrange = n_rows)

    Cp = cumsum(Cp)
    nnz_JtJ = CUDA.@allowscalar (Cp[end] - 1)
    Cj = CUDA.zeros(Ti, nnz_JtJ)

    kernel2! = fill_normal_indices!(backend)
    kernel2!(Cj, Cp, Jtp, Jtj, n_rows, n_cols; ndrange = n_rows)
    return (Cp, Cj)
end

MadIPM.sparse_csc_format(::Type{<:CuArray}) = CuSparseMatrixCSC
MadIPM._colptr(A::CuSparseMatrixCSC) = A.colPtr
MadIPM._rowval(A::CuSparseMatrixCSC) = A.rowVal
MadIPM._nzval(A::CuSparseMatrixCSC) = A.nzVal

function MadIPM._coo_to_scatter(
    coo_I, nrows::Int, n_entries::Int,
    proto_I, nzVals::CuMatrix{T}, batch_size::Int,
) where T
    if n_entries == 0
        rowptr = CUDA.ones(Int, nrows + 1)
        colidx = CuVector{Int}(undef, 0)
        return rowptr, colidx
    end
    coo_J = similar(proto_I, n_entries)
    coo_J .= Int32(1):Int32(n_entries)
    coo_V = similar(nzVals, n_entries)
    fill!(coo_V, one(T))
    scatter, _ = MadNLP.coo_to_csc(
        MadNLP.SparseMatrixCOO(nrows, n_entries, coo_I, coo_J, coo_V),
    )
    # Build CSR on host. Only used when first creating KKT system
    rowval_h = Int.(Array(MadIPM._rowval(scatter)))
    counts = zeros(Int, nrows)
    for r in rowval_h; counts[r] += 1; end
    rowptr_h = Vector{Int}(undef, nrows + 1)
    rowptr_h[1] = 1
    for r in 1:nrows; rowptr_h[r+1] = rowptr_h[r] + counts[r]; end
    colidx_h = Vector{Int}(undef, n_entries)
    pos = copy(rowptr_h[1:end-1])
    for k in 1:n_entries
        r = rowval_h[k]
        colidx_h[pos[r]] = k
        pos[r] += 1
    end
    rowptr = CuVector{Int}(rowptr_h)
    colidx = CuVector{Int}(colidx_h)
    return rowptr, colidx
end

#=
    CUDSS stream-ordered memory pool handler

    Setting this causes CUDSS to use cuMemAllocAsync/cuMemFreeAsync for its internal
    scratch allocations, which (a) eliminates device-synchronizing cudaMalloc stalls,
    and (b) makes those allocations capturable as graph memory nodes when used inside
    a CUDA graph.
=#

_cudss_pool_alloc(
    ctx::Ptr{Cvoid}, ptr::Ptr{Ptr{Cvoid}}, size::Csize_t, stream::CUDA.CUstream,
)::Cint = begin
    # Raw ccall: bypass initialize_context() which is unsafe to call from a C callback.
    # CUdeviceptr is UInt64 in the C ABI; CUresult is UInt32 (0 = CUDA_SUCCESS).
    p = Ref{UInt64}(zero(UInt64))
    err = ccall((:cuMemAllocAsync, CUDA.libcuda), UInt32,
                (Ptr{UInt64}, Csize_t, CUDA.CUstream), p, size, stream)
    err == 0 || return Cint(1)
    unsafe_store!(Ptr{UInt64}(ptr), p[])
    return Cint(0)
end

_cudss_pool_free(
    ctx::Ptr{Cvoid}, ptr::Ptr{Cvoid}, size::Csize_t, stream::CUDA.CUstream,
)::Cint = begin
    # ptr holds the device address as raw bits (stored by _cudss_pool_alloc above)
    err = ccall((:cuMemFreeAsync, CUDA.libcuda), UInt32,
                (UInt64, CUDA.CUstream), UInt64(UInt(ptr)), stream)
    return err == 0 ? Cint(0) : Cint(1)
end

# Initialized in __init__ to avoid stale pointers in precompile cache
const _CUDSS_POOL_ALLOC_FPTR = Ref{Ptr{Cvoid}}(C_NULL)
const _CUDSS_POOL_FREE_FPTR  = Ref{Ptr{Cvoid}}(C_NULL)

function _init_cudss_mempool_fptrs!()
    _CUDSS_POOL_ALLOC_FPTR[] = @cfunction(
        _cudss_pool_alloc, Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, CUDA.CUstream))
    _CUDSS_POOL_FREE_FPTR[] = @cfunction(
        _cudss_pool_free, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, CUDA.CUstream))
    return
end

function _set_cudss_mempool!(handle::CUDSS.cudssHandle_t)
    name = ntuple(64) do i
        s = "cuda_stream_pool"
        i <= ncodeunits(s) ? Cchar(codeunit(s, i)) : Cchar(0)
    end
    handler = CUDSS.cudssDeviceMemHandler_t(
        C_NULL, _CUDSS_POOL_ALLOC_FPTR[], _CUDSS_POOL_FREE_FPTR[], name)
    # Struct is copied internally by cudssSetDeviceMemHandler; no need to preserve
    try
        CUDSS.cudssSetDeviceMemHandler(handle, Ref(handler))
        return true
    catch e
        e isa CUDSS.CUDSSError || rethrow()
        @info "Failed to set cuDSS memhandler: $(e)"
        # Feature not supported by this CUDSS version/config; skip silently
        return false
    end
end

# we introduce a new constructor that takes the nzvals as a matrix explicitly
function MadNLPGPU.CUDSSSolver(
    aug_com::CUSPARSE.CuSparseMatrixCSC{T,Cint},
    nzvals_mat::CuMatrix{T},
    n::Int;
    opt::MadNLPGPU.CudssSolverOptions = MadNLPGPU.CudssSolverOptions(),
) where T
    batch_nzVal = vec(nzvals_mat)
    batch_aug_com = CUSPARSE.CuSparseMatrixCSC(
        aug_com.colPtr, aug_com.rowVal, batch_nzVal, size(aug_com),
    )
    # Set mempool BEFORE creating solver (batch mode rejects it after ubatch_size is set)
    _GRAPH_CAPTURE_OK[] = _set_cudss_mempool!(CUDSS.handle())
    solver = MadNLPGPU.CUDSSSolver(nothing, batch_aug_com; opt=opt)
    solver.tril.nzVal = batch_nzVal
    return solver
end

MadIPM.is_factorized(::MadNLPGPU.CUDSSSolver) = true

# --- CUDSS dispatch: annotated as graph breaks via CUDAGraphs.@graphbreak ---

function MadIPM._active_factorize!(s::MadNLPGPU.CUDSSSolver, na::Int)
    CUDSS.cudss_set(s.inner, "ubatch_size", na)
    MadNLP.factorize!(s)
    return
end

function MadIPM._active_solve!(s::MadNLPGPU.CUDSSSolver{T}, rhs::CuMatrix{T}, na::Int, n::Int) where T
    CUDSS.cudss_set(s.inner, "ubatch_size", na)
    CUDSS.cudss_update(s.b_gpu, rhs)
    CUDSS.cudss_update(s.x_gpu, rhs)
    CUDSS.cudss("solve", s.inner, s.x_gpu, s.b_gpu, asynchronous=s.opt.cudss_asynchronous)
    return
end

# --- Segmented graph capture via CUDAGraphs.@unsafe_scaptured ---

function MadIPM._captured_step!(
    batch_solver::MadIPM.UniformBatchMPCSolver{T, <:CuMatrix{T}},
) where T
    na = batch_solver.kkt.active_batch_size[]
    cache = _SEG_CACHE[]

    # Invalidate when active batch size changes (na is a frozen scalar in closures)
    if _SEG_CACHE_NA[] != na
        CUDAGraphs.invalidate!(cache)
        _SEG_CACHE_NA[] = na
    end

    CUDAGraphs.@unsafe_scaptured cache begin
        MadIPM._captured_work!(batch_solver)
    end
end

function MadNLPGPU.CUDSSSolver(
    ::Nothing,
    csc::CUSPARSE.CuSparseMatrixCSC{T,Cint};
    opt=CudssSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
    ) where T
    n, m = size(csc)
    @assert n == m

    view = 'U'
    structure = 'G'
    # We need view = 'F' for the sparse LU decomposition
    (opt.cudss_algorithm == MadNLP.LU) && error(logger, "The sparse LU of cuDSS is not supported.")
    (opt.cudss_algorithm == MadNLP.CHOLESKY) && (structure = "SPD")
    (opt.cudss_algorithm == MadNLP.LDL) && (structure = "S")

    solver = CUDSS.CudssSolver(csc.colPtr, csc.rowVal, csc.nzVal, structure, view)
    _set_cudss_mempool!(solver.data.handle)
    MadNLPGPU.set_cudss_options!(solver, opt)

    opt.cudss_ordering == MadNLPGPU.DEFAULT_ORDERING || error()
    # if opt.cudss_ordering != MadNLPGPU.DEFAULT_ORDERING
    #     if opt.cudss_ordering == MadNLPGPU.METIS_ORDERING
    #         A = SparseMatrixCSC(csc)
    #         A = A + A' - LinearAlgebra.Diagonal(A)
    #         G = Metis.graph(A, check_hermitian=false)
    #         opt.cudss_perm, _ = Metis.permutation(G)
    #     elseif opt.cudss_ordering == MadNLPGPU.AMD_ORDERING
    #         A = SparseMatrixCSC(csc)
    #         opt.cudss_perm = AMD.amd(A)
    #     elseif opt.cudss_ordering == MadNLPGPU.SYMAMD_ORDERING
    #         A = SparseMatrixCSC(csc)
    #         opt.cudss_perm = AMD.symamd(A)
    #     elseif opt.cudss_ordering == MadNLPGPU.COLAMD_ORDERING
    #         A = SparseMatrixCSC(csc)
    #         opt.cudss_perm = AMD.colamd(A)
    #     elseif opt.cudss_ordering == MadNLPGPU.USER_ORDERING
    #         (!isempty(opt.cudss_perm) && isperm(opt.cudss_perm)) || error(logger, "The vector opt.cudss_perm is not a valid permutation.")
    #     else
    #         error(logger, "The ordering $(opt.cudss_ordering) is not supported.")
    #     end
    #     CUDSS.cudss_set(solver, "user_perm", opt.cudss_perm)
    # end

    # Check if we want to use the batch solver for matrices with a common sparsity pattern
    nbatch = solver.matrix.nbatch
    if nbatch > 1
        CUDSS.cudss_set(solver, "ubatch_size", nbatch)
    end

    # The phase "analysis" is "reordering" combined with "symbolic_factorization"
    x_gpu = CUDSS.CudssMatrix(T, n; nbatch=nbatch)
    b_gpu = CUDSS.CudssMatrix(T, n; nbatch=nbatch)
    CUDSS.cudss("analysis", solver, x_gpu, b_gpu, asynchronous=opt.cudss_asynchronous)

    # Allocate additional buffer for iterative refinement
    # Always allocate it to support dynamic updates to opt.cudss_ir
    buffer = CuVector{T}(undef, n * nbatch)

    return MadNLPGPU.CUDSSSolver(
        solver, csc,
        x_gpu, b_gpu, buffer,
        opt, logger,
    )
end