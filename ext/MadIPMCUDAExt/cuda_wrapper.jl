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

MadIPM.@sync_annotate function MadNLP.transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC{Tv},
    src::MadNLP.SparseMatrixCOO{Tv},
    map::CuVector{Int},
) where {Tv}
    return MadNLP._transfer!(dest.nzVal, src.V, map)
end

MadIPM.@sync_annotate function MadNLP._transfer!(dest::CuVector{T}, src::CuVector{T}, map::CuVector{Int}) where T
    fill!(dest, zero(T))
    if length(map) > 0
        backend = CUDABackend()
        _transfer_to_map!(backend)(dest, map, src; ndrange=length(map))
    end
    return
end

MadIPM.@sync_annotate function MadNLP.compress_hessian!(
    kkt::MadNLP.SparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

MadIPM.@sync_annotate function MadNLP.compress_jacobian!(
    kkt::MadIPM.NormalKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    n_slack = length(kkt.ind_ineq)
    kkt.A.V[end-n_slack+1:end] .= -1.0
    # Transfer to the matrix A stored in CSC format
    fill!(kkt.AT.nzVal, 0.0)
    kkt.AT.nzVal .= kkt.A.V[kkt.A_csr_map]
    return
end

MadIPM.@sync_annotate function MadIPM.coo_to_csr(
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

MadIPM.@sync_annotate function MadIPM.assemble_normal_system!(
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

MadIPM.@sync_annotate function MadIPM.build_normal_system(
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

MadIPM.@sync_annotate function MadIPM._coo_to_scatter(
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

MadIPM.@sync_annotate function _init_cudss_mempool_fptrs!()
    _CUDSS_POOL_ALLOC_FPTR[] = @cfunction(
        _cudss_pool_alloc, Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, CUDA.CUstream))
    _CUDSS_POOL_FREE_FPTR[] = @cfunction(
        _cudss_pool_free, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, CUDA.CUstream))
    return
end

MadIPM.@sync_annotate function _set_cudss_mempool!(handle::CUDSS.cudssHandle_t)
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
MadIPM.@sync_annotate function MadNLPGPU.CUDSSSolver(
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

#=
    Segmented CUDA graph capture.

    mpc_step! has 3 CUDSS calls that cannot be captured in CUDA graphs when
    ubatch_size > 1. We split mpc_step! into 4 graph segments around these calls.

    During capture: each CUDSS dispatch ends the current segment capture, runs
    CUDSS eagerly (on stale data — harmless), and starts the next segment capture.

    During replay: segments are launched on the stream with CUDSS calls enqueued
    between them. Stream ordering guarantees correctness without explicit syncs.
=#

# --- Low-level graph helpers (raw handles) ---

function _begin_segment_capture(stream::CUDA.CuStream)
    CUDA.cuStreamBeginCapture_v2(stream, CUDA.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
end

function _end_segment_capture(stream::CUDA.CuStream)
    graph_ref = Ref{CUDA.CUgraph}()
    err = CUDA.unchecked_cuStreamEndCapture(stream, graph_ref)
    err != CUDA.CUDA_SUCCESS && throw(CUDA.CuError(err))
    return graph_ref[]
end

function _instantiate_raw(graph::CUDA.CUgraph)
    exec_ref = Ref{CUDA.CUgraphExec}()
    CUDA.cuGraphInstantiateWithFlags(exec_ref, graph, UInt64(0))
    return exec_ref[]
end

function _launch_raw(exec::CUDA.CUgraphExec, stream::CUDA.CuStream)
    CUDA.cuGraphLaunch(exec, stream)
end

# --- Segment boundary helpers ---

function _segment_break_capture!(cudss_work)
    stream = CUDA.stream()
    cache = _SEG_CACHE[]

    # End current segment capture
    _CAPTURING[] = false
    graph = _end_segment_capture(stream)
    exec = _instantiate_raw(graph)
    push!(cache.graphs, graph)
    push!(cache.execs, exec)
    push!(cache.cudss_calls, cudss_work)

    # Run CUDSS eagerly (data may be stale during capture — that's OK)
    cudss_work()

    # Start next segment capture
    _CURRENT_SEGMENT[] += 1
    _CAPTURING[] = true
    _begin_segment_capture(stream)
end

function _segment_break_replay!(cudss_work)
    cache = _SEG_CACHE[]
    seg = _CURRENT_SEGMENT[]
    _launch_raw(cache.execs[seg], CUDA.stream())
    cudss_work()  # stream-ordered after graph launch, no sync needed
    _CURRENT_SEGMENT[] += 1
end

# --- CUDSS dispatch with segment break support ---

MadIPM.@sync_annotate function MadIPM._active_factorize!(s::MadNLPGPU.CUDSSSolver, na::Int)
    cudss_work = () -> begin
        CUDSS.cudss_set(s.inner, "ubatch_size", na)
        MadNLP.factorize!(s)
    end
    mode = _SEGMENTED_MODE[]
    if mode === :capturing
        _segment_break_capture!(cudss_work)
    elseif mode === :replaying
        _segment_break_replay!(cudss_work)
    else
        cudss_work()
    end
    return
end

MadIPM.@sync_annotate function MadIPM._active_solve!(s::MadNLPGPU.CUDSSSolver{T}, rhs::CuMatrix{T}, na::Int, n::Int) where T
    cudss_work = () -> begin
        CUDSS.cudss_set(s.inner, "ubatch_size", na)
        CUDSS.cudss_update(s.b_gpu, rhs)
        CUDSS.cudss_update(s.x_gpu, rhs)
        CUDSS.cudss("solve", s.inner, s.x_gpu, s.b_gpu, asynchronous=s.opt.cudss_asynchronous)
    end
    mode = _SEGMENTED_MODE[]
    if mode === :capturing
        _segment_break_capture!(cudss_work)
    elseif mode === :replaying
        _segment_break_replay!(cudss_work)
    else
        cudss_work()
    end
    return
end

# --- Segmented capture and replay ---

function _capture_all!(cache::SegmentedGraphCache, batch_solver, na::Int)
    stream = CUDA.stream()
    gc_state = GC.enable(false)  # no GC during any capture segment
    _CURRENT_SEGMENT[] = 1
    _SEGMENTED_MODE[] = :capturing
    _CAPTURING[] = true
    ok = true
    try
        _begin_segment_capture(stream)
        MadIPM._captured_work!(batch_solver)
        # mpc_step! triggered 3 CUDSS breaks → segments 1-3 captured, now on segment 4
        # End the final segment
        _CAPTURING[] = false
        graph = _end_segment_capture(stream)
        exec = _instantiate_raw(graph)
        push!(cache.graphs, graph)
        push!(cache.execs, exec)
        cache.n_segments = _CURRENT_SEGMENT[]
        cache.na = na
        cache.valid = true
    catch e
        ok = false
        _CAPTURING[] = false
        @warn "Segmented graph capture failed" exception=(e, catch_backtrace())
        # Try to clean up any in-progress capture to avoid stream corruption
        try
            _end_segment_capture(stream)
        catch
        end
        _invalidate_seg_cache!()
    finally
        _SEGMENTED_MODE[] = :off
        _CAPTURING[] = false
        GC.enable(gc_state)
    end

    if ok
        # Replay to get correct first-iteration results
        # (capture-time kernel work was recorded but not executed)
        _replay_all!(cache)
    else
        # Capture failed (e.g. JIT on first call) — run ungraphed, retry next iteration
        MadIPM._captured_work!(batch_solver)
    end
end

function _replay_all!(cache::SegmentedGraphCache)
    stream = CUDA.stream()
    n = cache.n_segments
    for seg in 1:n
        _launch_raw(cache.execs[seg], stream)
        if seg < n
            cache.cudss_calls[seg]()  # stream-ordered after graph
        end
    end
end

MadIPM.@sync_annotate function MadIPM._captured_step!(
    batch_solver::MadIPM.UniformBatchMPCSolver{T, <:CuMatrix{T}},
) where T
    # Segmented capture: CUDSS calls run outside of graph capture,
    # so no mempool requirement (_GRAPH_CAPTURE_OK not checked).
    na = batch_solver.kkt.active_batch_size[]
    cache = _SEG_CACHE[]

    # Replay if cache is valid for current active batch size
    if cache.valid && cache.na == na
        _replay_all!(cache)
        return
    end

    # (Re-)capture: na changed or first call
    _invalidate_seg_cache!()
    _capture_all!(cache, batch_solver, na)
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