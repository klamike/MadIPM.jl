#=
MWE: CUDSS stream-ordered memory pool + graph capture

Tests whether CUDSS can use cuMemAllocAsync/cuMemFreeAsync for its
internal scratch, enabling CUDA graph capture of factorize+solve.

Usage:
  julia mwe_cudss_mempool.jl
=#

using CUDA, CUDSS, SparseArrays

@assert CUDA.functional() "CUDA not functional"
println("CUDA:  ", CUDA.runtime_version())
println("CUDSS: ", CUDSS.version())
println("GPU:   ", CUDA.name(CUDA.device()))

# ── Build a small SPD system: A = [4 1; 1 3] ──
n = 2
A_cpu = sparse([1,2,1,2], [1,1,2,2], [4.0, 1.0, 1.0, 3.0], n, n)
A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(A_cpu)
b_gpu = CuVector([1.0, 2.0])
x_gpu = CUDA.zeros(Float64, n)

# ── Set up CUDSS solver ──
solver = CUDSS.CudssSolver(A_gpu, "S", 'L')
CUDSS.cudss("analysis", solver, x_gpu, b_gpu)

# ── Define mempool callbacks ──
function pool_alloc(ctx::Ptr{Cvoid}, ptr::Ptr{Ptr{Cvoid}}, size::Csize_t, stream::CUDA.CUstream)::Cint
    p = Ref{UInt64}(zero(UInt64))
    err = ccall((:cuMemAllocAsync, CUDA.libcuda), UInt32,
                (Ptr{UInt64}, Csize_t, CUDA.CUstream), p, size, stream)
    err == 0 || return Cint(1)
    unsafe_store!(Ptr{UInt64}(ptr), p[])
    return Cint(0)
end

function pool_free(ctx::Ptr{Cvoid}, ptr::Ptr{Cvoid}, size::Csize_t, stream::CUDA.CUstream)::Cint
    err = ccall((:cuMemFreeAsync, CUDA.libcuda), UInt32,
                (UInt64, CUDA.CUstream), UInt64(UInt(ptr)), stream)
    return err == 0 ? Cint(0) : Cint(1)
end

alloc_fptr = @cfunction(pool_alloc, Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, CUDA.CUstream))
free_fptr  = @cfunction(pool_free, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, CUDA.CUstream))

name = ntuple(64) do i
    s = "cuda_stream_pool"
    i <= ncodeunits(s) ? Cchar(codeunit(s, i)) : Cchar(0)
end
handler = CUDSS.cudssDeviceMemHandler_t(C_NULL, alloc_fptr, free_fptr, name)

# ── Try setting the mempool handler ──
println("\n=== Setting CUDSS mempool handler ===")
mempool_ok = try
    CUDSS.cudssSetDeviceMemHandler(solver.data.handle, Ref(handler))
    println("  ✓ mempool handler set")
    true
catch e
    println("  ✗ failed: ", e)
    false
end

# ── Test 1: Normal factorize + solve ──
println("\n=== Test 1: Normal solve ===")
CUDSS.cudss("factorization", solver, x_gpu, b_gpu)
CUDSS.cudss("solve", solver, x_gpu, b_gpu)
CUDA.device_synchronize()
x_result = Array(x_gpu)
println("  x = ", x_result)
@assert isapprox(x_result, [4 1; 1 3] \ [1.0, 2.0]; atol=1e-10) "Wrong answer"
println("  ✓ correct")

# ── Test 2: Graph capture of factorize + solve ──
if !mempool_ok
    println("\n=== Test 2: SKIPPED (mempool not supported) ===")
else
    println("\n=== Test 2: Graph capture ===")

    # New RHS
    b2_gpu = CuVector([2.0, 1.0])
    x2_gpu = CUDA.zeros(Float64, n)

    # Warm up (JIT)
    CUDSS.cudss("refactorization", solver, x2_gpu, b2_gpu)
    CUDSS.cudss("solve", solver, x2_gpu, b2_gpu)
    CUDA.device_synchronize()

    # Capture
    println("  Capturing...")
    graph = CUDA.capture(flags=CUDA.STREAM_CAPTURE_MODE_THREAD_LOCAL, throw_error=false) do
        CUDSS.cudss("refactorization", solver, x2_gpu, b2_gpu)
        CUDSS.cudss("solve", solver, x2_gpu, b2_gpu)
    end

    if graph === nothing
        println("  ✗ capture failed")
    else
        println("  ✓ captured")
        exec = CUDA.instantiate(graph)
        println("  ✓ instantiated")

        # Reset and replay
        fill!(x2_gpu, 0.0)
        CUDA.launch(exec)
        CUDA.device_synchronize()
        x2_result = Array(x2_gpu)
        println("  x = ", x2_result)
        @assert isapprox(x2_result, [4 1; 1 3] \ [2.0, 1.0]; atol=1e-10) "Wrong answer from graph replay"
        println("  ✓ graph replay correct")
    end
end

# ── Test 3: Batch mode mempool (set BEFORE ubatch_size) ──
println("\n=== Test 3: Batch mode (nbatch=4) ===")
nbatch = 4
nnz_csc = length(nonzeros(A_cpu))
nzvals_batch = repeat(nonzeros(A_cpu), 1, nbatch) |> CuMatrix
A_batch = CUDA.CUSPARSE.CuSparseMatrixCSR(
    A_gpu.rowPtr, A_gpu.colVal, vec(nzvals_batch), size(A_gpu),
)

# Set mempool on the global handle BEFORE creating the batch solver
batch_mempool_ok = try
    CUDSS.cudssSetDeviceMemHandler(CUDSS.handle(), Ref(handler))
    println("  ✓ mempool set on global handle (before batch solver)")
    true
catch e
    println("  ✗ mempool failed on global handle: ", e)
    false
end

solver_batch = CUDSS.CudssSolver(A_batch, "S", 'L')
CUDSS.cudss_set(solver_batch, "ubatch_size", nbatch)

x_batch = CUDSS.CudssMatrix(Float64, n; nbatch=nbatch)
b_batch = CUDSS.CudssMatrix(CuMatrix(repeat([1.0, 2.0], 1, nbatch)))
CUDSS.cudss("analysis", solver_batch, x_batch, b_batch)

CUDSS.cudss("factorization", solver_batch, x_batch, b_batch)
CUDSS.cudss("solve", solver_batch, x_batch, b_batch)
CUDA.device_synchronize()
println("  batch solve OK")

if batch_mempool_ok
    # Warm up
    CUDSS.cudss("refactorization", solver_batch, x_batch, b_batch)
    CUDSS.cudss("solve", solver_batch, x_batch, b_batch)
    CUDA.device_synchronize()

    println("  Capturing batch graph...")
    graph = CUDA.capture(flags=CUDA.STREAM_CAPTURE_MODE_THREAD_LOCAL, throw_error=false) do
        CUDSS.cudss("refactorization", solver_batch, x_batch, b_batch)
        CUDSS.cudss("solve", solver_batch, x_batch, b_batch)
    end
    if graph !== nothing
        exec = CUDA.instantiate(graph)
        CUDA.launch(exec)
        CUDA.device_synchronize()
        println("  ✓ batch graph capture + replay OK")
    else
        println("  ✗ batch graph capture failed")
    end
else
    println("  SKIPPED graph capture (batch mempool not supported)")
end

println("\n=== Done ===")
