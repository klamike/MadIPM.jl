## try to eliminate some of CUDA.jl's overhead while solving
## we can do this because we assume that:
# - within a solve, device_reset! is never called, i.e. the original context always remains valid.
# - the only graph capture that happens during a solve is our own _captured_step!.

# _SOLVING: skip is_capturing/isvalid overhead (safe outside of graph capture)
# _CAPTURING: temporarily re-enable is_capturing during our own CUDA.capture block
const _SOLVING = Ref{Bool}(false)
const _CAPTURING = Ref{Bool}(false)

# Whether CUDSS mempool is set (required for graph capture)
const _GRAPH_CAPTURE_OK = Ref{Bool}(false)

# Graph cache for segmented CUDA graph capture (via CUDAGraphs.jl)
const _SEG_CACHE = Ref{CUDAGraphs.SegmentedGraphCache}()
const _SEG_CACHE_NA = Ref{Int}(-1)  # active batch size when captured

function MadIPM.solve!(batch_solver::MadIPM.UniformBatchMPCSolver{T, <:CuMatrix{T}}) where T
    _SOLVING[] = true
    # Invalidate graph cache at start of each solve (buffers may have moved)
    if isassigned(_SEG_CACHE)
        CUDAGraphs.invalidate!(_SEG_CACHE[])
        _SEG_CACHE_NA[] = -1
    end
    try
        invoke(MadIPM.solve!, Tuple{MadIPM.AbstractBatchMPCSolver{T}}, batch_solver)
    finally
        _SOLVING[] = false
    end
end

function _init_overhead!()
    _SEG_CACHE[] = CUDAGraphs.SegmentedGraphCache()
    _SEG_CACHE_NA[] = -1

    @eval begin
        # CUDA.jl/lib/cudadrv/graph.jl:162
        @inline CUDA.is_capturing(stream::CUDA.CuStream) =
            (_SOLVING[] && !_CAPTURING[]) ? false : (CUDA.capture_status(stream).status != CUDA.STREAM_CAPTURE_STATUS_NONE)

        # CUDA.jl/lib/cudadrv/context.jl:70-87
        @inline CUDA.isvalid(ctx::CUDA.CuContext) =
            _SOLVING[] ? true : _isvalid(ctx)
    end
end

function _isvalid(ctx::CUDA.CuContext)
    # we first try an API call to see if the context handle is usable
    if CUDA.driver_version() >= v"12"
        id_ref = Ref{CUDA.Culonglong}()
        res = CUDA.unchecked_cuCtxGetId(ctx, id_ref)
        res == CUDA.ERROR_CONTEXT_IS_DESTROYED && return false
        res != CUDA.SUCCESS && CUDA.throw_api_error(res)

        # detect handle reuse, which happens when destroying and re-creating a context, by
        # looking at the context's unique ID (which does change on re-creation)
        return ctx.id == id_ref[]
    else
        version_ref = Ref{CUDA.Cuint}()
        res = CUDA.unchecked_cuCtxGetApiVersion(ctx, version_ref)
        res == CUDA.ERROR_INVALID_CONTEXT && return false

        # we can't detect handle reuse, so we just assume the context is valid
        return true
    end
end
