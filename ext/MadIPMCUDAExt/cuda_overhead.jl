## try to eliminate some of CUDA.jl's overhead while solving
## we can do this because we assume that:
# - within a solve, we are not capturing a cuda graph. users cannot include batched MadIPM in captured code.
# - within a solve, device_reset! is never called, i.e. the original context always remains valid.
const _SOLVING = Ref{Bool}(false)

function __init__()
    # CUDA.jl/lib/cudadrv/graph.jl:162
    @inline CUDA.is_capturing(stream::CUDA.CuStream) =
        _SOLVING[] ? false : (CUDA.capture_status(stream).status != CUDA.STREAM_CAPTURE_STATUS_NONE)

    # CUDA.jl/lib/cudadrv/context.jl:70-87
    @inline CUDA.is_capturing(stream::CUDA.CuStream) =
        _SOLVING[] ? false : (CUDA.capture_status(stream).status != CUDA.STREAM_CAPTURE_STATUS_NONE)

    @inline CUDA.isvalid(ctx::CUDA.CuContext) = 
        _SOLVING[] ? true : _isvalid(ctx)

    function _isvalid(ctx::CUDA.CuContext)
        # we first try an API call to see if the context handle is usable
        if driver_version() >= v"12"
            id_ref = Ref{Culonglong}()
            res = unchecked_cuCtxGetId(ctx, id_ref)
            res == ERROR_CONTEXT_IS_DESTROYED && return false
            res != SUCCESS && throw_api_error(res)

            # detect handle reuse, which happens when destroying and re-creating a context, by
            # looking at the context's unique ID (which does change on re-creation)
            return ctx.id == id_ref[]
        else
            version_ref = Ref{Cuint}()
            res = unchecked_cuCtxGetApiVersion(ctx, version_ref)
            res == ERROR_INVALID_CONTEXT && return false

            # we can't detect handle reuse, so we just assume the context is valid
            return true
        end
    end

    function MadIPM.solve!(batch_solver::MadIPM.UniformBatchMPCSolver{T, <:CuMatrix{T}}) where T
        _SOLVING[] = true
        try
            invoke(MadIPM.solve!, Tuple{MadIPM.AbstractBatchMPCSolver{T}}, batch_solver)
        finally
            _SOLVING[] = false
        end
    end
end