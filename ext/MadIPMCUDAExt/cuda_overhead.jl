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

#=
    Segmented graph cache for batch CUDA graph capture.

    When CUDSS batch mode (ubatch_size > 1) is not graphable, mpc_step! is split
    into multiple graph segments around the 3 CUDSS calls (1 factorize + 2 solves).
    Each segment is captured as an independent CUDA graph. During replay, segments
    are launched sequentially with CUDSS calls (stream-ordered) in between.
=#
mutable struct SegmentedGraphCache
    execs::Vector{CUDA.CUgraphExec}    # raw exec handles (one per segment)
    graphs::Vector{CUDA.CUgraph}       # raw graph handles (for cleanup)
    cudss_calls::Vector{Any}           # CUDSS work closures between segments
    n_segments::Int                     # number of captured segments (expect 4)
    na::Int                             # active batch size when captured
    valid::Bool
end

function SegmentedGraphCache()
    SegmentedGraphCache(
        CUDA.CUgraphExec[], CUDA.CUgraph[], Any[], 0, -1, false,
    )
end

const _SEG_CACHE = Ref{SegmentedGraphCache}()

# Segment tracking during capture/replay
const _CURRENT_SEGMENT = Ref{Int}(0)
const _SEGMENTED_MODE = Ref{Symbol}(:off)  # :off, :capturing, :replaying

function _invalidate_seg_cache!()
    isassigned(_SEG_CACHE) || return
    cache = _SEG_CACHE[]
    for exec in cache.execs
        CUDA.cuGraphExecDestroy(exec)
    end
    for graph in cache.graphs
        CUDA.cuGraphDestroy(graph)
    end
    empty!(cache.execs)
    empty!(cache.graphs)
    empty!(cache.cudss_calls)
    cache.n_segments = 0
    cache.na = -1
    cache.valid = false
    return
end

function MadIPM.solve!(batch_solver::MadIPM.UniformBatchMPCSolver{T, <:CuMatrix{T}}) where T
    _SOLVING[] = true
    # Invalidate graph cache at start of each solve (buffers may have moved)
    _invalidate_seg_cache!()
    try
        invoke(MadIPM.solve!, Tuple{MadIPM.AbstractBatchMPCSolver{T}}, batch_solver)
    finally
        _SOLVING[] = false
    end
end

function _init_overhead!()
    _SEG_CACHE[] = SegmentedGraphCache()

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
