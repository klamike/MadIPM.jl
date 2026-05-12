module MadIPMCUDAGraphsExt

using CUDA
using CUDAGraphs
using CUDSS
using MadNLPGPU
import MadIPM
import MadNLP

const _SEG_CACHE = Ref{CUDAGraphs.SegmentedGraphCache}()
const _SEG_CACHE_NA = Ref(-1)
const _SEG_CACHE_ROOTS = Ref(Int32[])

function _active_roots_changed!(active::MadIPM.BatchView)
    na = MadIPM.local_batch_size(active)
    roots = active.local_to_root
    cached = _SEG_CACHE_ROOTS[]
    if length(cached) == na
        same = true
        @inbounds for i in 1:na
            if cached[i] != roots[i]
                same = false
                break
            end
        end
        same && return false
    end
    resize!(cached, na)
    @inbounds for i in 1:na
        cached[i] = roots[i]
    end
    return true
end

function __init__()
    _SEG_CACHE[] = CUDAGraphs.SegmentedGraphCache()
    _SEG_CACHE_NA[] = -1
    empty!(_SEG_CACHE_ROOTS[])
    @eval begin
        @inline CUDA.is_capturing(stream::CUDA.CuStream) =
            CUDAGraphs._in_unsafe_capture() ? true :
            CUDAGraphs._in_unsafe_replay() ? false :
            (CUDA.capture_status(stream).status != CUDA.STREAM_CAPTURE_STATUS_NONE)

        @inline CUDA.isvalid(ctx::CUDA.CuContext) =
            CUDAGraphs._in_unsafe_scaptured() ? true : CUDAGraphs._isvalid_ctx(ctx)
    end
end

function MadIPM.solve!(
    batch_solver::MadIPM.UniformBatchMPCSolver{T,MT,VT},
    ;
    kwargs...,
) where {T,MT<:CuMatrix{T},VT}
    if isassigned(_SEG_CACHE)
        CUDAGraphs.invalidate!(_SEG_CACHE[])
        _SEG_CACHE_NA[] = -1
        empty!(_SEG_CACHE_ROOTS[])
    end
    return invoke(
        MadIPM.solve!,
        Tuple{MadIPM.AbstractBatchMPCSolver{T,MT,VT}},
        batch_solver,
        ;
        kwargs...,
    )
end

CUDAGraphs.@graphbreak function MadIPM.factorize_active!(
    s::MadNLPGPU.CUDSSSolver{T,<:CuVector{T}},
    active::MadIPM.BatchView,
) where {T}
    na = MadIPM.local_batch_size(active)
    CUDSS.cudss_set(s.inner, "ubatch_size", na)
    CUDSS.cudss_set(s.inner, "ubatch_index", -1)
    MadNLP.factorize!(s)
    return
end

CUDAGraphs.@graphbreak function MadIPM.solve_active!(
    s::MadNLPGPU.CUDSSSolver{T,<:CuVector{T}},
    rhs::CuMatrix{T},
    active::MadIPM.BatchView,
) where T
    na = MadIPM.local_batch_size(active)
    CUDSS.cudss_set(s.inner, "ubatch_size", na)
    CUDSS.cudss_set(s.inner, "ubatch_index", -1)
    n = size(rhs, 1)
    rhs_active = unsafe_wrap(CuArray{T,2}, pointer(rhs), (n, na))
    CUDSS.cudss_update(s.b_gpu, rhs_active)
    CUDSS.cudss_update(s.x_gpu, rhs_active)
    CUDSS.cudss("solve", s.inner, s.x_gpu, s.b_gpu; asynchronous = s.opt.cudss_asynchronous)
    return
end

CUDAGraphs.@graphbreak function MadIPM.increment_k!(
    batch_solver::MadIPM.UniformBatchMPCSolver{T,MT,VT},
) where {T,MT<:CuMatrix{T},VT}
    return invoke(
        MadIPM.increment_k!,
        Tuple{MadIPM.AbstractBatchMPCSolver},
        batch_solver,
    )
end

function MadIPM.mpc_step!(
    batch_solver::MadIPM.UniformBatchMPCSolver{T,MT,VT},
) where {T,MT<:CuMatrix{T},VT}
    cache = _SEG_CACHE[]
    active = MadIPM.active_view(batch_solver.batch_views)
    na = MadIPM.local_batch_size(active)
    if _SEG_CACHE_NA[] != na || _active_roots_changed!(active)
        CUDAGraphs.invalidate!(cache)
        _SEG_CACHE_NA[] = na
    end
    CUDAGraphs.@unsafe_scaptured cache begin
        invoke(MadIPM.mpc_step!, Tuple{MadIPM.AbstractBatchMPCSolver}, batch_solver)
    end
    return
end

end
