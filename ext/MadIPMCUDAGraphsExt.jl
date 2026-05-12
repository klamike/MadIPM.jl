module MadIPMCUDAGraphsExt

using CUDA
using CUDAGraphs
using CUDSS
using MadNLPGPU
import MadIPM
import MadNLP

const _SEG_CACHE = Ref{CUDAGraphs.SegmentedGraphCache}()
const _SEG_CACHE_NA = Ref(-1)

function __init__()
    _SEG_CACHE[] = CUDAGraphs.SegmentedGraphCache()
    _SEG_CACHE_NA[] = -1
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
) where {T,MT<:CuMatrix{T},VT}
    if isassigned(_SEG_CACHE)
        CUDAGraphs.invalidate!(_SEG_CACHE[])
        _SEG_CACHE_NA[] = -1
    end
    return invoke(
        MadIPM.solve!,
        Tuple{MadIPM.AbstractBatchMPCSolver{T,MT,VT}},
        batch_solver,
    )
end

CUDAGraphs.@graphbreak function MadIPM.factorize_active!(
    s::MadNLPGPU.CUDSSSolver,
    active::MadIPM.BatchView,
)
    na = MadIPM.local_batch_size(active)
    CUDSS.cudss_set(s.inner, "ubatch_size", na)
    MadNLP.factorize!(s)
    return
end

CUDAGraphs.@graphbreak function MadIPM.solve_active!(
    s::MadNLPGPU.CUDSSSolver{T},
    rhs::CuMatrix{T},
    active::MadIPM.BatchView,
) where T
    na = MadIPM.local_batch_size(active)
    n = size(rhs, 1)
    rhs_active = unsafe_wrap(CuArray{T,2}, pointer(rhs), (n, na))
    CUDSS.cudss_update(s.b_gpu, rhs_active)
    CUDSS.cudss_update(s.x_gpu, rhs_active)
    CUDSS.cudss("solve", s.inner, s.x_gpu, s.b_gpu; asynchronous = s.opt.cudss_asynchronous)
    return
end

CUDAGraphs.@graphbreak function _factorize_system!(batch_solver)
    MadIPM.factorize_system!(batch_solver)
    return
end

CUDAGraphs.@graphbreak function _prediction_step!(batch_solver)
    MadIPM.prediction_step!(batch_solver)
    return
end

CUDAGraphs.@graphbreak function _mehrotra_correction_direction!(batch_solver)
    MadIPM.mehrotra_correction_direction!(batch_solver)
    return
end

CUDAGraphs.@graphbreak function _evaluate_model!(batch_solver)
    MadIPM.evaluate_model!(batch_solver)
    return
end

function MadIPM.mpc_step!(
    batch_solver::MadIPM.UniformBatchMPCSolver{T,MT,VT},
) where {T,MT<:CuMatrix{T},VT}
    cache = _SEG_CACHE[]
    na = MadIPM.active_batch_size(batch_solver)
    if _SEG_CACHE_NA[] != na
        CUDAGraphs.invalidate!(cache)
        _SEG_CACHE_NA[] = na
    end
    CUDAGraphs.@unsafe_scaptured cache begin
        fill!(batch_solver.workspace._ls_error, zero(Int32))
        _factorize_system!(batch_solver)
        _prediction_step!(batch_solver)
        _mehrotra_correction_direction!(batch_solver)
        MadIPM.update_step!(batch_solver.opt.step_rule, batch_solver)
        MadIPM.zero_inactive_step!(batch_solver)
        MadIPM.apply_step!(batch_solver)
        _evaluate_model!(batch_solver)
    end
    return
end

end
