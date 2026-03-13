# atomic mapreduce to avoid allocation of partials
@inline function _atomic_colreduce!(::typeof(max), out, j, val)
    old = out[1, j]
    while val > old
        result = Atomix.@atomicreplace out[1, j] old => val
        old = result.old; result.success && break
    end
end
@inline function _atomic_colreduce!(::typeof(min), out, j, val)
    old = out[1, j]
    while val < old
        result = Atomix.@atomicreplace out[1, j] old => val
        old = result.old; result.success && break
    end
end
@inline _atomic_colreduce!(::typeof(+), out, j, val) = Atomix.@atomic out[1, j] += val

# see CUDA.jl src/mapreduce.jl:83-85
Base.@propagate_inbounds _src_getindex(srcs::Tuple, i, j) =
    (srcs[1][i, j], _src_getindex(Base.tail(srcs), i, j)...)
Base.@propagate_inbounds _src_getindex(srcs::Tuple{Any}, i, j) = (srcs[1][i, j],)
Base.@propagate_inbounds _src_getindex(srcs::Tuple{}, i, j)    = ()

_batch_mapreduce_kernel(f::F, op::OP, neutral::T, out, srcs::NTuple{N}) where {F, OP, T, N} = begin
    bs    = size(out, 2)
    nrows = size(first(srcs), 1)
    blockIdx_reduce, j = fldmod1(blockIdx().x, bs)
    gridDim_reduce = gridDim().x ÷ bs

    @inbounds if j <= bs
        val = neutral
        i = threadIdx().x + (blockIdx_reduce - 1) * blockDim().x
        while i <= nrows
            val = op(val, f(_src_getindex(srcs, i, j)...))
            i += blockDim().x * gridDim_reduce
        end

        val = CUDA.reduce_block(op, val, neutral, Val(true))

        if threadIdx().x == 1
            _atomic_colreduce!(op, out, j, val)
        end
    end
    return
end

function MadIPM.batch_mapreduce!(f, op, neutral::T, out::CuMatrix{T}, srcs::CuMatrix{T}...) where T
    nrows = size(first(srcs), 1)
    nrows == 0 && return
    fill!(out, neutral)
    kernel = @cuda launch=false _batch_mapreduce_kernel(f, op, neutral, out, srcs)
    config = launch_configuration(kernel.fun)
    threads = (config.threads ÷ 32) * 32           # round down to warp multiple
    reduce_blocks = min(cld(nrows, threads), max(1, cld(config.blocks, size(out, 2))))
    kernel(f, op, neutral, out, srcs; threads, blocks = reduce_blocks * size(out, 2))
    return
end

MadIPM.batch_maximum!(out::CuMatrix{T}, src::CuMatrix{T}) where T =
    MadIPM.batch_mapreduce!(identity, max, typemin(T), out, src)
MadIPM.batch_minimum!(out::CuMatrix{T}, src::CuMatrix{T}) where T =
    MadIPM.batch_mapreduce!(identity, min, typemax(T), out, src)
MadIPM.batch_sum!(out::CuMatrix{T}, src::CuMatrix{T}) where T =
    MadIPM.batch_mapreduce!(identity, +, zero(T), out, src)
