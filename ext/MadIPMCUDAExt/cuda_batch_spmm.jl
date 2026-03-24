const _SPMM_W = 4
const _SPMM_BATCH_TILE = 32 ÷ _SPMM_W
const _SPMM_ROWS_PER_BLOCK = 4  # to get (32 * 4) = 128 threads

_batch_mul_kernel!(
    out, A, B, flat_nz, flat_val, rowptr,
    alpha, beta, val_offset::Int32, nout::Int32, bs::Int32,
) = begin
    t = threadIdx().x - Int32(1)       # 0-indexed lane within warp
    batch_local = t % Int32(_SPMM_BATCH_TILE)
    lane = t ÷ Int32(_SPMM_BATCH_TILE)

    j = (blockIdx().x - Int32(1)) * Int32(_SPMM_BATCH_TILE) + batch_local + Int32(1)
    r = (blockIdx().y - Int32(1)) * Int32(_SPMM_ROWS_PER_BLOCK) + threadIdx().y

    (r > nout || j > bs) && return nothing

    # Strided accumulation over nonzeros
    val = zero(eltype(out))
    @inbounds begin
        start = rowptr[r]
        stop  = rowptr[r + Int32(1)] - Int32(1)
        k = start + lane
        while k <= stop
            val = muladd(A[flat_nz[k], j], B[flat_val[k] + val_offset, j], val)
            k += Int32(_SPMM_W)
        end
    end

    # Warp shuffle reduction: sum across W lanes separated by BATCH_TILE positions
    delta = UInt32(_SPMM_BATCH_TILE)
    while delta < UInt32(32)
        val += CUDA.shfl_down_sync(0xffffffff, val, delta)
        delta <<= UInt32(1)
    end

    # Only lane 0 of each group writes the result
    @inbounds if lane == Int32(0)
        out[r, j] = muladd(alpha, val, beta * out[r, j])
    end
    return nothing
end

function MadIPM._batch_mul_impl!(
    out::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
    flat_nz::CuVector{<:Integer}, flat_val::CuVector{<:Integer},
    rowptr::CuVector{<:Integer}, alpha::T, beta::T,
    val_offset::Int32 = Int32(0),
) where T
    nout = length(rowptr) - 1
    bs = size(out, 2)
    if nout > 0
        threads = (32, _SPMM_ROWS_PER_BLOCK)
        blocks  = (cld(bs, _SPMM_BATCH_TILE), cld(nout, _SPMM_ROWS_PER_BLOCK))
        CUDA.@cuda threads=threads blocks=blocks _batch_mul_kernel!(
            out, A, B, flat_nz, flat_val, rowptr,
            alpha, beta, val_offset, Int32(nout), Int32(bs),
        )
    end
    return out
end
