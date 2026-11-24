# Fused kernels for the uniform batch MPC iteration on CUDA.
#
# Each file of this directory replaces one step of the generic batch code
# (src/batch/madipm) by one or two kernel launches, instead of a sequence of
# broadcasts, `batch_mapreduce!` calls and small kernels. Most kernels run one
# block per instance (column j = blockIdx().x): each thread accumulates a
# strided share of the rows, then the block reduces the partial results (as in
# MadNLP#736). A fused method falls back to the generic method only where the
# fused path does not apply (see `@fallback`).
#
# This file holds the machinery shared by the fused kernels.

"""
    @fallback cond f(a::A, b::B, ...)

Return `invoke(f, Tuple{A, B, ...}, a, b, ...)` (the generic method) from the
enclosing method unless `cond` holds. An argument without a type is `Any`.
"""
macro fallback(cond, call)
    Meta.isexpr(call, :call) || error("@fallback: expected a call, got $call")
    f, args = call.args[1], call.args[2:end]
    names = [Meta.isexpr(a, :(::)) ? a.args[1] : a for a in args]
    types = [Meta.isexpr(a, :(::)) ? a.args[2] : :Any for a in args]
    return esc(:($cond || return invoke($f, Tuple{$(types...)}, $(names...))))
end

"""
    @block_stride for i in 1:n
        body
    end

Run `body` for the rows `i = threadIdx().x, threadIdx().x + blockDim().x, ...`
up to `n`: the share of the rows of one thread in a one-block-per-instance
kernel. The rows of each thread are visited in increasing order.
"""
macro block_stride(loop)
    Meta.isexpr(loop, :for) || error("@block_stride: expected a for loop")
    iter, body = loop.args
    i, range = iter.args
    Meta.isexpr(range, :call) && range.args[1] === :(:) && range.args[2] == 1 ||
        error("@block_stride: expected `for i in 1:n`")
    n = range.args[3]
    return esc(quote
        $i = threadIdx().x
        while $i <= $n
            $body
            $i += blockDim().x
        end
    end)
end

# block reductions of one value per thread; the result is valid in thread 1
@inline _block_sum(v) = CUDACore.reduce_block(+, v, zero(v), Val(true))
@inline _block_max(v) = CUDACore.reduce_block(max, v, zero(v), Val(true))  # of values >= 0
@inline _block_min(v) = CUDACore.reduce_block(min, v, oftype(v, Inf), Val(true))

# row i and instance j of a kernel with one thread per row (blocks of rows
# along x, one instance per block along y)
@inline _row_and_instance() = ((blockIdx().x - 1) * blockDim().x + threadIdx().x, blockIdx().y)

# threads of the one-block-per-instance kernels: enough to cover the longest
# vector of the problem, up to 1024
_fused_threads(s) = min(1024, max(64, 32 * cld(s.kkt.n_tot + s.kkt.m, 32 * 8)))

# launch a one-block-per-instance kernel with at most the threads its
# registers allow
function _launch_per_instance(f, s, args...)
    k = @cuda launch = false f(args...)
    threads = min(_fused_threads(s), CUDA.maxthreads(k))
    threads = max(32, 32 * (threads ÷ 32))
    k(args...; threads, blocks = s.batch_size)
    return
end

# launch a one-thread-per-row kernel over `nrows` rows and `batch_size` instances
function _launch_per_row(f, nrows, batch_size, args...)
    @cuda threads = 256 blocks = (cld(nrows, 256), batch_size) f(args...)
    return
end

# The bounded rows and their steps, with the names of the generic code
# (x_lr = lower(x), xl_r = lower(xl), zl_r = lower(zl), dx_lr = xp_lr(d), ...).
# The kernels take these NamedTuples of views instead of one argument per view.
_lower_rows(s) = (; x = MadIPM.lower(s.x), xl = MadIPM.lower(s.xl), zl = MadIPM.lower(s.zl))
_upper_rows(s) = (; x = MadIPM.upper(s.x), xu = MadIPM.upper(s.xu), zu = MadIPM.upper(s.zu))
_lower_rows(s, d) = (; _lower_rows(s)..., dx = MadIPM.xp_lr(d), dzl = MadNLP.dual_lb(d))
_upper_rows(s, d) = (; _upper_rows(s)..., dx = MadIPM.xp_ur(d), dzu = MadNLP.dual_ub(d))

# The fraction-to-the-boundary steps of `get_fraction_to_boundary_step!` with
# tau = 1 (both callers fill ws.tau with 1, and the factor tau is exact), for
# instance j, block-reduced: (alpha_xl, alpha_zl, alpha_xu, alpha_zu), valid
# in thread 1. An empty bound set gives Inf here; the caller sets it to 1.
@inline function _block_fraction_to_boundary(lb, ub, j)
    T = eltype(lb.x)
    alpha_xl = alpha_zl = alpha_xu = alpha_zu = T(Inf)
    @inbounds begin
        @block_stride for i in 1:size(lb.x, 1)
            dx = lb.dx[i, j]
            dx < zero(T) && (alpha_xl = min(alpha_xl, (-lb.x[i, j] + lb.xl[i, j]) / dx))
            dz = lb.dzl[i, j]
            dz < zero(T) && (alpha_zl = min(alpha_zl, -lb.zl[i, j] / dz))
        end
        @block_stride for i in 1:size(ub.x, 1)
            dx = ub.dx[i, j]
            dx > zero(T) && (alpha_xu = min(alpha_xu, (-ub.x[i, j] + ub.xu[i, j]) / dx))
            dz = ub.dzu[i, j]
            zu = ub.zu[i, j]
            (dz < zero(T) && zu + dz < zero(T)) && (alpha_zu = min(alpha_zu, -zu / dz))
        end
    end
    alpha_xl = _block_min(alpha_xl)
    alpha_zl = _block_min(alpha_zl)
    alpha_xu = _block_min(alpha_xu)
    alpha_zu = _block_min(alpha_zu)
    return alpha_xl, alpha_zl, alpha_xu, alpha_zu
end
