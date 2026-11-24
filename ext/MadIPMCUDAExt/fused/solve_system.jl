# Fused residual check of `solve_system!` (src/batch/madipm/solver.jl). The
# solve and the residual w = p - K d are as in the generic method; one kernel,
# one block per instance, then does the rest:
#
#   w .*= active_mask; p .*= active_mask
#   norm_w = max |w|; norm_p = max |p|        (batch_maximum!)
#   norm_w /= max(1, norm_p)
#   ls_error |= isnan(norm_w) | (check_residual & (norm_w > tol_linear_solve))

function _residual_kernel(w, p, mask, ws, check_residual::Bool, tol::T) where {T}
    j = blockIdx().x
    @inbounds begin
        m = mask[1, j]
        norm_w = norm_p = zero(T)
        @block_stride for i in 1:size(w, 1)
            w_i = w[i, j] * m
            p_i = p[i, j] * m
            # as the generic path, which leaves |p| in w (reused as a buffer)
            w[i, j] = abs(p_i)
            p[i, j] = p_i
            norm_w = max(norm_w, abs(w_i))
            norm_p = max(norm_p, abs(p_i))
        end
        norm_w = _block_max(norm_w)
        norm_p = _block_max(norm_p)
        if threadIdx().x == 1
            ratio = norm_w / max(one(T), norm_p)
            ws.norm_w[1, j] = ratio
            ws.norm_p[1, j] = norm_p
            ws.ls_error[1, j] |= isnan(ratio) | (check_residual & (ratio > tol))
        end
    end
    return nothing
end

function MadIPM.solve_system!(
    d::MadIPM.BatchUnreducedKKTVector{T},
    s::MadIPM.UniformBatchMPCSolver{T, MT},
    p::MadIPM.BatchUnreducedKKTVector{T},
) where {T, MT <: CuMatrix{T}}
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(s.kkt, s)
    w = s._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, s.kkt, d, -one(T), one(T))
    ws = s.workspace
    outputs = (; norm_w = ws._norm_gpu_w, norm_p = ws._norm_gpu_p, ls_error = ws._ls_error)
    _launch_per_instance(_residual_kernel, s,
        MadNLP.full(w), MadNLP.full(p), ws.active_mask, outputs, s.opt.check_residual, T(s.opt.tol_linear_solve))
    return d
end
