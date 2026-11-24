# Fused first part of `update_step!(::MehrotraAdaptiveStep, ...)`
# (src/batch/madipm/kernels.jl): one kernel, one block per instance, does
#
#   fill!(ws.tau, 1); get_fraction_to_boundary_step!   phase 1
#   get_affine_complementarity_measure!                phase 2 -> sum_lb, sum_ub, mu_affine
#   mu_full = mu_affine / gamma_a                      -> mu_curr
#
# then Mehrotra's step kernel (`_mehrotra_step!`) follows as in the generic
# method. Phase 1 is the same as in the fused prediction step, without
# zero_inactive_step!.

function _step_measure_kernel(lb, ub, ws, gamma_a::T) where {T}
    j = blockIdx().x
    nlb = size(lb.x, 1)
    nub = size(ub.x, 1)
    shared_alpha = CuStaticSharedArray(T, 2)
    @inbounds begin
        # phase 1: fraction to the boundary
        alpha_xl, alpha_zl, alpha_xu, alpha_zu = _block_fraction_to_boundary(lb, ub, j)
        if threadIdx().x == 1
            ws.tau[1, j] = one(T)
            nlb == 0 && (alpha_xl = alpha_zl = one(T))
            nub == 0 && (alpha_xu = alpha_zu = one(T))
            ws.alpha_xl[1, j] = alpha_xl
            ws.alpha_zl[1, j] = alpha_zl
            ws.alpha_xu[1, j] = alpha_xu
            ws.alpha_zu[1, j] = alpha_zu
            alpha_p = min(alpha_xl, alpha_xu, one(T))
            alpha_d = min(alpha_zl, alpha_zu, one(T))
            ws.alpha_p[1, j] = alpha_p
            ws.alpha_d[1, j] = alpha_d
            shared_alpha[1] = alpha_p
            shared_alpha[2] = alpha_d
        end
        sync_threads()
        alpha_p = shared_alpha[1]
        alpha_d = shared_alpha[2]

        # phase 2: affine complementarity at (alpha_p, alpha_d)
        sum_lb = zero(T)
        @block_stride for i in 1:nlb
            sum_lb += (lb.x[i, j] + alpha_p * lb.dx[i, j] - lb.xl[i, j]) * (lb.zl[i, j] + alpha_d * lb.dzl[i, j])
        end
        sum_ub = zero(T)
        @block_stride for i in 1:nub
            sum_ub += (ub.xu[i, j] - (ub.x[i, j] + alpha_p * ub.dx[i, j])) * (ub.zu[i, j] + alpha_d * ub.dzu[i, j])
        end
        sum_lb = _block_sum(sum_lb)
        sum_ub = _block_sum(sum_ub)
        if threadIdx().x == 1
            ws.sum_lb[1, j] = sum_lb
            ws.sum_ub[1, j] = sum_ub
            mu_affine = nlb + nub == 0 ? zero(T) : (sum_lb + sum_ub) / (nlb + nub)
            ws.mu_affine[1, j] = mu_affine
            ws.mu_full[1, j] = mu_affine / gamma_a
        end
    end
    return nothing
end

function MadIPM.update_step!(
    rule::MadIPM.MehrotraAdaptiveStep,
    s::MadIPM.UniformBatchMPCSolver{T, MT},
) where {T, MT <: CuMatrix{T}}
    ws = s.workspace
    d = s.d
    gamma_f = T(rule.gamma_f)
    gamma_a = one(T) / (one(T) - gamma_f)
    lb = _lower_rows(s, d)
    ub = _upper_rows(s, d)
    outputs = (; ws.tau, ws.alpha_xl, ws.alpha_zl, ws.alpha_xu, ws.alpha_zu, ws.alpha_p, ws.alpha_d,
                 ws.sum_lb, ws.sum_ub, ws.mu_affine, mu_full = ws.mu_curr)
    _launch_per_instance(_step_measure_kernel, s, lb, ub, outputs, gamma_a)
    MadIPM._mehrotra_step!(
        ws.alpha_p, ws.alpha_d, ws.mu_curr, gamma_f,
        lb.dx, lb.x, lb.xl, d.nlb, lb.dzl, lb.zl,
        ub.dx, ub.x, ub.xu, d.nub, ub.dzu, ub.zu,
        d.values, d.ind_lb, d.ind_ub, d.n + d.m, d.n + d.m + d.nlb)
    return
end
