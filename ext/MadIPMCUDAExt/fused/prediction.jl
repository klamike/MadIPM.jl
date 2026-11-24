# Fused `prediction_step!` (src/batch/madipm/solver.jl) for the Mehrotra
# barrier update. After `affine_direction!`, one kernel, one block per
# instance, does the rest of the generic method:
#
#   fill!(ws.tau, 1); get_fraction_to_boundary_step!   phase 1
#   zero_inactive_step!                                phase 1 (alpha * mask)
#   get_affine_complementarity_measure!                phase 2 -> mu_affine
#   get_correction!                                    phase 2 -> correction_lb/ub
#   update_barrier!(::Mehrotra, ...)                   phase 2 -> mu_curr, mu_batch
#     (with get_complementarity_measure!)
#
# Phase 2 needs the step lengths of phase 1, which thread 1 shares with the
# block through shared memory.

function _prediction_kernel(lb, ub, mask, ws, corr, mu_min::T) where {T}
    j = blockIdx().x
    nlb = size(lb.x, 1)
    nub = size(ub.x, 1)
    shared_alpha = CuStaticSharedArray(T, 2)
    @inbounds begin
        # phase 1: fraction to the boundary and zero_inactive_step!
        alpha_xl, alpha_zl, alpha_xu, alpha_zu = _block_fraction_to_boundary(lb, ub, j)
        if threadIdx().x == 1
            ws.tau[1, j] = one(T)
            nlb == 0 && (alpha_xl = alpha_zl = one(T))
            nub == 0 && (alpha_xu = alpha_zu = one(T))
            ws.alpha_xl[1, j] = alpha_xl
            ws.alpha_zl[1, j] = alpha_zl
            ws.alpha_xu[1, j] = alpha_xu
            ws.alpha_zu[1, j] = alpha_zu
            m = mask[1, j]
            alpha_p = min(alpha_xl, alpha_xu, one(T)) * m
            alpha_d = min(alpha_zl, alpha_zu, one(T)) * m
            ws.alpha_p[1, j] = alpha_p
            ws.alpha_d[1, j] = alpha_d
            shared_alpha[1] = alpha_p
            shared_alpha[2] = alpha_d
        end
        sync_threads()
        alpha_p = shared_alpha[1]
        alpha_d = shared_alpha[2]

        # phase 2: affine and current complementarity, correction terms
        sum_aff_l = sum_l = zero(T)  # sum (x + alpha_p dx - xl)(zl + alpha_d dzl), sum (x - xl) zl
        @block_stride for i in 1:nlb
            x = lb.x[i, j]
            xl = lb.xl[i, j]
            zl = lb.zl[i, j]
            dx = lb.dx[i, j]
            dzl = lb.dzl[i, j]
            sum_aff_l += (x + alpha_p * dx - xl) * (zl + alpha_d * dzl)
            sum_l += (x - xl) * zl
            corr.lb[i, j] = dx * dzl
        end
        sum_aff_u = sum_u = zero(T)  # sum (xu - (x + alpha_p dx))(zu + alpha_d dzu), sum (xu - x) zu
        @block_stride for i in 1:nub
            x = ub.x[i, j]
            xu = ub.xu[i, j]
            zu = ub.zu[i, j]
            dx = ub.dx[i, j]
            dzu = ub.dzu[i, j]
            sum_aff_u += (xu - (x + alpha_p * dx)) * (zu + alpha_d * dzu)
            sum_u += (xu - x) * zu
            corr.ub[i, j] = dx * dzu
        end
        sum_aff_l = _block_sum(sum_aff_l)
        sum_aff_u = _block_sum(sum_aff_u)
        sum_l = _block_sum(sum_l)
        sum_u = _block_sum(sum_u)

        # get_affine_complementarity_measure!, get_complementarity_measure!, update_barrier!
        if threadIdx().x == 1
            nbounds = nlb + nub
            if nbounds == 0
                ws.mu_affine[1, j] = zero(T)
                ws.mu_curr[1, j] = zero(T)
                ws.mu_batch[1, j] = max(mu_min, zero(T))
            else
                mu_affine = (sum_aff_l + sum_aff_u) / nbounds
                mu_curr = (sum_l + sum_u) / nbounds
                ws.mu_affine[1, j] = mu_affine
                ws.mu_curr[1, j] = mu_curr
                sigma = clamp((mu_affine / mu_curr)^3, T(1e-6), T(10.0))
                ws.mu_batch[1, j] = max(mu_min, sigma * mu_curr)
            end
        end
    end
    return nothing
end

function MadIPM.prediction_step!(s::MadIPM.UniformBatchMPCSolver{T, MT}) where {T, MT <: CuMatrix{T}}
    @fallback s.opt.barrier_update isa MadIPM.Mehrotra MadIPM.prediction_step!(
        s::MadIPM.AbstractBatchMPCSolver)
    ws = s.workspace
    MadIPM.affine_direction!(s)
    outputs = (; ws.tau, ws.alpha_xl, ws.alpha_zl, ws.alpha_xu, ws.alpha_zu, ws.alpha_p, ws.alpha_d,
                 ws.mu_affine, ws.mu_curr, ws.mu_batch)
    corrections = (; lb = MadNLP.full(s.correction_lb), ub = MadNLP.full(s.correction_ub))
    _launch_per_instance(_prediction_kernel, s,
        _lower_rows(s, s.d), _upper_rows(s, s.d), ws.active_mask, outputs, corrections, T(s.opt.mu_min))
    return
end
