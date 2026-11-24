# Fused setup of `factorize_system!` (src/batch/madipm/solver.jl) for
# FixedRegularization with cuDSS: one kernel, one block per instance, does
#
#   update_regularization!(::FixedRegularization)   del_w, del_c
#   set_aug_diagonal_reg! (masked variant)          reg, pr_diag, du_diag,
#                                                   l_diag, u_diag, l_lower, u_lower,
#                                                   pr_diag[ind_lb/ub] -= lower / diag
#
# for the active instances (mask == 1), then factorizes. cuDSS always reports
# success (is_factorized), so the generic retry loop is not needed.
#
# The bound terms are subtracted from the primal diagonal after it is reset,
# hence a block barrier between the two phases; the indices are unique within
# each bound set, but a variable can have both bounds, so the two sets are
# applied one after the other (another barrier).

function _factorize_setup_kernel(kkt, lb, ub, mask, reg, delta_p::T, delta_d::T, du_off::Int) where {T}
    j = blockIdx().x
    @inbounds if mask[1, j] == one(T)
        # update_regularization!
        if threadIdx().x == 1
            reg.del_w[1, j] = delta_p
            reg.del_c[1, j] = delta_d
        end
        # set_aug_diagonal_reg!: kkt.reg, pr_diag and du_diag
        @block_stride for i in 1:kkt.n_tot
            kkt.reg[i, j] = delta_p
            kkt.nzVals[i, j] = delta_p
        end
        @block_stride for i in 1:kkt.m
            kkt.nzVals[du_off + i, j] = delta_d
        end
        # l_diag, l_lower, u_diag, u_lower
        @block_stride for i in 1:size(kkt.l_diag, 1)
            kkt.l_diag[i, j] = lb.xl[i, j] - lb.x[i, j]
            kkt.l_lower[i, j] = lb.zl[i, j]
        end
        @block_stride for i in 1:size(kkt.u_diag, 1)
            kkt.u_diag[i, j] = ub.x[i, j] - ub.xu[i, j]
            kkt.u_lower[i, j] = ub.zu[i, j]
        end
        sync_threads()
        # pr_diag_lb .-= l_lower ./ l_diag, then pr_diag_ub .-= u_lower ./ u_diag
        @block_stride for k in 1:size(kkt.l_diag, 1)
            r = kkt.ind_lb[k]
            kkt.nzVals[r, j] -= kkt.l_lower[k, j] / kkt.l_diag[k, j]
        end
        sync_threads()
        @block_stride for k in 1:size(kkt.u_diag, 1)
            r = kkt.ind_ub[k]
            kkt.nzVals[r, j] -= kkt.u_lower[k, j] / kkt.u_diag[k, j]
        end
    end
    return nothing
end

function MadIPM.factorize_system!(s::MadIPM.UniformBatchMPCSolver{T, MT}) where {T, MT <: CuMatrix{T}}
    reg = s.opt.regularization
    kkt = s.kkt
    @fallback (reg isa MadIPM.FixedRegularization && kkt.batch_solver isa CUDSSSolver) MadIPM.factorize_system!(
        s::MadIPM.AbstractBatchMPCSolver)
    kkt_diag = (;
        kkt.nzVals, kkt.reg, kkt.l_diag, kkt.u_diag, kkt.l_lower, kkt.u_lower,
        ind_lb = MadIPM._get_ind_lb(s), ind_ub = MadIPM._get_ind_ub(s), kkt.n_tot, kkt.m,
    )
    du_off = size(kkt.nzVals, 1) - kkt.m   # du_diag(kkt) is the last m rows of nzVals
    _launch_per_instance(_factorize_setup_kernel, s,
        kkt_diag, _lower_rows(s), _upper_rows(s), s.workspace.active_mask, (; s.del_w, s.del_c),
        T(reg.delta_p), T(reg.delta_d), du_off)
    MadNLP.factorize_wrapper!(s)
    return
end
