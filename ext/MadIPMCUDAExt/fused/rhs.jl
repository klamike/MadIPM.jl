# Fused `set_predictive_rhs!` and `set_correction_rhs!`
# (src/batch/madipm/kernels.jl): the four blocks of the right-hand side p in
# one kernel, one thread per row,
#
#   px  .= -f .+ zl .- zu .- jacl
#   py  .= -c
#   pzl .= (xl_r .- x_lr) .* zl_r [.+ mu .- correction_lb]
#   pzu .= (xu_r .- x_ur) .* zu_r [.- mu .- correction_ub]
#
# where the bracketed terms are those of the corrector. Every entry of p is
# written, so the predictor needs no fill!.

function _rhs_kernel(p, primal_dual, lb, ub, mu, corr, correction::Bool)
    i, j = _row_and_instance()
    @inbounds begin
        if i <= size(p.x, 1)
            p.x[i, j] = -primal_dual.f[i, j] + primal_dual.zl[i, j] - primal_dual.zu[i, j] - primal_dual.jacl[i, j]
        end
        if i <= size(p.y, 1)
            p.y[i, j] = -primal_dual.c[i, j]
        end
        if i <= size(p.zl, 1)
            v = (lb.xl[i, j] - lb.x[i, j]) * lb.zl[i, j]
            correction && (v = v + mu[1, j] - corr.lb[i, j])
            p.zl[i, j] = v
        end
        if i <= size(p.zu, 1)
            v = (ub.xu[i, j] - ub.x[i, j]) * ub.zu[i, j]
            correction && (v = v - mu[1, j] - corr.ub[i, j])
            p.zu[i, j] = v
        end
    end
    return nothing
end

function _fused_rhs!(s, correction::Bool)
    full = MadNLP.full
    p = (; x = MadNLP.primal(s.p), y = MadNLP.dual(s.p), zl = MadNLP.dual_lb(s.p), zu = MadNLP.dual_ub(s.p))
    primal_dual = (; f = MadNLP.primal(s.f), zl = full(s.zl), zu = full(s.zu), jacl = full(s.jacl), c = full(s.c))
    corrections = (; lb = full(s.correction_lb), ub = full(s.correction_ub))
    nrows = max(size(p.x, 1), size(p.y, 1), size(p.zl, 1), size(p.zu, 1))
    _launch_per_row(_rhs_kernel, nrows, s.batch_size,
        p, primal_dual, _lower_rows(s), _upper_rows(s), s.workspace.mu_batch, corrections, correction)
    return
end

function MadIPM.set_predictive_rhs!(
    s::MadIPM.UniformBatchMPCSolver{T, MT},
    kkt::MadIPM.AbstractBatchKKTSystem,
) where {T, MT <: CuMatrix{T}}
    _fused_rhs!(s, false)
end

# fused only for the corrector of the MPC iteration (mu_batch and the
# corrections of the solver); any other call takes the generic method
function MadIPM.set_correction_rhs!(
    s::MadIPM.UniformBatchMPCSolver{T, MT},
    kkt::MadIPM.AbstractBatchKKTSystem,
    mu, correction_lb, correction_ub, ind_lb, ind_ub,
) where {T, MT <: CuMatrix{T}}
    is_corrector = mu === s.workspace.mu_batch &&
        correction_lb === MadNLP.full(s.correction_lb) && correction_ub === MadNLP.full(s.correction_ub)
    @fallback is_corrector MadIPM.set_correction_rhs!(
        s::MadIPM.AbstractBatchMPCSolver, kkt::MadIPM.AbstractBatchKKTSystem,
        mu, correction_lb, correction_ub, ind_lb, ind_ub)
    _fused_rhs!(s, true)
end
