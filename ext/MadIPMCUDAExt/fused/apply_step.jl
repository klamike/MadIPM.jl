# Fused `apply_step!` (src/batch/madipm/solver.jl): two kernels, one thread
# per row,
#
#   _apply_step_kernel:   x += alpha_p dx, y += alpha_d dy,
#                         zl_r += alpha_d dzl, zu_r += alpha_d dzu
#   _adjust_bounds_kernel: _adjust_boundary_active! (reads the updated x)

function _apply_step_kernel(iterate, step, alpha_p, alpha_d)
    i, j = _row_and_instance()
    @inbounds begin
        ap = alpha_p[1, j]
        ad = alpha_d[1, j]
        i <= size(iterate.x, 1) && (iterate.x[i, j] += ap * step.x[i, j])
        i <= size(iterate.y, 1) && (iterate.y[i, j] += ad * step.y[i, j])
        i <= size(iterate.zl, 1) && (iterate.zl[i, j] += ad * step.zl[i, j])
        i <= size(iterate.zu, 1) && (iterate.zu[i, j] += ad * step.zu[i, j])
    end
    return nothing
end

# move a bound away from x when x is closer to it than eps * mu
function _adjust_bounds_kernel(lb, ub, mu, mask)
    i, j = _row_and_instance()
    T = eltype(lb.xl)
    c1 = eps(T)
    c2 = eps(T)^(T(3) / T(4))
    @inbounds if mask[1, j] != 0
        m = mu[1, j]
        if i <= size(lb.x, 1)
            x = lb.x[i, j]
            xl = lb.xl[i, j]
            x - xl < c1 * m && (lb.xl[i, j] = xl - c2 * max(one(T), abs(x)))
        end
        if i <= size(ub.x, 1)
            x = ub.x[i, j]
            xu = ub.xu[i, j]
            xu - x < c1 * m && (ub.xu[i, j] = xu + c2 * max(one(T), abs(x)))
        end
    end
    return nothing
end

function MadIPM.apply_step!(s::MadIPM.UniformBatchMPCSolver{T, MT}) where {T, MT <: CuMatrix{T}}
    ws = s.workspace
    d = s.d
    iterate = (; x = MadNLP.full(s.x), y = MadNLP.full(s.y), zl = MadIPM.lower(s.zl), zu = MadIPM.upper(s.zu))
    step = (; x = MadNLP.primal(d), y = MadNLP.dual(d), zl = MadNLP.dual_lb(d), zu = MadNLP.dual_ub(d))
    nrows = max(size(iterate.x, 1), size(iterate.y, 1), size(iterate.zl, 1), size(iterate.zu, 1))
    _launch_per_row(_apply_step_kernel, nrows, s.batch_size, iterate, step, ws.alpha_p, ws.alpha_d)

    lb = (; x = MadIPM.lower(s.x), xl = MadIPM.lower(s.xl))
    ub = (; x = MadIPM.upper(s.x), xu = MadIPM.upper(s.xu))
    nbound_rows = max(size(lb.x, 1), size(ub.x, 1), 1)
    _launch_per_row(_adjust_bounds_kernel, nbound_rows, s.batch_size, lb, ub, ws.mu_batch, ws.active_mask)
    MadIPM.increment_k!(s)
    return
end
