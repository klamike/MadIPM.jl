# Fused bound-row helpers of the batch augmented KKT solve
# (src/batch/KKT/Sparse/augmented.jl), one launch each instead of one per
# bound set:
#
#   _reduce_rhs_batch!: values[ind_lb] -= values[lb rows] / l_diag, then the
#     same for the upper bounds (a variable can have both bounds, so the lower
#     set is applied before the upper set: one block per instance, with a
#     barrier in between)
#   _finish_aug_solve_batch!: recovery of the bound-multiplier steps; the
#     rows are independent (one thread per row)

function _reduce_rhs_kernel(values, lb, ub)
    j = blockIdx().x
    @inbounds begin
        @block_stride for k in 1:length(lb.ind)
            values[lb.ind[k], j] -= values[lb.off + k, j] / lb.diag[k, j]
        end
        sync_threads()
        @block_stride for k in 1:length(ub.ind)
            values[ub.ind[k], j] -= values[ub.off + k, j] / ub.diag[k, j]
        end
    end
    return nothing
end

function _finish_aug_solve_kernel(values, lb, ub)
    i, j = _row_and_instance()
    @inbounds begin
        if i <= length(lb.ind)
            r = lb.off + i
            values[r, j] = (-values[r, j] + lb.lower[i, j] * values[lb.ind[i], j]) / lb.diag[i, j]
        end
        if i <= length(ub.ind)
            r = ub.off + i
            values[r, j] = (values[r, j] - ub.lower[i, j] * values[ub.ind[i], j]) / ub.diag[i, j]
        end
    end
    return nothing
end

function MadIPM._reduce_rhs_batch!(values::CuMatrix, ind_lb, lb_off, l_diag, ind_ub, ub_off, u_diag)
    length(ind_lb) + length(ind_ub) == 0 && return
    lb = (; ind = ind_lb, off = Int(lb_off), diag = l_diag)
    ub = (; ind = ind_ub, off = Int(ub_off), diag = u_diag)
    k = @cuda launch = false _reduce_rhs_kernel(values, lb, ub)
    threads = min(CUDA.maxthreads(k), max(32, 32 * cld(max(length(ind_lb), length(ind_ub)), 32)), 1024)
    k(values, lb, ub; threads, blocks = size(values, 2))
    return
end

function MadIPM._finish_aug_solve_batch!(values::CuMatrix, ind_lb, lb_off, l_lower, l_diag, ind_ub, ub_off, u_lower, u_diag)
    nrows = max(length(ind_lb), length(ind_ub))
    nrows == 0 && return
    lb = (; ind = ind_lb, off = Int(lb_off), lower = l_lower, diag = l_diag)
    ub = (; ind = ind_ub, off = Int(ub_off), lower = u_lower, diag = u_diag)
    _launch_per_row(_finish_aug_solve_kernel, nrows, size(values, 2), values, lb, ub)
    return
end
