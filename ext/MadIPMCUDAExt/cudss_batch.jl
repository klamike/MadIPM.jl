# ============================================================================
# CUDSSBatchLinearSolver — native CUDSS uniform-batch solver.
#
# CUDSS accepts a CSC matrix whose `nzVal` is a `(nnz × nbatch)` matrix and
# solves every column's system in one call (`ubatch_size`). We wrap that
# under the same `factorize_active!` / `solve_active!` / `is_factorized!`
# interface that `LoopedBatchLinearSolver` provides, so the batch KKT
# system can pick either transparently.
#
# Active-set handling: since `compact_active_columns_inplace!` packs active
# columns into positions `1:na` of `aug_com_nzvals` (and the rhs), we only
# need to set `ubatch_size = na` before each factor/solve — CUDSS reads the
# first `na` columns and ignores the tail.
# ============================================================================

using CUDSS: CudssSolver, CudssMatrix, cudss, cudss_set, cudss_update

mutable struct CUDSSBatchLinearSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::CudssSolver{T}
    nzvals_mat::CuMatrix{T}            # (nnz × batch_size)
    x_gpu::CudssMatrix{T}
    b_gpu::CudssMatrix{T}
    batch_size::Int
    active_size::Base.RefValue{Int}
    fresh::Base.RefValue{Bool}
end

function CUDSSBatchLinearSolver(
    aug_com::CuSparseMatrixCSC{T, Cint},
    nzvals_mat::CuMatrix{T},
    n::Int;
    cudss_ir::Int = 0,
) where {T}
    batch_size = size(nzvals_mat, 2)
    # LDLᵀ with upper-triangular storage matches MadNLPGPU.CUDSSSolver's default.
    solver = CudssSolver(aug_com.colPtr, aug_com.rowVal, vec(nzvals_mat), "S", 'U')
    cudss_set(solver, "ubatch_size", batch_size)
    # Iterative refinement: CUDSS-LDL on the unreduced augmented system is
    # fragile without IR (some Netlib LPs stall even post-Ruiz). A few IR
    # steps are cheap and broadly stabilizing.
    cudss_ir > 0 && cudss_set(solver, "ir_n_steps", cudss_ir)

    x_gpu = CudssMatrix(T, n; nbatch = batch_size)
    b_gpu = CudssMatrix(T, n; nbatch = batch_size)
    cudss("analysis", solver, x_gpu, b_gpu; asynchronous = true)

    return CUDSSBatchLinearSolver{T}(
        solver, nzvals_mat, x_gpu, b_gpu,
        batch_size, Ref(batch_size), Ref(true),
    )
end

@inline function _set_active!(s::CUDSSBatchLinearSolver, na::Int)
    if s.active_size[] != na
        cudss_set(s.inner, "ubatch_size", na)
        s.active_size[] = na
    end
    return
end

function MadIPM.factorize_active!(s::CUDSSBatchLinearSolver, factor_view::MadIPM.BatchView)
    na = MadIPM.local_batch_size(factor_view)
    na == 0 && return
    _set_active!(s, na)
    cudss_update(s.inner.matrix, vec(s.nzvals_mat))
    phase = s.fresh[] ? "factorization" : "refactorization"
    cudss(phase, s.inner, s.x_gpu, s.b_gpu; asynchronous = true)
    s.fresh[] = false
    return
end

function MadIPM.solve_active!(
    s::CUDSSBatchLinearSolver{T}, rhs::CuMatrix{T}, active::MadIPM.BatchView,
) where {T}
    na = MadIPM.local_batch_size(active)
    na == 0 && return
    _set_active!(s, na)
    cudss_update(s.b_gpu, rhs)
    cudss_update(s.x_gpu, rhs)
    cudss("solve", s.inner, s.x_gpu, s.b_gpu; asynchronous = true)
    return rhs
end

# Per-iteration factorize-failure detection: CUDSS doesn't expose a per-system
# success flag on batched solves, so we assume success. If a single instance
# fails, the batch-wide `info` is non-zero and the next solve will surface it.
MadIPM.is_factorized!(
    ::Vector{Int32}, ::CUDSSBatchLinearSolver, ::MadIPM.BatchView,
) = 0

# ----------------------------------------------------------------------------
# ScaledSparseUniformBatchKKTSystem — GPU kernel for per-position K2.5 scaling
# of the batched COO `nzVals`. Mirrors the CPU loop in
# `src/batch/KKT/Sparse/scaled_augmented.jl`.
# ----------------------------------------------------------------------------

# Mirror scalar MadNLPGPU's `_scale_augmented_system_coo_kernel!`: the primal
# diag / hess / jac / dual-diag branches are dispatched by `k <= n_tot` (not by
# `i == j`) so hess diagonal entries also get the sf² factor.
@kernel function _scale_batch_nzvals_kernel!(scaled, src, @Const(I), @Const(J),
                                              @Const(sf), n_tot::Int, m::Int)
    k, b = @index(Global, NTuple)
    @inbounds begin
        i = Int(I[k]); j = Int(J[k])
        if k <= n_tot
            scaled[k, b] = src[k, b]                                  # primal diag
        elseif i <= n_tot && j <= n_tot
            scaled[k, b] = src[k, b] * sf[i, b] * sf[j, b]            # hess
        elseif n_tot + 1 <= i <= n_tot + m && j <= n_tot
            scaled[k, b] = src[k, b] * sf[j, b]                        # jac
        elseif n_tot + 1 <= i <= n_tot + m && n_tot + 1 <= j <= n_tot + m
            scaled[k, b] = src[k, b]                                  # dual diag
        end
    end
end

function MadIPM._scale_batch_nzvals!(scaled::CuMatrix{T}, src::CuMatrix{T},
                                      I::CuVector, J::CuVector,
                                      sf::CuMatrix{T}, n_tot::Int, m::Int) where {T}
    nnz = length(I); bs = size(scaled, 2)
    (nnz == 0 || bs == 0) && return
    _scale_batch_nzvals_kernel!(CUDABackend())(scaled, src, I, J, sf, n_tot, m;
                                               ndrange = (nnz, bs))
    return
end

# When the batch KKT system runs on GPU and the user selects `CUDSSSolver`,
# return a native CUDSS-batched solver instead of the one-solver-per-column
# loop. The type-specific signature (`CuSparseMatrixCSC` + `CuMatrix`) makes
# this override more specific than the generic `LoopedBatchLinearSolver` in
# src/, so dispatch picks this automatically on GPU.
function MadIPM.LoopedBatchLinearSolver(
    aug_com::CuSparseMatrixCSC{T, Cint},
    nzvals_mat::CuMatrix{T},
    n::Int;
    opt::MadIPM.LoopedBatchLinearSolverOptions = MadIPM.LoopedBatchLinearSolverOptions(),
) where {T}
    opt.looped_linear_solver === MadNLPGPU.CUDSSSolver ||
        throw(ArgumentError("GPU batch path requires `linear_solver = CUDSSSolver` " *
                            "(got $(opt.looped_linear_solver))."))
    return CUDSSBatchLinearSolver(aug_com, nzvals_mat, n)
end
