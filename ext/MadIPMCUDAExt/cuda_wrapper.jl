using MadNLPGPU
import LinearAlgebra: BlasFloat

# ============================================================================
# COO → CSC scatter.
# ============================================================================

@kernel function _transfer_to_map!(dest, to_map, src)
    k = @index(Global, Linear)
    @inbounds Atomix.@atomic dest[to_map[k]] += src[k]
end

function MadNLP.transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC{Tv},
    src::MadNLP.SparseMatrixCOO{Tv}, map::CuVector{Int},
) where {Tv}
    fill!(nonzeros(dest), zero(Tv))
    length(map) > 0 &&
        _transfer_to_map!(CUDABackend())(nonzeros(dest), map, src.V;
                                          ndrange = length(map))
    return nothing
end

# Raw-vector variant used by `SparseUniformBatchKKTSystem.build_kkt!`.
function MadIPM._scatter_to_csc!(dest::AnyCuArray{T}, src::AnyCuArray{T},
                                  map::AnyCuArray) where {T}
    fill!(dest, zero(T))
    length(map) > 0 &&
        _transfer_to_map!(CUDABackend())(dest, map, src; ndrange = length(map))
    return nothing
end

# ============================================================================
# KKT value plumbing — CUDA specializations.
# ============================================================================

MadNLP.compress_hessian!(
    kkt::MadNLP.SparseKKTSystem{T, VT, MT},
) where {T, VT, MT <: CUSPARSE.CuSparseMatrixCSC{T, Int32}} =
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)

function MadNLP.compress_jacobian!(
    kkt::MadIPM.NormalKKTSystem{T, VT, MT},
) where {T, VT, MT <: CUSPARSE.CuSparseMatrixCSC{T, Int32}}
    ns = length(kkt.ind_ineq)
    kkt.A.V[end-ns+1:end] .= -one(T)
    kkt.AT.nzVal .= kkt.A.V[kkt.A_csr_map]
    return nothing
end

# ============================================================================
# Sparse format shims and COO → CSR on CUDA.
# ============================================================================

MadIPM.sparse_csc_format(::Type{<:CuArray}) = CuSparseMatrixCSC

MadIPM._csc_with_nzval(A::CuSparseMatrixCSC, nzval, n) =
    CuSparseMatrixCSC{eltype(nzval), eltype(A.colPtr)}(
        A.colPtr, A.rowVal, nzval, (n, n))

MadIPM._colptr(A::CuSparseMatrixCSC) = A.colPtr
MadIPM._rowval(A::CuSparseMatrixCSC) = A.rowVal
MadIPM._nzval(A::CuSparseMatrixCSC)  = A.nzVal

function MadIPM.coo_to_csr(
    n_rows, n_cols,
    Ai::CuVector{Ti}, Aj::CuVector{Ti}, Ax::CuVector{Tv},
) where {Tv, Ti}
    @assert length(Ai) == length(Aj) == length(Ax)
    B = sparse(Ai, Aj, Ax, n_rows, n_cols; fmt = :csr)
    return B.rowPtr, B.colVal, B.nzVal
end

# ============================================================================
# Normal matrix build (A Σ⁻¹ Aᵀ).
# ============================================================================

@kernel function _assemble_normal_system_kernel!(
    @Const(n_rows), @Const(n_cols),
    @Const(Jtp), @Const(Jtj), @Const(Jtx),
    @Const(Cp),  @Const(Cj),  Cx, @Const(Dx), @Const(Tv),
)
    i = @index(Global, Linear)
    for c in Cp[i]:(Cp[i+1] - 1)
        j   = Cj[c]
        p1, p2       = Jtp[i], Jtp[j]
        p1_end, p2_end = Jtp[i+1] - 1, Jtp[j+1] - 1
        acc = zero(Tv)
        while p1 <= p1_end && p2 <= p2_end
            k1, k2 = Jtj[p1], Jtj[p2]
            if k1 == k2
                acc += Jtx[p1] * Dx[k1] * Jtx[p2]
                p1 += 1; p2 += 1
            elseif k1 < k2
                p1 += 1
            else
                p2 += 1
            end
        end
        Cx[c] = acc
    end
    nothing
end

function MadIPM.assemble_normal_system!(
    n_rows, n_cols,
    Jtp::CuArray{Ti}, Jtj::CuArray{Ti}, Jtx::CuArray{Tv},
    Cp::CuArray{Ti},  Cj::CuArray{Ti},  Cx::CuArray{Tv}, Dx::CuArray{Tv},
) where {Ti, Tv}
    _assemble_normal_system_kernel!(CUDABackend())(
        n_rows, n_cols, Jtp, Jtj, Jtx, Cp, Cj, Cx, Dx, Tv; ndrange = n_rows,
    )
end

# `@localmem T N` requires `N` at compile time; thread `Val{N}` through.

@kernel function _count_normal_nnz!(Cp, @Const(Jtp), @Const(Jtj),
                                      @Const(n_rows), ::Val{N}) where {N}
    i  = @index(Global, Linear)
    xb = @localmem UInt8 N
    for k = 1:N; xb[k] = 0; end
    for c = Jtp[i]:(Jtp[i+1] - 1); xb[Jtj[c]] = 1; end

    count = 0
    for j = i:n_rows
        for c = Jtp[j]:(Jtp[j+1] - 1)
            if xb[Jtj[c]] == 1
                count += 1; break
            end
        end
    end
    Cp[i + 1] = count
    nothing
end

@kernel function _fill_normal_indices!(Cj, @Const(Cp), @Const(Jtp), @Const(Jtj),
                                         @Const(n_rows), ::Val{N}) where {N}
    i   = @index(Global, Linear)
    xb  = @localmem UInt8 N
    for k = 1:N; xb[k] = 0; end
    for c = Jtp[i]:(Jtp[i+1] - 1); xb[Jtj[c]] = 1; end

    pos = Cp[i]
    for j = i:n_rows
        for c = Jtp[j]:(Jtp[j+1] - 1)
            if xb[Jtj[c]] == 1
                Cj[pos] = j; pos += 1; break
            end
        end
    end
    nothing
end

# Max shared memory per block on common CUDA arches is 48 KB; the bitmap is
# `UInt8[n_cols]`, so kernels can't run when `n_cols` goes above that limit
# (see the kernel's `@localmem UInt8 N`). For wide LPs (e.g. `osa-14` with
# `n = 52460`) we'd blow past it and hit a `ptxas uses too much shared data`
# compile error, so fall back to the generic CPU `build_normal_system` (which
# uses a single global bitmap — no smem bound) and lift the CSR arrays back
# to the device. `n_rows` is always modest (equals `m`), so the transfer of
# `(Jtp, Jtj)` is the only bulk cost.
const _NORMAL_SMEM_LIMIT = 48_000

function MadIPM.build_normal_system(
    n_rows, n_cols,
    Jtp::CuVector{Ti}, Jtj::CuVector{Ti},
) where {Ti}
    if n_cols * sizeof(UInt8) > _NORMAL_SMEM_LIMIT
        Cp_h, Cj_h = MadIPM.build_normal_system(n_rows, n_cols, Array(Jtp), Array(Jtj))
        return CuArray(Cp_h), CuArray(Cj_h)
    end

    backend = CUDABackend()
    Cp = CUDA.ones(Ti, n_rows + 1)
    _count_normal_nnz!(backend)(Cp, Jtp, Jtj, n_rows, Val(n_cols); ndrange = n_rows)
    Cp      = cumsum(Cp)
    nnz_JtJ = CUDA.@allowscalar (Cp[end] - 1)
    Cj      = CUDA.zeros(Ti, nnz_JtJ)
    _fill_normal_indices!(backend)(Cj, Cp, Jtp, Jtj, n_rows, Val(n_cols); ndrange = n_rows)
    return Cp, Cj
end

# ============================================================================
# Batch NormalKKTSystem — per-column GPU build.
#
# `jtprod!` and `_batch_mul_A!` go through BQM's batched `BatchSparseOperator`,
# whose GPU dispatch lives in `BatchQuadraticModelsCUDAExt`.
# ============================================================================

function MadNLP.build_kkt!(
    bkkt::MadIPM.NormalUniformBatchKKTSystem{T, LS, MT},
) where {T, LS, MT <: CuMatrix{T}}
    AT  = bkkt.AT
    Ap, Aj   = AT.colPtr,            AT.rowVal
    AAp, AAj = bkkt.aug_com.colPtr, bkkt.aug_com.rowVal
    m, n_tot = bkkt.m, bkkt.n_tot

    @inbounds for k in 1:bkkt.batch_size
        # Per-instance `AT.nzVal` materialized via the csr-map gather, and
        # per-instance `D = 1 / pr_diag`.
        Ax_k  = view(bkkt.A_vals, :, k)[bkkt.A_csr_map]
        D_k   = view(bkkt.r_primal, :, k)
        prd_k = view(bkkt.pr_diag,  :, k)
        @. D_k = one(T) / prd_k

        Cx_k = view(bkkt.aug_com_nzvals, :, k)
        MadIPM.assemble_normal_system!(m, n_tot, Ap, Aj, Ax_k, AAp, AAj, Cx_k, D_k)

        # Subtract `du_diag` from the diagonal. `m` is small, so an allowscalar
        # loop is cheap.
        CUDA.@allowscalar for i in 1:m, p in AAp[i]:(AAp[i+1] - 1)
            AAj[p] == i || continue
            Cx_k[p] -= bkkt.du_diag[i, k]
            break
        end
    end
    return nothing
end

# ============================================================================
# Fraction-to-boundary GPU kernels.
#
# The CPU kernels in `src/kernels/step.jl` iterate scalar-over-`(i, j)`; on GPU
# the SubArray-of-CuMatrix views fail scalar indexing, so we reduce per column
# on-device (batch_size threads, each scanning `n` rows).
# ============================================================================

@kernel function _ftb_primal_lb_kernel!(alpha_out, dx, x, xb, tau, n::Int)
    j = @index(Global, Linear)
    @inbounds begin
        T = eltype(alpha_out)
        a = T(Inf); τ = tau[1, j]
        for i in 1:n
            d = dx[i, j]
            d < zero(T) && (a = min(a, (-x[i, j] + xb[i, j]) * τ / d))
        end
        alpha_out[1, j] = a
    end
end

@kernel function _ftb_primal_ub_kernel!(alpha_out, dx, x, xb, tau, n::Int)
    j = @index(Global, Linear)
    @inbounds begin
        T = eltype(alpha_out)
        a = T(Inf); τ = tau[1, j]
        for i in 1:n
            d = dx[i, j]
            d > zero(T) && (a = min(a, (-x[i, j] + xb[i, j]) * τ / d))
        end
        alpha_out[1, j] = a
    end
end

@kernel function _ftb_dual_lb_kernel!(alpha_out, dz, z, tau, n::Int)
    j = @index(Global, Linear)
    @inbounds begin
        T = eltype(alpha_out)
        a = T(Inf); τ = tau[1, j]
        for i in 1:n
            d = dz[i, j]
            d < zero(T) && (a = min(a, -z[i, j] * τ / d))
        end
        alpha_out[1, j] = a
    end
end

@kernel function _ftb_dual_ub_kernel!(alpha_out, dz, z, tau, n::Int)
    j = @index(Global, Linear)
    @inbounds begin
        T = eltype(alpha_out)
        a = T(Inf); τ = tau[1, j]
        for i in 1:n
            d  = dz[i, j]; zi = z[i, j]
            (d < zero(T) && zi + d < zero(T)) &&
                (a = min(a, -zi * τ / d))
        end
        alpha_out[1, j] = a
    end
end

function _ftb_launch!(kernel!, alpha_out, mat, args...)
    n, bs = size(mat)
    n > 0 && bs > 0 && kernel!(CUDABackend())(alpha_out, mat, args..., n; ndrange = bs)
    return alpha_out
end

MadIPM._ftb_primal_lb!(a::AnyCuArray, dx::AnyCuArray, x::AnyCuArray, xb::AnyCuArray, τ::AnyCuArray) =
    _ftb_launch!(_ftb_primal_lb_kernel!, a, dx, x, xb, τ)
MadIPM._ftb_primal_ub!(a::AnyCuArray, dx::AnyCuArray, x::AnyCuArray, xb::AnyCuArray, τ::AnyCuArray) =
    _ftb_launch!(_ftb_primal_ub_kernel!, a, dx, x, xb, τ)
MadIPM._ftb_dual_lb!(a::AnyCuArray, dz::AnyCuArray, z::AnyCuArray, τ::AnyCuArray) =
    _ftb_launch!(_ftb_dual_lb_kernel!, a, dz, z, τ)
MadIPM._ftb_dual_ub!(a::AnyCuArray, dz::AnyCuArray, z::AnyCuArray, τ::AnyCuArray) =
    _ftb_launch!(_ftb_dual_ub_kernel!, a, dz, z, τ)

# ============================================================================
# Affine complementarity GPU kernels.
# ============================================================================

@kernel function _affine_compl_lb_kernel!(out, x, xl, z, dx, dz, αp, αd, n::Int)
    j = @index(Global, Linear)
    @inbounds begin
        T = eltype(out)
        s = zero(T); ap = αp[1, j]; ad = αd[1, j]
        for i in 1:n
            s += (x[i, j] + ap * dx[i, j] - xl[i, j]) * (z[i, j] + ad * dz[i, j])
        end
        out[1, j] = s
    end
end

@kernel function _affine_compl_ub_kernel!(out, xu, x, z, dx, dz, αp, αd, n::Int)
    j = @index(Global, Linear)
    @inbounds begin
        T = eltype(out)
        s = zero(T); ap = αp[1, j]; ad = αd[1, j]
        for i in 1:n
            s += (xu[i, j] - (x[i, j] + ap * dx[i, j])) * (z[i, j] + ad * dz[i, j])
        end
        out[1, j] = s
    end
end

function _affine_launch!(kernel!, out, mat, args...)
    n, bs = size(mat)
    n > 0 && bs > 0 && kernel!(CUDABackend())(out, mat, args..., n; ndrange = bs)
    return out
end

MadIPM._affine_compl_lb!(out::AnyCuArray, x::AnyCuArray, xl::AnyCuArray, z::AnyCuArray,
                          dx::AnyCuArray, dz::AnyCuArray, αp::AnyCuArray, αd::AnyCuArray) =
    _affine_launch!(_affine_compl_lb_kernel!, out, x, xl, z, dx, dz, αp, αd)

MadIPM._affine_compl_ub!(out::AnyCuArray, xu::AnyCuArray, x::AnyCuArray, z::AnyCuArray,
                          dx::AnyCuArray, dz::AnyCuArray, αp::AnyCuArray, αd::AnyCuArray) =
    _affine_launch!(_affine_compl_ub_kernel!, out, xu, x, z, dx, dz, αp, αd)
