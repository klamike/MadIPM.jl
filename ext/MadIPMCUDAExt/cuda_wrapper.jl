using MadNLPGPU
import LinearAlgebra: BlasFloat

@kernel function _transfer_to_map!(dest, to_map, src)
    k = @index(Global, Linear)
    @inbounds begin
        Atomix.@atomic dest[to_map[k]] += src[k]
    end
end

function MadNLP.transfer!(dest::CUSPARSE.CuSparseMatrixCSC{Tv}, src::MadNLP.SparseMatrixCOO{Tv}, map::CuVector{Int}) where {Tv}
    fill!(nonzeros(dest), zero(Tv))
    length(map) > 0 && _transfer_to_map!(CUDABackend())(nonzeros(dest), map, src.V; ndrange = length(map))
    return
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.SparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadNLP.compress_jacobian!(
    kkt::MadIPM.NormalKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    n_slack = length(kkt.ind_ineq)
    kkt.A.V[end-n_slack+1:end] .= -1.0
    kkt.AT.nzVal .= kkt.A.V[kkt.A_csr_map]
    return
end

function MadIPM.coo_to_csr(
    n_rows,
    n_cols,
    Ai::CuVector{Ti},
    Aj::CuVector{Ti},
    Ax::CuVector{Tv},
) where {Tv, Ti}
    @assert length(Ai) == length(Aj) == length(Ax)
    B = sparse(Ai, Aj, Ax, n_rows, n_cols; fmt=:csr)
    return (B.rowPtr, B.colVal, B.nzVal)
end

@kernel function assemble_normal_system_kernel!(@Const(n_rows), @Const(n_cols), @Const(Jtp), @Const(Jtj), @Const(Jtx),
                                                @Const(Cp), @Const(Cj), Cx, @Const(Dx), @Const(Tv))
    i = @index(Global, Linear)

    for c in Cp[i]:Cp[i+1]-1
        j = Cj[c]
        acc = zero(Tv)

        p1 = Jtp[i]
        p2 = Jtp[j]
        p1_end = Jtp[i+1] - 1
        p2_end = Jtp[j+1] - 1

        while p1 <= p1_end && p2 <= p2_end
            k1 = Jtj[p1]
            k2 = Jtj[p2]

            if k1 == k2
                acc += Jtx[p1] * Dx[k1] * Jtx[p2]
                p1 += 1
                p2 += 1
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

function MadIPM.assemble_normal_system!(n_rows, n_cols, Jtp::CuArray{Ti}, Jtj::CuArray{Ti}, Jtx::CuArray{Tv}, Cp::CuArray{Ti}, Cj::CuArray{Ti}, Cx::CuArray{Tv}, Dx::CuArray{Tv}) where {Ti, Tv}
    assemble_normal_system_kernel!(CUDABackend())(n_rows, n_cols, Jtp, Jtj, Jtx, Cp, Cj, Cx, Dx, Tv; ndrange = n_rows)
end

# `@localmem T N` requires `N` to be a compile-time constant, so pass the
# scratch length through as `Val{N}` to make it part of the kernel's type
# signature.

@kernel function count_normal_nnz!(Cp, @Const(Jtp), @Const(Jtj), @Const(n_rows), ::Val{N}) where {N}
    i = @index(Global, Linear)

    # thread-local binary buffer
    xb = @localmem UInt8 N
    for k = 1:N
        xb[k] = 0
    end

    for c = Jtp[i]:Jtp[i+1]-1
        j = Jtj[c]
        xb[j] = 1
    end

    count = 0
    for j = i:n_rows
        for c = Jtp[j]:Jtp[j+1]-1
            k = Jtj[c]
            if xb[k] == 1
                count += 1
                break
            end
        end
    end

    Cp[i+1] = count
    nothing
end

@kernel function fill_normal_indices!(Cj, @Const(Cp), @Const(Jtp), @Const(Jtj), @Const(n_rows), ::Val{N}) where {N}
    i = @index(Global, Linear)

    xb = @localmem UInt8 N
    for k = 1:N
        xb[k] = 0
    end

    for c = Jtp[i]:Jtp[i+1]-1
        j = Jtj[c]
        xb[j] = 1
    end

    pos = Cp[i]
    for j = i:n_rows
        for c = Jtp[j]:Jtp[j+1]-1
            k = Jtj[c]
            if xb[k] == 1
                Cj[pos] = j
                pos += 1
                break
            end
        end
    end
    nothing
end

function MadIPM.build_normal_system(n_rows, n_cols, Jtp::CuVector{Ti}, Jtj::CuVector{Ti}) where {Ti}
    backend = CUDABackend()
    Cp = CUDA.ones(Ti, n_rows + 1)
    count_normal_nnz!(backend)(Cp, Jtp, Jtj, n_rows, Val(n_cols); ndrange = n_rows)
    Cp = cumsum(Cp)
    nnz_JtJ = CUDA.@allowscalar (Cp[end] - 1)
    Cj = CUDA.zeros(Ti, nnz_JtJ)
    fill_normal_indices!(backend)(Cj, Cp, Jtp, Jtj, n_rows, Val(n_cols); ndrange = n_rows)
    return (Cp, Cj)
end

MadIPM.sparse_csc_format(::Type{<:CuArray}) = CuSparseMatrixCSC

function MadIPM._csc_with_nzval(A::CuSparseMatrixCSC, nzval, n)
    return CuSparseMatrixCSC{eltype(nzval), eltype(A.colPtr)}(A.colPtr, A.rowVal, nzval, (n, n))
end
MadIPM._colptr(A::CuSparseMatrixCSC) = A.colPtr
MadIPM._rowval(A::CuSparseMatrixCSC) = A.rowVal
MadIPM._nzval(A::CuSparseMatrixCSC) = A.nzVal

# ===== Batch NormalKKTSystem GPU dispatches =====

# Per-column normal-matrix build using existing scalar GPU kernel.
function MadNLP.build_kkt!(
    bkkt::MadIPM.NormalUniformBatchKKTSystem{T, LS, MT},
) where {T, LS, MT <: CuMatrix{T}}
    AT = bkkt.AT  # CuSparseMatrixCSC
    Ap = AT.colPtr; Aj = AT.rowVal
    AAp = bkkt.aug_com.colPtr; AAj = bkkt.aug_com.rowVal
    n_tot = bkkt.n_tot; m = bkkt.m
    # AT.nzVal is shared across columns; per-instance Ax view uses A_vals.
    @inbounds for k in 1:bkkt.batch_size
        # Per-instance AT nzval: walk via A_csr_map to pull values from A_vals[:, k].
        Ax_k = view(bkkt.A_vals, :, k)[bkkt.A_csr_map]
        # Per-instance D = 1 / pr_diag.
        D_k     = view(bkkt.r_primal, :, k)
        prd_k   = view(bkkt.pr_diag,  :, k)
        @. D_k = one(T) / prd_k
        Cx_k = view(bkkt.aug_com_nzvals, :, k)
        MadIPM.assemble_normal_system!(m, n_tot, Ap, Aj, Ax_k, AAp, AAj, Cx_k, D_k)
        # Subtract du_diag from diagonal of normal matrix.
        # GPU diagonal subtract: use scalar indexing wrapped in @allowscalar (small m).
        CUDA.@allowscalar for i in 1:m
            for p in AAp[i]:(AAp[i + 1] - 1)
                if AAj[p] == i
                    Cx_k[p] -= bkkt.du_diag[i, k]
                    break
                end
            end
        end
    end
    return
end

# NormalKKT's `jtprod!` and `_batch_mul_A!` now route through BQM's
# `batch_spmv!(::BatchSparseOp, ...)`, which has its own GPU dispatch via
# `BatchQuadraticModelsCUDAExt`, so there's nothing GPU-specific to add
# here for those entry points.
