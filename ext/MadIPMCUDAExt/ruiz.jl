# ============================================================================
# GPU kernels for Ruiz equilibration.
#
# Mirrors `src/scaling.jl` but on `CuSparseMatrixCSR` / `CuSparseMatrixCSC`:
# row / column max-abs reductions, in-place row/col scaling, symmetric Q
# scaling, and scalar helpers (sqrt-or-one, convergence, signature). All
# kernels stay on the device — only a single Float64 signature crosses the
# PCIe bus per `refresh_scaling!` call.
# ============================================================================

# ---- nzval accessors ----
MadIPM._sparse_nzval(A::CUSPARSE.CuSparseMatrixCSR) = A.nzVal
MadIPM._sparse_nzval(A::CUSPARSE.CuSparseMatrixCSC) = A.nzVal

# ---- signature: size + pointer + sum of |nzval| (one scalar transfer) ----
function MadIPM._signature(A::CUSPARSE.CuSparseMatrixCSR)
    nz = A.nzVal
    s  = length(nz) == 0 ? zero(eltype(nz)) : sum(abs, nz)
    return hash((size(A), UInt(pointer(A.rowPtr)), UInt(pointer(A.colVal)),
                 length(nz), s))
end
function MadIPM._signature(A::CUSPARSE.CuSparseMatrixCSC)
    nz = A.nzVal
    s  = length(nz) == 0 ? zero(eltype(nz)) : sum(abs, nz)
    return hash((size(A), UInt(pointer(A.colPtr)), UInt(pointer(A.rowVal)),
                 length(nz), s))
end

# ---- row / col max-abs reductions ----

@kernel function _row_maxabs_csr_kernel!(r, rowPtr, nzVal)
    i = @index(Global, Linear)
    @inbounds begin
        T = eltype(r)
        m = zero(T)
        for p in rowPtr[i]:(rowPtr[i+1] - 1)
            v = abs(nzVal[p]); v > m && (m = v)
        end
        r[i] = m
    end
end

@kernel function _col_maxabs_csr_kernel!(c, rowPtr, colVal, nzVal, m::Int)
    j = @index(Global, Linear)
    @inbounds begin
        T = eltype(c)
        mx = zero(T)
        # Column scan is O(nnz) per column — acceptable: we only do it inside
        # the (≪ 100) Ruiz iterations, and Netlib-size matrices have small nnz.
        for i in 1:m
            for p in rowPtr[i]:(rowPtr[i+1] - 1)
                if colVal[p] == j
                    v = abs(nzVal[p]); v > mx && (mx = v)
                    break
                end
            end
        end
        c[j] = mx
    end
end

@kernel function _col_maxabs_csc_kernel!(c, colPtr, nzVal)
    j = @index(Global, Linear)
    @inbounds begin
        T = eltype(c)
        m = zero(T)
        for p in colPtr[j]:(colPtr[j+1] - 1)
            v = abs(nzVal[p]); v > m && (m = v)
        end
        c[j] = m
    end
end

@kernel function _row_maxabs_csc_kernel!(r, colPtr, rowVal, nzVal, n::Int)
    i = @index(Global, Linear)
    @inbounds begin
        T = eltype(r)
        mx = zero(T)
        for j in 1:n
            for p in colPtr[j]:(colPtr[j+1] - 1)
                if rowVal[p] == i
                    v = abs(nzVal[p]); v > mx && (mx = v)
                    break
                end
            end
        end
        r[i] = mx
    end
end

function MadIPM._row_maxabs!(r::CuVector, A::CUSPARSE.CuSparseMatrixCSR)
    m = size(A, 1)
    m == 0 && return r
    _row_maxabs_csr_kernel!(CUDABackend())(r, A.rowPtr, A.nzVal; ndrange = m)
    return r
end

function MadIPM._col_maxabs!(c::CuVector, A::CUSPARSE.CuSparseMatrixCSR)
    m, n = size(A)
    n == 0 && return c
    # Column reduction on CSR: we materialize a CSC copy once (nnz-sized one-time
    # cost) to avoid the O(m*nnz) per-column scan. The CSC view is only used by
    # this reduction; no downstream code needs it.
    A_csc = CUSPARSE.CuSparseMatrixCSC(A)
    _col_maxabs_csc_kernel!(CUDABackend())(c, A_csc.colPtr, A_csc.nzVal; ndrange = n)
    return c
end

function MadIPM._col_maxabs!(c::CuVector, A::CUSPARSE.CuSparseMatrixCSC)
    n = size(A, 2)
    n == 0 && return c
    _col_maxabs_csc_kernel!(CUDABackend())(c, A.colPtr, A.nzVal; ndrange = n)
    return c
end

function MadIPM._row_maxabs!(r::CuVector, A::CUSPARSE.CuSparseMatrixCSC)
    m, n = size(A)
    m == 0 && return r
    A_csr = CUSPARSE.CuSparseMatrixCSR(A)
    _row_maxabs_csr_kernel!(CUDABackend())(r, A_csr.rowPtr, A_csr.nzVal; ndrange = m)
    return r
end

# ---- √-or-1 and convergence ----
# `max(zero, v)` guards dead rows; broadcast avoids a bespoke kernel.
function MadIPM._sqrt_or_one!(v::CuVector{T}) where {T}
    @. v = ifelse(v > zero(T), sqrt(v), one(T))
    return v
end

function MadIPM._converged(r::CuVector{T}, c::CuVector{T}, tol::T) where {T}
    err_r = length(r) == 0 ? zero(T) : maximum(x -> abs(x - one(T)), r)
    err_c = length(c) == 0 ? zero(T) : maximum(x -> abs(x - one(T)), c)
    return max(err_r, err_c) < tol
end

# ---- in-place A[i,j] ← A[i,j] / (r[i] c[j]) ----

@kernel function _scale_csr_kernel!(rowPtr, colVal, nzVal, r, c)
    i = @index(Global, Linear)
    @inbounds begin
        ri = r[i]
        for p in rowPtr[i]:(rowPtr[i+1] - 1)
            nzVal[p] = nzVal[p] / (ri * c[colVal[p]])
        end
    end
end

@kernel function _scale_csc_kernel!(colPtr, rowVal, nzVal, r, c)
    j = @index(Global, Linear)
    @inbounds begin
        cj = c[j]
        for p in colPtr[j]:(colPtr[j+1] - 1)
            nzVal[p] = nzVal[p] / (r[rowVal[p]] * cj)
        end
    end
end

function MadIPM._scale_rows_cols!(A::CUSPARSE.CuSparseMatrixCSR, r::CuVector, c::CuVector)
    m = size(A, 1)
    m == 0 && return A
    _scale_csr_kernel!(CUDABackend())(A.rowPtr, A.colVal, A.nzVal, r, c; ndrange = m)
    return A
end

function MadIPM._scale_rows_cols!(A::CUSPARSE.CuSparseMatrixCSC, r::CuVector, c::CuVector)
    n = size(A, 2)
    n == 0 && return A
    _scale_csc_kernel!(CUDABackend())(A.colPtr, A.rowVal, A.nzVal, r, c; ndrange = n)
    return A
end

MadIPM._scale_rows_cols_from_identity!(A::CUSPARSE.CuSparseMatrixCSR, r, c) =
    MadIPM._scale_rows_cols!(A, r, c)
MadIPM._scale_rows_cols_from_identity!(A::CUSPARSE.CuSparseMatrixCSC, r, c) =
    MadIPM._scale_rows_cols!(A, r, c)

# ---- symmetric Q ← diag(c)⁻¹ Q diag(c)⁻¹ ----
# Stored-triangle entries each get `Q[i,j] /= c[i]*c[j]`.

@kernel function _scale_symm_csr_kernel!(rowPtr, colVal, nzVal, c)
    i = @index(Global, Linear)
    @inbounds begin
        ci = c[i]
        for p in rowPtr[i]:(rowPtr[i+1] - 1)
            nzVal[p] = nzVal[p] / (ci * c[colVal[p]])
        end
    end
end

@kernel function _scale_symm_csc_kernel!(colPtr, rowVal, nzVal, c)
    j = @index(Global, Linear)
    @inbounds begin
        cj = c[j]
        for p in colPtr[j]:(colPtr[j+1] - 1)
            nzVal[p] = nzVal[p] / (c[rowVal[p]] * cj)
        end
    end
end

function MadIPM._scale_symmetric_from_identity!(Q::CUSPARSE.CuSparseMatrixCSR, c::CuVector)
    m = size(Q, 1)
    m == 0 && return Q
    _scale_symm_csr_kernel!(CUDABackend())(Q.rowPtr, Q.colVal, Q.nzVal, c; ndrange = m)
    return Q
end

function MadIPM._scale_symmetric_from_identity!(Q::CUSPARSE.CuSparseMatrixCSC, c::CuVector)
    n = size(Q, 2)
    n == 0 && return Q
    _scale_symm_csc_kernel!(CUDABackend())(Q.colPtr, Q.rowVal, Q.nzVal, c; ndrange = n)
    return Q
end
