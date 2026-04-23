# ============================================================================
# Ruiz equilibration scaling for the standard-form LP/QP.
#
# Given `min cᵀx + ½ xᵀQx  s.t.  Ax = b, x ≥ 0`, we diagonally rescale to
# `min c̃ᵀy + ½ yᵀQ̃y  s.t.  Ãy = b̃, y ≥ 0` where `y = D_c · x` and
#
#   Ã = D_r⁻¹ A D_c⁻¹,   b̃ = D_r⁻¹ b,   c̃ = D_c⁻¹ c,   Q̃ = D_c⁻¹ Q D_c⁻¹.
#
# `D_r`, `D_c` are the Ruiz fixed-point of the row/column max-norms: both are
# driven toward 1, which improves the conditioning seen by the sparse direct
# solver (CUDSS-LDL in particular is fragile on unscaled LPs).
#
# Integration:
#   * Scaling is computed inside the `MPCSolver` / `UniformBatchMPCSolver`
#     constructor, right after `standard_form`, and applied in-place to the
#     std-form model data.
#   * `update!(solver; …)` re-runs Ruiz only when `A` / `Q` changed
#     (tracked by signature). Pure `c / b / bounds / x0` updates reuse the
#     cached scaling — the std-form push already wrote unscaled values, we
#     just reapply the cached `row_scale` / `col_scale`.
#   * On solve exit, `update_solution!` unscales the iterate before the BQM
#     presolve workspace maps std → orig.
#
# Dispatch boundaries: everything here is generic over the matrix/vector
# types. GPU implementations of `_row_maxabs!` / `_col_maxabs!` /
# `_scale_rows_cols!` live in `MadIPMCUDAExt/cudss_batch.jl`; the CPU
# fallbacks below run on `SparseMatrixCSC` directly.
# ============================================================================

abstract type AbstractScaling end

"""
    NoScaling()

Identity scaling — std form is fed to the IPM unchanged.
"""
struct NoScaling <: AbstractScaling end

"""
    RuizScaling(; max_iter = 100, tol = 1e-8)

Ruiz row/column equilibration of the std-form constraint Jacobian `A`.
Iteratively rescales rows and columns by `1/√(row-max)` and `1/√(col-max)`
until the largest deviation from 1 is below `tol` (or `max_iter` is hit).
The accumulated scales are applied once to `A`, `c`, `b`, `Q` and the
initial iterate; the IPM runs in the scaled coordinate system.
"""
Base.@kwdef struct RuizScaling <: AbstractScaling
    max_iter::Int = 100
    tol::Float64 = 1e-8
end

# ----------------------------------------------------------------------------
# Cached per-problem state.
# ----------------------------------------------------------------------------

"""
    AbstractScaler

Per-problem state carrying the accumulated row/column scales and enough
metadata to decide whether `compute_scaling!` needs to run again after an
`update_standard_form!`.
"""
abstract type AbstractScaler end

struct NullScaler <: AbstractScaler end

mutable struct RuizScaler{T, VT <: AbstractVector{T}} <: AbstractScaler
    row_scale::VT              # length m  (D_r diagonal)
    col_scale::VT              # length n  (D_c diagonal)
    opt::RuizScaling
    a_signature::UInt          # sparsity+values hash of the A used for the last compute
    q_signature::UInt          # same for Q (0 if LP / no Q)
    applied::Bool              # has `apply_scaling!` been run with these scales yet?
end

# Build a scaler sized to the std-form NLP. Scales default to `1` so an
# unapplied scaler is an identity.
function RuizScaler(std_nlp; opt::RuizScaling = RuizScaling())
    A = _std_A(std_nlp)
    m, n = size(A)
    T  = eltype(_sparse_nzval(A))
    VT = typeof(similar(_sparse_nzval(A), T, 0))
    row = fill!(VT(undef, m), one(T))
    col = fill!(VT(undef, n), one(T))
    return RuizScaler{T, VT}(row, col, opt, zero(UInt), zero(UInt), false)
end

make_scaler(::NoScaling, _std_nlp) = NullScaler()
make_scaler(r::RuizScaling, std_nlp) = RuizScaler(std_nlp; opt = r)

# ----------------------------------------------------------------------------
# Model-type-specific accessors. The scaling engine is shared (one code
# path computes row / col scales from `_std_A(nlp)`), but applying them back
# to the model touches either a scalar `.data.{c,lcon,ucon}` + `.meta.x0`
# (`LinearModel` / `QuadraticModel`) or batched `.c_batch` + `.meta.{lcon,
# ucon, x0}` matrices (`BatchQuadraticModel`). The `_scaled_*` accessors
# below yield the right buffer for each.
# ----------------------------------------------------------------------------

@inline _std_A(std_nlp::Union{LinearModel, QuadraticModel}) =
    BatchQuadraticModels.operator_sparse_matrix(std_nlp.data.A)
@inline _std_Q(std_nlp::QuadraticModel) =
    BatchQuadraticModels.operator_sparse_matrix(std_nlp.data.Q)
@inline _std_Q(std_nlp::LinearModel) = nothing
@inline _scaled_c(std_nlp::Union{LinearModel, QuadraticModel})  = std_nlp.data.c
@inline _scaled_lcon(std_nlp::Union{LinearModel, QuadraticModel}) = std_nlp.data.lcon
@inline _scaled_ucon(std_nlp::Union{LinearModel, QuadraticModel}) = std_nlp.data.ucon
@inline _scaled_x0(std_nlp::Union{LinearModel, QuadraticModel})   = std_nlp.meta.x0

@inline _std_A(std_bnlp::BatchQuadraticModel) =
    BatchQuadraticModels.operator_sparse_matrix(std_bnlp.A)
@inline _std_Q(std_bnlp::BatchQuadraticModel) =
    BatchQuadraticModels.operator_sparse_matrix(std_bnlp.Q)
@inline _scaled_c(std_bnlp::BatchQuadraticModel)    = std_bnlp.c_batch
@inline _scaled_lcon(std_bnlp::BatchQuadraticModel) = std_bnlp.meta.lcon
@inline _scaled_ucon(std_bnlp::BatchQuadraticModel) = std_bnlp.meta.ucon
@inline _scaled_x0(std_bnlp::BatchQuadraticModel)   = std_bnlp.meta.x0

@inline _sparse_nzval(A::SparseArrays.SparseMatrixCSC) = SparseArrays.nonzeros(A)

# ----------------------------------------------------------------------------
# Public lifecycle.
# ----------------------------------------------------------------------------

refresh_scaling!(::NullScaler, _std_nlp) = nothing

"""
    refresh_scaling!(scaler, std_nlp)

Recompute the Ruiz scales if `A` (or `Q`) changed since the last call, then
reapply them to the std-form model. Idempotent when nothing moved: a fresh
signature check short-circuits both the compute and the apply.
"""
function refresh_scaling!(scaler::RuizScaler, std_nlp)
    a_sig = _signature(_std_A(std_nlp))
    Q = _std_Q(std_nlp)
    q_sig = Q === nothing ? zero(UInt) : _signature(Q)
    structure_changed = a_sig != scaler.a_signature || q_sig != scaler.q_signature

    if structure_changed
        compute_scaling!(scaler, std_nlp)
        scaler.a_signature = a_sig
        scaler.q_signature = q_sig
        scaler.applied = false
    end
    # Always (re)apply: `update_standard_form!` writes unscaled values into
    # `c / b / Q / A / x0`, so we have to push the cached scales back in.
    apply_scaling!(scaler, std_nlp)
    scaler.applied = true
    return scaler
end

"""
    compute_scaling!(scaler, std_nlp)

Run Ruiz iterations against `std_nlp.data.A` and populate `scaler.row_scale`,
`scaler.col_scale`. The A matrix itself is NOT touched here — the accumulated
scales are applied separately via `apply_scaling!`.
"""
function compute_scaling!(scaler::RuizScaler{T, VT}, std_nlp) where {T, VT}
    A = _std_A(std_nlp)
    row_scale = fill!(scaler.row_scale, one(T))
    col_scale = fill!(scaler.col_scale, one(T))

    r_work = similar(row_scale)
    c_work = similar(col_scale)
    A_work = copy(A)  # scratch: we mutate this, not the real A.

    tol_T = T(scaler.opt.tol)
    @inbounds for _ in 1:scaler.opt.max_iter
        _row_maxabs!(r_work, A_work)
        _col_maxabs!(c_work, A_work)
        _sqrt_or_one!(r_work)
        _sqrt_or_one!(c_work)
        _scale_rows_cols!(A_work, r_work, c_work)
        row_scale .*= r_work
        col_scale .*= c_work
        _converged(r_work, c_work, tol_T) && break
    end
    return scaler
end

"""
    apply_scaling!(scaler, std_nlp)

Push `scaler.row_scale` / `scaler.col_scale` into the std-form data:
`A, Q, c, b, x0` are mutated so the IPM sees the scaled problem. Safe to
call repeatedly — always operates on the current std data, so it composes
naturally with BQM's `update_standard_form!` dirty-tracking flow.
"""
apply_scaling!(::NullScaler, _std_nlp) = nothing
function apply_scaling!(scaler::RuizScaler, std_nlp)
    r = scaler.row_scale
    c = scaler.col_scale
    A = _std_A(std_nlp)
    Q = _std_Q(std_nlp)
    _scale_rows_cols_from_identity!(A, r, c)  # A ← D_r⁻¹ A D_c⁻¹
    if Q !== nothing && SparseArrays.nnz(Q) > 0
        _scale_symmetric_from_identity!(Q, c) # Q ← D_c⁻¹ Q D_c⁻¹
    end
    _scaled_c(std_nlp) ./= c                   # broadcasts over columns for batch
    lcon = _scaled_lcon(std_nlp); ucon = _scaled_ucon(std_nlp)
    lcon ./= r
    lcon === ucon || (ucon ./= r)               # std-form builder often aliases lcon ≡ ucon
    _scaled_x0(std_nlp) .*= c
    return scaler
end

# ----------------------------------------------------------------------------
# Iterate unscaling (solution recovery).
# ----------------------------------------------------------------------------

unscale_iterate!(::NullScaler, _x_std, _y_std, _zl_std) = nothing
function unscale_iterate!(scaler::RuizScaler, x_std, y_std, zl_std)
    # y = D_c x  →  x = y ./ D_c        (primal)
    # μ = D_r⁻¹ μ̃  →  μ̃ = D_r μ (scaled). Recover: μ = μ̃ ./ D_r.
    # z = D_c z̃  →  z = z̃ .* D_c.
    x_std  ./= scaler.col_scale
    y_std  ./= scaler.row_scale
    zl_std .*= scaler.col_scale
    return
end

# ----------------------------------------------------------------------------
# Sparsity/value signature (cheap change detection).
# ----------------------------------------------------------------------------
# Hashes structure (size + column pointers + row indices) plus a sketch of
# the non-zero values. Not cryptographic — good enough to detect whether A
# was rewritten by `update_standard_form!`.

_signature(A::SparseArrays.SparseMatrixCSC) =
    hash((size(A), SparseArrays.getcolptr(A), SparseArrays.rowvals(A),
          _values_sketch(SparseArrays.nonzeros(A))))

# Simple sketch: length + first / last / sum of non-zero values.
function _values_sketch(v::AbstractVector)
    n = length(v)
    n == 0 && return hash((0,))
    return hash((n, @inbounds(v[1]), @inbounds(v[n]), sum(v)))
end

# ----------------------------------------------------------------------------
# CPU kernels. GPU ones are defined in the CUDA extension.
# ----------------------------------------------------------------------------

# r[i] = max_j |A[i,j]|
function _row_maxabs!(r::AbstractVector{T}, A::SparseArrays.SparseMatrixCSC{T}) where {T}
    fill!(r, zero(T))
    rows = SparseArrays.rowvals(A); vals = SparseArrays.nonzeros(A)
    @inbounds for j in 1:size(A, 2)
        for p in SparseArrays.nzrange(A, j)
            v = abs(vals[p])
            i = rows[p]
            r[i] < v && (r[i] = v)
        end
    end
    return r
end

# c[j] = max_i |A[i,j]|
function _col_maxabs!(c::AbstractVector{T}, A::SparseArrays.SparseMatrixCSC{T}) where {T}
    fill!(c, zero(T))
    vals = SparseArrays.nonzeros(A)
    @inbounds for j in 1:size(A, 2)
        m = zero(T)
        for p in SparseArrays.nzrange(A, j)
            v = abs(vals[p])
            v > m && (m = v)
        end
        c[j] = m
    end
    return c
end

# Replace each entry with √x if positive, else leave as 1 (dead rows/cols stay neutral).
function _sqrt_or_one!(v::AbstractVector{T}) where {T}
    @inbounds for i in eachindex(v)
        v[i] = v[i] > zero(T) ? sqrt(v[i]) : one(T)
    end
    return v
end

# A[i,j] ← A[i,j] / (r[i] * c[j])
function _scale_rows_cols!(A::SparseArrays.SparseMatrixCSC{T}, r::AbstractVector{T},
                            c::AbstractVector{T}) where {T}
    rows = SparseArrays.rowvals(A); vals = SparseArrays.nonzeros(A)
    @inbounds for j in 1:size(A, 2)
        cj = c[j]
        for p in SparseArrays.nzrange(A, j)
            vals[p] = vals[p] / (r[rows[p]] * cj)
        end
    end
    return A
end

_scale_rows_cols_from_identity!(A::SparseArrays.SparseMatrixCSC, r, c) =
    _scale_rows_cols!(A, r, c)

# Q ← diag(c)⁻¹ Q diag(c)⁻¹  (Q is stored upper- or lower-triangular; apply to every stored entry).
function _scale_symmetric_from_identity!(Q::SparseArrays.SparseMatrixCSC{T},
                                          c::AbstractVector{T}) where {T}
    rows = SparseArrays.rowvals(Q); vals = SparseArrays.nonzeros(Q)
    @inbounds for j in 1:size(Q, 2)
        cj = c[j]
        for p in SparseArrays.nzrange(Q, j)
            vals[p] = vals[p] / (c[rows[p]] * cj)
        end
    end
    return Q
end

function _converged(r::AbstractVector{T}, c::AbstractVector{T}, tol::T) where {T}
    m = zero(T)
    @inbounds for x in r; d = abs(x - one(T)); d > m && (m = d); end
    @inbounds for x in c; d = abs(x - one(T)); d > m && (m = d); end
    return m < tol
end
