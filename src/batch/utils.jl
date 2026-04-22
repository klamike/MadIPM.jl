# ---------- KKT-matrix view helpers ----------

# Unsafe-wrap one column of a column-major batch matrix as a `Vector`-typed
# alias without copying. Used to feed a per-instance kkt slice into scalar
# code (e.g. MadNLP's `LoopedBatchLinearSolver` factors each column with a
# plain scalar solver).
function _madnlp_unsafe_column_wrap(mat::MT, n, shift, ::Type{VT}) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return unsafe_wrap(VT, pointer(mat, shift), n)
end

# Rebuild a CSC with new nzval but shared structure — used to swap per-
# instance values into a CSC shell without reallocating the symbolic layout.
function _csc_with_nzval(A::SparseArrays.SparseMatrixCSC, nzval, n)
    return SparseArrays.SparseMatrixCSC(n, n, SparseArrays.getcolptr(A), SparseArrays.rowvals(A), nzval)
end

# ---------- `BatchSparseOperator` builders for the KKT's `J`, `Jᵀ`, `H` ----
# Each `_build_*_op` assembles a per-instance SpMV operator over the batch
# augmented system's nzvals matrix. The `coo_I` / `val_map` arguments index
# into that matrix; `aug_csc_map` / `jac_range` / `hess_range` identify the
# slots within the augmented COO triple.

"""
    _build_batch_op(nzVals, nz_map, val_map, coo_I, nrows) -> BatchSparseOperator

Assemble a `BatchSparseOperator` from the KKT COO triple. `coo_I` is the row
index per nonzero (operator rows); `val_map` is the col index (both the
structural column and the B-row lookup during SpMV — they coincide here);
`nz_map` is the row of the per-instance nzvals matrix to read for each
scatter slot.
"""
function _build_batch_op(nzVals, nz_map, val_map, coo_I, nrows)
    coo_I_int   = _as_int_vec(coo_I)
    val_map_int = _as_int_vec(val_map)
    rowptr, colidx = BatchQuadraticModels._coo_to_csr(coo_I_int, nrows)
    return BatchQuadraticModels._build_op(
        nzVals,
        coo_I_int, val_map_int,       # structural rows, cols
        rowptr, _as_int_vec(nz_map), val_map_int, colidx,
    )
end

@inline _as_int_vec(v::AbstractVector{Int}) = v
@inline function _as_int_vec(v::AbstractVector)
    out = similar(v, Int)
    out .= v
    return out
end

# `Jᵀ`: rows = var space (n_tot), cols = con space (m). Reads from
# `jac_range` of the COO triple; its original I-coord is con-indexed
# (needs `- n_tot` offset), J-coord is the var index used as the op's row.
function _build_jt_op(
    aug_I, aug_J, jac_range, n_tot,
    nzVals::AbstractMatrix{T}, aug_csc_map,
) where T
    n_jac = length(jac_range)
    coo_I = similar(aug_I, n_jac)
    coo_I .= aug_J[jac_range]
    nz_map = similar(aug_csc_map, n_jac)
    nz_map .= jac_range
    con_map = similar(aug_csc_map, n_jac)
    con_map .= aug_I[jac_range] .- Int32(n_tot)
    return _build_batch_op(nzVals, nz_map, con_map, coo_I, n_tot)
end

# `J`: rows = con space (m), cols = var space (n_tot). Mirror of `_build_jt_op`
# with rows/cols swapped.
function _build_j_op(
    aug_I, aug_J, jac_range, n_tot, m,
    nzVals::AbstractMatrix{T}, aug_csc_map,
) where T
    n_jac = length(jac_range)
    coo_I = similar(aug_I, n_jac)
    coo_I .= aug_I[jac_range] .- Int32(n_tot)
    nz_map = similar(aug_csc_map, n_jac)
    nz_map .= jac_range
    var_map = similar(aug_csc_map, n_jac)
    var_map .= aug_J[jac_range]
    return _build_batch_op(nzVals, nz_map, var_map, coo_I, m)
end

# `H` symmetric (both triangles). The COO triple carries only the lower
# triangle (`hess_range`); we replicate off-diagonal entries into the upper
# triangle via `offdiag_idx` so the resulting op computes full `Hx`.
function _build_hess_op(
    aug_I, aug_J, n_tot, n_hess,
    nzVals::AbstractMatrix{T}, aug_csc_map,
) where T
    if n_hess == 0
        nz_map = similar(aug_csc_map, 0)
        var_map = similar(aug_csc_map, 0)
        return _build_batch_op(nzVals, nz_map, var_map, similar(aug_I, 0), n_tot)
    end

    hess_range = n_tot+1:n_tot+n_hess
    hess_I = aug_I[hess_range]
    hess_J = aug_J[hess_range]

    offdiag_idx = findall(hess_I .!= hess_J)
    n_hess_sym = n_hess + length(offdiag_idx)

    coo_rows = similar(aug_I, n_hess_sym)
    coo_rows[1:n_hess]     .= hess_I
    coo_rows[n_hess+1:end] .= hess_J[offdiag_idx]

    nz_map = similar(aug_csc_map, n_hess_sym)
    nz_map[1:n_hess]     .= hess_range
    nz_map[n_hess+1:end] .= n_tot .+ offdiag_idx

    var_map = similar(aug_csc_map, n_hess_sym)
    var_map[1:n_hess]     .= hess_J
    var_map[n_hess+1:end] .= hess_I[offdiag_idx]

    return _build_batch_op(nzVals, nz_map, var_map, coo_rows, n_tot)
end

# ---------- public batch-vector container ----------

"""
    BatchVector{T, MT}

Thin wrapper around a `(len, batch_size)` matrix carrying batched vector
data. Exposes `MadNLP.full(bv) = bv.values` so MadNLP kernels that expect a
vector-shaped view can operate on all batch columns at once via broadcast.
"""
struct BatchVector{T, MT<:AbstractMatrix{T}}
    values::MT
end

MadNLP.full(bv::BatchVector) = bv.values

function BatchVector(
    ::Type{MT}, ::Type{VT},
    len::Int, batch_size::Int,
) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    values = MT(undef, len, batch_size)
    fill!(values, zero(T))
    return BatchVector{T, MT}(values)
end

# ---------- batch stats / counters ----------

"""
    BatchExecutionStats{T, VT, MT}

Public return value from `solve!(::UniformBatchMPCSolver)`. Matrix fields
(`solution`, `constraints`, `multipliers`, `multipliers_L/U`) hold one
column per batch instance; vector fields are `(nbatch,)`.

`stats[i]` returns a `NamedTuple` with the i-th instance's slice.
"""
mutable struct BatchExecutionStats{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    status::Vector{MadNLP.Status}  # (bs,)
    solution::MT                   # (nvar_nlp, bs)
    objective::VT                  # (bs,)
    constraints::MT                # (ncon, bs)
    dual_feas::VT                  # (bs,)
    primal_feas::VT                # (bs,)
    multipliers::MT                # (ncon, bs)
    multipliers_L::MT              # (nvar_nlp, bs)
    multipliers_U::MT              # (nvar_nlp, bs)
    iter::Vector{Int}              # (bs,)
    total_time::Vector{Float64}    # (bs,)
end

function BatchExecutionStats(::Type{MT}, ::Type{VT}, nvar_nlp::Int, ncon::Int, batch_size::Int) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BatchExecutionStats{T, VT, MT}(
        fill(MadNLP.INITIAL, batch_size),
        MT(undef, nvar_nlp, batch_size),
        VT(undef, batch_size),
        MT(undef, ncon, batch_size),
        VT(undef, batch_size),
        VT(undef, batch_size),
        MT(undef, ncon, batch_size),
        MT(undef, nvar_nlp, batch_size),
        MT(undef, nvar_nlp, batch_size),
        zeros(Int, batch_size),
        zeros(Float64, batch_size),
    )
end

function Base.getindex(stats::BatchExecutionStats, i::Int)
    return (
        status        = stats.status[i],
        solution      = view(stats.solution,      :, i),
        objective     = stats.objective[i],
        constraints   = view(stats.constraints,   :, i),
        dual_feas     = stats.dual_feas[i],
        primal_feas   = stats.primal_feas[i],
        multipliers   = view(stats.multipliers,   :, i),
        multipliers_L = view(stats.multipliers_L, :, i),
        multipliers_U = view(stats.multipliers_U, :, i),
        iter          = stats.iter[i],
        total_time    = stats.total_time[i],
    )
end

"""
    BatchCounters

Per-instance iteration / timing counters for the batched solve. The shared
timing fields (`linear_solver_time`, `eval_function_time`) accumulate
wall-clock across the whole batch; `k` and `total_time` are per-instance.
"""
mutable struct BatchCounters
    k::Vector{Int}              # per-instance iteration count
    start_time::Float64
    total_time::Vector{Float64} # per-instance total solve time
    linear_solver_time::Float64
    eval_function_time::Float64
    obj_cnt::Int
    obj_grad_cnt::Int
    con_cnt::Int
    factorization_cnt::Int
end
BatchCounters(batch_size::Int) = BatchCounters(zeros(Int, batch_size), 0.0, zeros(Float64, batch_size), 0.0, 0.0, 0, 0, 0, 0)
