# ============================================================================
# UniformBatchCallback — batch counterpart of `MadNLP.SparseCallback`.
# ============================================================================

"""
    UniformBatchCallback{T, VT, MT, VI, BM, FH, EH}

Carries the batched NLP plus the scratch buffers MadNLP reuses across calls.

- Sparsity (`jac_I/J`, `hess_I/J`) is shared across the batch.
- Scale factors (`obj_scale`, `obj_sign`, `con_scale`, `jac_scale`) are
  per-instance.
- `fixed_handler` and `equality_handler` pick how `lvar == uvar` and
  `lcon == ucon` rows are treated.
"""
struct UniformBatchCallback{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    VI <: AbstractVector{Int},
    BM <: NLPModels.AbstractBatchNLPModel,
    FH <: MadNLP.AbstractFixedVariableTreatment,
    EH <: MadNLP.AbstractEqualityTreatment,
} <: MadNLP.AbstractCallback{T, VT, FH}
    nlp::BM
    batch_size::Int

    # sizes (post fixed-variable removal)
    nvar::Int
    ncon::Int
    nnzj::Int
    nnzh::Int

    # per-instance buffers
    con_buffer::MT       # ncon × bs
    jac_buffer::MT       # nnzj × bs
    grad_buffer::MT      # nvar_nlp × bs
    hess_buffer::MT      # nnzh × bs

    # shared sparsity
    jac_I::VI
    jac_J::VI
    hess_I::VI
    hess_J::VI

    # per-instance scale factors
    obj_scale::MT        # 1 × bs
    obj_sign::MT         # 1 × bs, ±1
    con_scale::MT        # ncon × bs
    jac_scale::MT        # nnzj × bs

    # shared structural data
    fixed_handler::FH
    equality_handler::EH
    ind_eq::VI
    ind_ineq::VI
    ind_fixed::VI
    ind_lb::VI
    ind_ub::VI
    ind_llb::VI
    ind_uub::VI
end

# ---------- fixed-variable handler ----------

function MadNLP.create_sparse_fixed_handler(
    ::Type{MadNLP.MakeParameter},
    bnlp::NLPModels.AbstractBatchNLPModel{T},
    jac_I, jac_J, hess_I, hess_J, _hess_buffer,
) where {T}
    n    = NLPModels.get_nvar(bnlp)
    bs   = NLPModels.get_nbatch(bnlp)
    lvar = view(bnlp.meta.lvar, :, 1)
    uvar = view(bnlp.meta.uvar, :, 1)
    nnzj = NLPModels.get_nnzj(bnlp)
    nnzh = NLPModels.get_nnzh(bnlp)

    isfixed = lvar .== uvar
    isfree  = lvar .< uvar
    fixed   = findall(isfixed)
    isempty(fixed) && return MadNLP.NoFixedVariables(), n, nnzj, nnzh

    free = findall(isfree)
    nx   = length(free)

    map_full_to_free = fill!(similar(jac_I, n), -1)
    map_full_to_free[free] .= 1:nx

    ind_jac_free  = findall(@view(isfree[jac_J]))
    ind_hess_free = findall(@view(isfree[hess_I]) .&& @view(isfree[hess_J]))

    # Rewrite the Jacobian / Hessian sparsity into the free-variable space.
    _rewrite_sparsity!(hess_I, hess_J, ind_hess_free, map_full_to_free)
    _rewrite_sparsity!(jac_I,  jac_J,  ind_jac_free,  map_full_to_free; i_identity = true)

    fixed_handler = MadNLP.MakeParameter(
        free, fixed, ind_jac_free, ind_hess_free, Ref(NaN),
        similar(lvar, n * bs), similar(lvar, n * bs),
    )
    return fixed_handler, nx, length(ind_jac_free), length(ind_hess_free)
end

# Rewrite `(I, J)` in place to `(map[I[sel]], map[J[sel]])` (or leave `I`
# as-is when `i_identity = true` and only `J` is remapped).
function _rewrite_sparsity!(I, J, sel, map; i_identity::Bool = false)
    n = length(sel)
    Ii = i_identity ? I[sel]         : map[I[sel]]
    Jj =              map[J[sel]]
    resize!(I, n); copyto!(I, Ii)
    resize!(J, n); copyto!(J, Jj)
    return nothing
end

# ---------- callback builder ----------

function MadNLP.create_callback(
    ::Type{UniformBatchCallback{T, VT, MT, VI}},
    bnlp::NLPModels.AbstractBatchNLPModel{T};
    fixed_variable_treatment = MadNLP.MakeParameter,
    equality_treatment       = MadNLP.EnforceEquality,
    check_batch_structure::Bool = true,
) where {T, VT, MT, VI}
    bmeta      = bnlp.meta
    batch_size = bmeta.nbatch
    n, m       = bmeta.nvar, bmeta.ncon
    nnzj, nnzh = bmeta.nnzj, bmeta.nnzh

    x0 = NLPModels.get_x0(bnlp)

    jac_I  = similar(x0, Int, nnzj); jac_J  = similar(x0, Int, nnzj)
    hess_I = similar(x0, Int, nnzh); hess_J = similar(x0, Int, nnzh)

    obj_scale   = fill!(similar(x0, 1,    batch_size), one(T))
    con_scale   = fill!(similar(x0, m,    batch_size), one(T))
    con_buffer  = fill!(similar(x0, m,    batch_size), zero(T))
    jac_buffer  = fill!(similar(x0, nnzj, batch_size), zero(T))
    hess_buffer = fill!(similar(x0, nnzh, batch_size), zero(T))

    nnzj > 0 && NLPModels.jac_structure!(bnlp, jac_I, jac_J)
    nnzh > 0 && NLPModels.hess_structure!(bnlp, hess_I, hess_J)

    check_batch_structure && _assert_batch_structure(bmeta, batch_size)

    lvar, uvar = view(bmeta.lvar, :, 1), view(bmeta.uvar, :, 1)
    lcon, ucon = view(bmeta.lcon, :, 1), view(bmeta.ucon, :, 1)

    fixed_handler, nvar, nnzj, nnzh = MadNLP.create_sparse_fixed_handler(
        fixed_variable_treatment, bnlp, jac_I, jac_J, hess_I, hess_J, nothing,
    )
    equality_handler = equality_treatment()

    # Downsize jac_scale / grad_buffer to the free-variable space.
    jac_scale   = fill!(similar(x0, nnzj, batch_size), one(T))
    grad_buffer = fill!(similar(x0, nvar, batch_size), zero(T))

    ind_fixed = findall(lvar .== uvar)
    if !isempty(ind_fixed) && fixed_variable_treatment === MadNLP.MakeParameter
        ind_free = findall(lvar .< uvar)
        lvar = lvar[ind_free]
        uvar = uvar[ind_free]
    end
    indexes = MadNLP._parse_indexes(lvar, uvar, lcon, ucon, equality_treatment)
    obj_sign = fill!(similar(x0, 1, batch_size), bmeta.minimize ? one(T) : -one(T))

    return UniformBatchCallback{T, VT, MT, VI,
                                 typeof(bnlp), typeof(fixed_handler),
                                 typeof(equality_handler)}(
        bnlp, batch_size, nvar, m, nnzj, nnzh,
        con_buffer, jac_buffer, grad_buffer, hess_buffer,
        jac_I, jac_J, hess_I, hess_J,
        obj_scale, obj_sign, con_scale, jac_scale,
        fixed_handler, equality_handler,
        indexes.ind_eq, indexes.ind_ineq, ind_fixed,
        indexes.ind_lb, indexes.ind_ub,
        indexes.ind_llb, indexes.ind_uub,
    )
end

function _assert_batch_structure(bmeta, batch_size)
    row_sums = vcat(
        sum(bmeta.lvar .== bmeta.uvar; dims = 2),
        sum(isfinite.(bmeta.lvar);     dims = 2),
        sum(isfinite.(bmeta.uvar);     dims = 2),
        sum(bmeta.lcon .== bmeta.ucon; dims = 2),
        sum(isfinite.(bmeta.lcon);     dims = 2),
        sum(isfinite.(bmeta.ucon);     dims = 2),
    )
    @assert all((row_sums .== 0) .| (row_sums .== batch_size)) (
        "Batch fixed/bound/equality structure must match across instances")
end

# ---------- sparsity / structure getters ----------

function MadNLP._jac_sparsity_wrapper!(bcb::UniformBatchCallback,
                                        I::AbstractVector, J::AbstractVector)
    copyto!(I, bcb.jac_I); copyto!(J, bcb.jac_J); return nothing
end

function MadNLP._hess_sparsity_wrapper!(bcb::UniformBatchCallback,
                                         I::AbstractVector, J::AbstractVector)
    copyto!(I, bcb.hess_I); copyto!(J, bcb.hess_J); return nothing
end

function MadNLP.build_hessian_structure(bcb::UniformBatchCallback,
                                         ::Type{<:MadNLP.ExactHessian})
    hess_I = MadNLP.create_array(bcb, Int32, bcb.nnzh)
    hess_J = MadNLP.create_array(bcb, Int32, bcb.nnzh)
    MadNLP._hess_sparsity_wrapper!(bcb, hess_I, hess_J)
    return hess_I, hess_J
end
