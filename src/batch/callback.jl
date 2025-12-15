abstract type AbstractBatchCallback{CB<:MadNLP.AbstractCallback} end

struct SparseBatchCallback{T,
    ConBufT <: AbstractArray{T}, JacBufT <: AbstractArray{T}, GradBufT <: AbstractArray{T}, HessBufT <: AbstractArray{T},
    JacIdxT <: AbstractArray{Int}, HessIdxT <: AbstractArray{Int},
    ObjScaleT <: AbstractArray{T}, ConScaleT <: AbstractArray{T}, JacScaleT <: AbstractArray{T},
    SharedFields <: Tuple, CB <: MadNLP.SparseCallback,
} <: AbstractBatchCallback{CB}
    con_buffer::ConBufT
    jac_buffer::JacBufT
    grad_buffer::GradBufT
    hess_buffer::HessBufT

    jac_I::JacIdxT
    jac_J::JacIdxT
    hess_I::HessIdxT
    hess_J::HessIdxT

    obj_scale::ObjScaleT
    con_scale::ConScaleT
    jac_scale::JacScaleT

    callbacks::Vector{CB}

    nvar::Int
    ncon::Int
    nnzj::Int
    nnzh::Int
    batch_size::Int
    shared::SharedFields

    function SparseBatchCallback(
        con_buffer::ConBufT, jac_buffer::JacBufT, grad_buffer::GradBufT, hess_buffer::HessBufT,
        jac_I::JacIdxT, jac_J::JacIdxT,
        hess_I::HessIdxT, hess_J::HessIdxT,
        obj_scale::ObjScaleT, con_scale::ConScaleT, jac_scale::JacScaleT,
        callbacks::Vector{CB},
        nvar::Int, ncon::Int, nnzj::Int, nnzh::Int, batch_size::Int,
        shared::SharedFields,
    ) where {T, ConBufT<:AbstractArray{T}, JacBufT<:AbstractArray{T}, GradBufT<:AbstractArray{T}, HessBufT<:AbstractArray{T}, JacIdxT<:AbstractArray{Int}, HessIdxT<:AbstractArray{Int}, ObjScaleT<:AbstractVector{T}, ConScaleT<:AbstractArray{T}, JacScaleT<:AbstractArray{T}, SharedFields<:Tuple, CB<:MadNLP.SparseCallback}
        new{T, ConBufT, JacBufT, GradBufT, HessBufT, JacIdxT, HessIdxT, ObjScaleT, ConScaleT, JacScaleT, SharedFields, CB}(
            con_buffer, jac_buffer, grad_buffer, hess_buffer,
            jac_I, jac_J, hess_I, hess_J,
            obj_scale, con_scale, jac_scale,
            callbacks,
            nvar, ncon, nnzj, nnzh, batch_size, shared
        )
    end
end

function SparseBatchCallback(
    ::Type{MT}, ::Type{VT}, ::Type{VI},
    nlps::Vector{Model};
    fixed_variable_treatment::Type{FH}=MadNLP.MakeParameter,
    equality_treatment::Type{EH}=MadNLP.EnforceEquality,
    shared::Tuple{Vararg{Symbol}}=(),
) where {T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}, VI <: AbstractVector{Int}, Model <: NLPModels.AbstractNLPModel{T, VT}, FH <: MadNLP.AbstractFixedVariableTreatment, EH <: MadNLP.AbstractEqualityTreatment}

    batch_size = length(nlps)

    nlp1 = nlps[1]
    nvar = NLPModels.get_nvar(nlp1)
    ncon = NLPModels.get_ncon(nlp1)
    nnzj = NLPModels.get_nnzj(nlp1.meta)
    nnzh = NLPModels.get_nnzh(nlp1.meta)

    x0 = NLPModels.get_x0(nlp1)
    MI = typeof(similar(x0, Int, 0, 0))

    con_buffer  = _alloc_batch_buffer(:con_buffer,  shared, ncon, batch_size, VT, MT)
    jac_buffer  = _alloc_batch_buffer(:jac_buffer,  shared, nnzj, batch_size, VT, MT)
    grad_buffer = _alloc_batch_buffer(:grad_buffer, shared, nvar, batch_size, VT, MT)
    hess_buffer = _alloc_batch_buffer(:hess_buffer, shared, nnzh, batch_size, VT, MT)

    jac_I  = _alloc_batch_buffer(:jac_I,  shared, nnzj, batch_size, VI, MI)
    jac_J  = _alloc_batch_buffer(:jac_J,  shared, nnzj, batch_size, VI, MI)
    hess_I = _alloc_batch_buffer(:hess_I, shared, nnzh, batch_size, VI, MI)
    hess_J = _alloc_batch_buffer(:hess_J, shared, nnzh, batch_size, VI, MI)

    # NOTE: obj_scale uses CPU arrays to avoid scalar indexing
    obj_scale = Vector{T}(undef, batch_size)
    fill!(obj_scale, one(T))
    obj_sign = NLPModels.get_minimize(nlp1) ? one(T) : -one(T)
    con_scale = _alloc_batch_buffer(:con_scale, shared, ncon, batch_size, VT, MT)
    jac_scale = _alloc_batch_buffer(:jac_scale, shared, nnzj, batch_size, VT, MT)

    fill!(con_buffer, zero(T))
    fill!(jac_buffer, zero(T))
    fill!(grad_buffer, zero(T))
    fill!(hess_buffer, zero(T))
    fill!(con_scale, one(T))
    fill!(jac_scale, one(T))

    if :jac_I in shared && :jac_J in shared && nnzj > 0
        NLPModels.jac_structure!(nlp1, jac_I, jac_J)
    end
    if :hess_I in shared && :hess_J in shared && nnzh > 0
        NLPModels.hess_structure!(nlp1, hess_I, hess_J)
    end

    CBType = MadNLP.SparseCallback{T, VT, Vector{T}, Base.RefValue{T}, VI, Model, FH{VT,VI}, EH}
    callbacks = Vector{CBType}(undef, batch_size)

    populate_structure = !(:jac_I in shared && :jac_J in shared)

    for i in 1:batch_size
        callbacks[i] = MadNLP.init_sparse_callback!(
            nlps[i],
            _batch_view(con_buffer,  :con_buffer,  shared, ncon, i, VT),
            _batch_view(jac_buffer,  :jac_buffer,  shared, nnzj, i, VT),
            _batch_view(grad_buffer, :grad_buffer, shared, nvar, i, VT),
            _batch_view(hess_buffer, :hess_buffer, shared, nnzh, i, VT),
            _batch_view(jac_I,  :jac_I,  shared, nnzj, i, VI),
            _batch_view(jac_J,  :jac_J,  shared, nnzj, i, VI),
            _batch_view(hess_I, :hess_I, shared, nnzh, i, VI),
            _batch_view(hess_J, :hess_J, shared, nnzh, i, VI),
            _scalar_view(obj_scale, i),
            obj_sign,
            _batch_view(con_scale, :con_scale, shared, ncon, i, VT),
            _batch_view(jac_scale, :jac_scale, shared, nnzj, i, VT);
            fixed_variable_treatment,
            equality_treatment,
            populate_structure = populate_structure || i == 1,
        )
    end

    return SparseBatchCallback(
        con_buffer,
        jac_buffer,
        grad_buffer,
        hess_buffer,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        obj_scale,
        con_scale,
        jac_scale,
        callbacks,
        nvar,
        ncon,
        nnzj,
        nnzh,
        batch_size,
        shared,
    )
end
