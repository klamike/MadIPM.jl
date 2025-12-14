abstract type AbstractBatchCallback{CB<:MadNLP.AbstractCallback} end

struct SparseBatchCallback{
    T,
    VT <: AbstractVector{T},
    ConBufT <: AbstractArray{T},
    JacBufT <: AbstractArray{T},
    GradBufT <: AbstractArray{T},
    HessBufT <: AbstractArray{T},
    JacIdxT <: AbstractArray{Int},
    HessIdxT <: AbstractArray{Int},
    ConScaleT <: AbstractArray{T},
    JacScaleT <: AbstractArray{T},
    SharedFields <: Tuple,
    CB <: MadNLP.SparseCallback,
} <: AbstractBatchCallback{CB}
    con_buffer::ConBufT
    jac_buffer::JacBufT
    grad_buffer::GradBufT
    hess_buffer::HessBufT

    jac_I::JacIdxT
    jac_J::JacIdxT
    hess_I::HessIdxT
    hess_J::HessIdxT

    obj_scale::VT
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
        con_buffer::ConBufT,
        jac_buffer::JacBufT,
        grad_buffer::GradBufT,
        hess_buffer::HessBufT,
        jac_I::JacIdxT,
        jac_J::JacIdxT,
        hess_I::HessIdxT,
        hess_J::HessIdxT,
        obj_scale::VT,
        con_scale::ConScaleT,
        jac_scale::JacScaleT,
        callbacks::Vector{CB},
        nvar::Int,
        ncon::Int,
        nnzj::Int,
        nnzh::Int,
        batch_size::Int,
        shared::SharedFields,
    ) where {T, VT<:AbstractVector{T}, ConBufT<:AbstractArray{T}, JacBufT<:AbstractArray{T}, GradBufT<:AbstractArray{T}, HessBufT<:AbstractArray{T}, JacIdxT<:AbstractArray{Int}, HessIdxT<:AbstractArray{Int}, ConScaleT<:AbstractArray{T}, JacScaleT<:AbstractArray{T}, SharedFields<:Tuple, CB<:MadNLP.SparseCallback}
        new{T, VT, ConBufT, JacBufT, GradBufT, HessBufT, JacIdxT, HessIdxT, ConScaleT, JacScaleT, SharedFields, CB}(
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
    batch_size == 0 && error("SparseBatchCallback requires at least one model")

    nlp1 = nlps[1]
    nvar = NLPModels.get_nvar(nlp1)
    ncon = NLPModels.get_ncon(nlp1)
    nnzj = NLPModels.get_nnzj(nlp1.meta)
    nnzh = NLPModels.get_nnzh(nlp1.meta)

    for (i, nlp) in enumerate(nlps)
        @assert NLPModels.get_nvar(nlp) == nvar "All models must have same nvar (model $i differs)"
        @assert NLPModels.get_ncon(nlp) == ncon "All models must have same ncon (model $i differs)"
        @assert NLPModels.get_nnzj(nlp.meta) == nnzj "All models must have same nnzj (model $i differs)"
        @assert NLPModels.get_nnzh(nlp.meta) == nnzh "All models must have same nnzh (model $i differs)"
    end

    MI = Matrix{Int}

    con_buffer = :con_buffer in shared ? VT(undef, ncon) : MT(undef, ncon, batch_size)
    jac_buffer = :jac_buffer in shared ? VT(undef, nnzj) : MT(undef, nnzj, batch_size)
    grad_buffer = :grad_buffer in shared ? VT(undef, nvar) : MT(undef, nvar, batch_size)
    hess_buffer = :hess_buffer in shared ? VT(undef, nnzh) : MT(undef, nnzh, batch_size)

    jac_I = :jac_I in shared ? VI(undef, nnzj) : MI(undef, nnzj, batch_size)
    jac_J = :jac_J in shared ? VI(undef, nnzj) : MI(undef, nnzj, batch_size)
    hess_I = :hess_I in shared ? VI(undef, nnzh) : MI(undef, nnzh, batch_size)
    hess_J = :hess_J in shared ? VI(undef, nnzh) : MI(undef, nnzh, batch_size)

    obj_scale = VT(undef, batch_size)
    fill!(obj_scale, one(T))
    obj_sign = Ref(NLPModels.get_minimize(nlp1) ? one(T) : -one(T))  # NOTE: assumes shared nlp.meta.minimize
    con_scale = :con_scale in shared ? VT(undef, ncon) : MT(undef, ncon, batch_size)
    jac_scale = :jac_scale in shared ? VT(undef, nnzj) : MT(undef, nnzj, batch_size)

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

    CBType = MadNLP.SparseCallback{T, VT, VT, Base.RefValue{T}, VI, Model, FH{VT,VI}, EH}
    callbacks = Vector{CBType}(undef, batch_size)

    for i in 1:batch_size
        nlp = nlps[i]

        con_buffer_i = :con_buffer in shared ? con_buffer : _madnlp_unsafe_column_wrap(con_buffer, ncon, (i-1)*ncon + 1, VT)
        jac_buffer_i = :jac_buffer in shared ? jac_buffer : _madnlp_unsafe_column_wrap(jac_buffer, nnzj, (i-1)*nnzj + 1, VT)
        grad_buffer_i = :grad_buffer in shared ? grad_buffer : _madnlp_unsafe_column_wrap(grad_buffer, nvar, (i-1)*nvar + 1, VT)
        hess_buffer_i = :hess_buffer in shared ? hess_buffer : _madnlp_unsafe_column_wrap(hess_buffer, nnzh, (i-1)*nnzh + 1, VT)

        jac_I_i = :jac_I in shared ? jac_I : _madnlp_unsafe_column_wrap(jac_I, nnzj, (i-1)*nnzj + 1, VI)
        jac_J_i = :jac_J in shared ? jac_J : _madnlp_unsafe_column_wrap(jac_J, nnzj, (i-1)*nnzj + 1, VI)
        hess_I_i = :hess_I in shared ? hess_I : _madnlp_unsafe_column_wrap(hess_I, nnzh, (i-1)*nnzh + 1, VI)
        hess_J_i = :hess_J in shared ? hess_J : _madnlp_unsafe_column_wrap(hess_J, nnzh, (i-1)*nnzh + 1, VI)

        con_scale_i = :con_scale in shared ? con_scale : _madnlp_unsafe_column_wrap(con_scale, ncon, (i-1)*ncon + 1, VT)
        jac_scale_i = :jac_scale in shared ? jac_scale : _madnlp_unsafe_column_wrap(jac_scale, nnzj, (i-1)*nnzj + 1, VT)

        if !(:jac_I in shared) && nnzj > 0
            NLPModels.jac_structure!(nlp, jac_I_i, jac_J_i)
        end
        if !(:hess_I in shared) && nnzh > 0
            NLPModels.hess_structure!(nlp, hess_I_i, hess_J_i)
        end

        fixed_handler, nnzj_final, nnzh_final = MadNLP.create_sparse_fixed_handler(
            fixed_variable_treatment,
            nlp,
            jac_I_i, jac_J_i, hess_I_i, hess_J_i,
            hess_buffer_i,
        )

        equality_handler = equality_treatment()

        obj_scale_i = MadNLP._madnlp_unsafe_wrap(obj_scale, 1, i)

        callbacks[i] = MadNLP.SparseCallback(
            nlp,
            nvar, ncon, nnzj_final, nnzh_final,
            con_buffer_i,
            jac_buffer_i,
            grad_buffer_i,
            hess_buffer_i,
            jac_I_i,
            jac_J_i,
            hess_I_i,
            hess_J_i,
            obj_scale_i,
            obj_sign,
            con_scale_i,
            jac_scale_i,
            fixed_handler,
            equality_handler,
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
