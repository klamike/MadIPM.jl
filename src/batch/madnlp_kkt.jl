abstract type AbstractBatchKKTSystem{T, KKT<:MadNLP.AbstractKKTSystem} end

struct SparseSameStructureBatchKKTSystem{
    T,
    VT <: AbstractVector{T},
    VI32 <: AbstractVector{Int32},
    KKT <: MadNLP.SparseKKTSystem,
    LS <: MadNLP.AbstractLinearSolver{T},
} <: AbstractBatchKKTSystem{T, KKT}
    nzVals::VT

    aug_I::VI32
    aug_J::VI32

    kkts::Vector{KKT}

    batch_solver::LS

    nzVal_length::Int
    batch_size::Int

    function SparseSameStructureBatchKKTSystem(
        nzVals::VT,
        aug_I::VI32,
        aug_J::VI32,
        kkts::Vector{KKT},
        batch_solver::LS,
        nzVal_length::Int,
        batch_size::Int,
    ) where {T, VT<:AbstractVector{T}, VI32<:AbstractVector{Int32}, KKT<:MadNLP.SparseKKTSystem, LS<:MadNLP.AbstractLinearSolver{T}}
        new{T, VT, VI32, KKT, LS}(nzVals, aug_I, aug_J, kkts, batch_solver, nzVal_length, batch_size)
    end
end

function SparseSameStructureBatchKKTSystem(
    ::Type{KKT},
    batch_cb::SparseBatchCallback,
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
) where {KKT <: MadNLP.SparseKKTSystem}
    error("SparseSameStructureBatchKKTSystem only supported with CUDSS")
end

Base.length(batch_kkt::SparseSameStructureBatchKKTSystem) = batch_kkt.batch_size
Base.iterate(batch_kkt::SparseSameStructureBatchKKTSystem, i=1) = i > length(batch_kkt) ? nothing : (batch_kkt.kkts[i], i+1)
Base.getindex(batch_kkt::SparseSameStructureBatchKKTSystem, i::Int) = batch_kkt.kkts[i]
