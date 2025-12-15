abstract type AbstractBatchKKTSystem{T, KKT<:MadNLP.AbstractKKTSystem} end

struct SparseSameStructureBatchKKTSystem{T,
    KKT <: MadNLP.SparseKKTSystem, LS <: MadNLP.AbstractLinearSolver{T},
    VT <: AbstractVector{T}, VI32 <: AbstractVector{Int32},
} <: AbstractBatchKKTSystem{T, KKT}
    nzVals::VT
    aug_I::VI32
    aug_J::VI32
    kkts::Vector{KKT}
    batch_solver::LS
    nzVal_length::Int
    batch_size::Int

    function SparseSameStructureBatchKKTSystem(
        nzVals::VT, aug_I::VI32, aug_J::VI32,
        kkts::Vector{KKT}, batch_solver::LS,
        nzVal_length::Int, batch_size::Int,
    ) where {T,
        KKT<:MadNLP.SparseKKTSystem, LS<:MadNLP.AbstractLinearSolver{T},
        VT<:AbstractVector{T}, VI32<:AbstractVector{Int32},
    }
        new{T,KKT,LS,VT,VI32}(nzVals, aug_I, aug_J, kkts, batch_solver, nzVal_length, batch_size)
    end
end

function SparseSameStructureBatchKKTSystem(
    batch_cb::SparseBatchCallback,
    ind_cons, linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
)
    error("SparseSameStructureBatchKKTSystem only supported with CUDSS")
end
