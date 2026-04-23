"""
    MadIPM

Mehrotra predictor-corrector interior-point solver for LPs and convex QPs,
with a batched variant that solves a collection of problems sharing the
same sparsity/bound structure in parallel.

Scalar entry points: [`madipm`](@ref), [`MPCSolver`](@ref), [`solve!`](@ref),
[`update!`](@ref). Batch entry points: [`madipm_batch`](@ref),
[`UniformBatchMPCSolver`](@ref). Problem data comes from
[`BatchQuadraticModels`](@ref) (`LPData`/`QPData`/`LinearModel`/
`QuadraticModel` scalar types + `BatchQuadraticModel` / `ObjRHSBatchQuadraticModel`
for the batch).

The solver reformulates the user's input into standard form
(`Ax = b, z ≥ 0`) via `BatchQuadraticModels.standard_form`, runs the IPM on
the standard-form KKT, then recovers the solution / multipliers in the
original space. Repeat-solves with updated parameter values go through
`update!(solver; ...)` which pushes changes through the presolve workspace
without reconstructing anything.

GPU support is activated automatically when CUDA + KernelAbstractions +
MadNLPGPU are loaded; MathOptInterface integration is enabled by loading
MathOptInterface.
"""
module MadIPM

using Adapt
using Printf
using LinearAlgebra
using BatchQuadraticModels
import SparseArrays
import SparseArrays: SparseMatrixCSC, sparse
import MadNLP
import MadNLP: full, LDLFactorizations
import NLPModels
import BatchQuadraticModels: LPData, LinearModel, QPData, QuadraticModel,
    StandardFormWorkspace, StandardFormBatchWorkspace,
    standard_form, update_standard_form!,
    recover_primal, recover_primal!, recover_variable_multipliers!,
    ObjRHSBatchQuadraticModel, BatchQuadraticModel,
    BatchSparseOperator, batch_spmv!, batch_mapreduce!, batch_maximum!,
    _copy_sparse_structure!, _copy_sparse_values!, sparse_operator, operator_sparse_matrix

include("scaling.jl")
include("utils.jl")
include("KKT/normalkkt.jl")
include("structure.jl")
include("logging.jl")
include("nlpmodels.jl")

include("batch/utils.jl")
include("batch/views.jl")
include("batch/madnlp/rhs.jl")
include("batch/madnlp/callback.jl")
include("batch/KKT/KKT.jl")
include("batch/structure.jl")
include("batch/madnlp/linear_solver.jl")
include("batch/madnlp/kernels.jl")
include("batch/madnlp/nlpmodels.jl")
include("batch/KKT/Sparse/normal.jl")
include("batch/KKT/Sparse/scaled_augmented.jl")

include("solver/linear_solver.jl")

include("kernels/rhs.jl")
include("kernels/aug_diagonal.jl")
include("kernels/complementarity.jl")
include("kernels/step.jl")
include("kernels/regularization.jl")

include("solver/initialize.jl")
include("solver/termination.jl")
include("solver/factorize.jl")
include("solver/loop.jl")
include("solver/stats.jl")
include("solver/entry.jl")

export LPData, LinearModel, MPCSolver, QPData, QuadraticModel, madipm, madipm_batch, standard_form, update!,
    NoScaling, RuizScaling

MadNLP.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPM
