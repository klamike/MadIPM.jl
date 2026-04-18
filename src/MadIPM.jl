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
    StandardFormWorkspace, BatchStandardFormWorkspace, standard_form, update_standard_form!,
    recover_primal, recover_primal!, recover_variable_multipliers!,
    ObjRHSBatchQuadraticModel, BatchSparseOp, batch_spmv!, batch_mapreduce!, batch_maximum!,
    _copy_sparse_structure!, _copy_sparse_values!, sparse_operator, operator_sparse_matrix

include("utils.jl")
include("KKT/normalkkt.jl")
include("structure.jl")
include("logging.jl")
include("nlpmodels.jl")

# Batch infrastructure (madipm-agnostic): types, callbacks, KKT systems,
# linear solver. These can be reused by any other batched IPM.
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

# Linear-solve glue: scalar + batch `solve_system!` and `factorize_wrapper!`.
include("linear_solver.jl")

# IPM kernels — split by section. Each file collocates the unified
# `AnyMPCSolver` dispatch (when applicable) with the scalar (`MPCSolver`)
# and batch (`AbstractBatchMPCSolver`) specializations.
include("kernels/rhs.jl")
include("kernels/aug_diagonal.jl")
include("kernels/complementarity.jl")
include("kernels/step.jl")
include("kernels/regularization.jl")

# Solver loop, split by section (mirrors src/kernels/). Each file
# collocates the scalar and batch implementations of one phase.
include("solver/initialize.jl")
include("solver/termination.jl")
include("solver/factorize.jl")
include("solver/loop.jl")
include("solver/stats.jl")
include("solver/entry.jl")

export LPData, LinearModel, MPCSolver, QPData, QuadraticModel, madipm, madipm_batch, standard_form, update!

MadNLP.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPM
