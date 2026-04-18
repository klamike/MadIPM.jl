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
include("solver.jl")
include("linear_solver.jl")

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
include("batch/madnlp/initialization.jl")
include("batch/madnlp/nlpmodels.jl")
include("batch/KKT/Sparse/normal.jl")

# Batch IPM hot-path kernels and main solver loop (std-form-aware).
include("batch/madipm/kernels.jl")
include("batch/madipm/solver.jl")

# Unified IPM kernels — dispatch on `Union{MPCSolver, AbstractBatchMPCSolver}`.
# Loaded last so both solver types are in scope.
include("kernels.jl")

export LPData, LinearModel, MPCSolver, QPData, QuadraticModel, madipm, madipm_batch, standard_form, update!

MadNLP.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPM
