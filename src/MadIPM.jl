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
    StandardFormWorkspace, standard_form, update_standard_form!,
    recover_primal, recover_primal!, recover_variable_multipliers!,
    _copy_sparse_structure!, _copy_sparse_values!, sparse_operator, operator_sparse_matrix

include("utils.jl")
include("KKT/normalkkt.jl")
include("structure.jl")
include("logging.jl")
include("nlpmodels.jl")
include("solver.jl")
include("kernels.jl")
include("linear_solver.jl")

export LPData, LinearModel, MPCSolver, QPData, QuadraticModel, madipm, standard_form, update!

MadNLP.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPM
