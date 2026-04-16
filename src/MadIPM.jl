module MadIPM

using Printf
using LinearAlgebra
using BatchQuadraticModels
import SparseArrays
import MadNLP
import MadNLP: full, LDLFactorizations
import NLPModels
import BatchQuadraticModels: LPData, LinearModel, QPData, QuadraticModel

include("utils.jl")
include("structure.jl")
include("kernels.jl")
include("KKT/normalkkt.jl")
include("linear_solver.jl")
include("solver.jl")

export LPData, LinearModel, MPCSolver, QPData, QuadraticModel, madipm, presolve_qp, standard_form_qp

MadNLP.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPM
