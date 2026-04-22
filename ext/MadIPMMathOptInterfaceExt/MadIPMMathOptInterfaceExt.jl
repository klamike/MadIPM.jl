# ============================================================================
# MadIPM × MathOptInterface — JuMP entry point.
#
# Adapted from NLPModelsJuMP.jl (MPL 2.0):
# https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl/blob/main/LICENSE.md
# ============================================================================

module MadIPMMathOptInterfaceExt

using Adapt
using BatchQuadraticModels
using MathOptInterface

import MadIPM
import MadNLP

const MOI = MathOptInterface
const VI  = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}

const _SCALAR_SETS = Union{
    MOI.EqualTo{Float64}, MOI.GreaterThan{Float64},
    MOI.LessThan{Float64}, MOI.Interval{Float64},
}

include("MOI_wrapper.jl")

__init__() = @eval MadIPM const Optimizer = $Optimizer

end # module
