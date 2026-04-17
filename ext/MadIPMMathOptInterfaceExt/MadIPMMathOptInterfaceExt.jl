#=
    Adapted from NLPModelsJuMP.jl

    The following source code is subject to the following LICENSE:

    Copyright (c) 2018-2019: Abel Soares Siqueira and Dominique Orban
    NLPModelsJuMP.jl is licensed under the MPL version 2.0.

    Full license available at this URL:
    https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl/blob/main/LICENSE.md

=#

module MadIPMMathOptInterfaceExt

using Adapt
using BatchQuadraticModels
using MathOptInterface
import MadNLP
import MadIPM

const MOI = MathOptInterface
const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const _SCALAR_SETS = Union{
    MOI.EqualTo{Float64},
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.Interval{Float64},
}

include("MOI_wrapper.jl")

function __init__()
    @eval MadIPM const Optimizer = $Optimizer
end

end
