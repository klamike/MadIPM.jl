module TestMOI

using Test

import MathOptInterface as MOI
import MadIPM
import CUDA

function test_result_contract()
    model = MadIPM.Optimizer()
    @test MOI.get(model, MOI.ResultCount()) == 0
    @test_throws MOI.ResultIndexBoundsError MOI.get(model, MOI.ObjectiveValue())
    @test MOI.get(model, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    @test MOI.get(model, MOI.SolveTimeSec()) == 0.0
    @test MOI.get(model, MOI.RawStatusString()) == "OPTIMIZE_NOT_CALLED"
end

function test_array_type_attribute()
    model = MadIPM.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("array_type"), CUDA.CuArray)
    @test model.config.array_type === CUDA.CuArray
    @test MOI.get(model, MOI.RawOptimizerAttribute("array_type")) === CUDA.CuArray
end

function test_copy_to_resets_results()
    src1 = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src1)
    MOI.add_constraint(src1, x, MOI.GreaterThan(1.0))
    MOI.set(src1, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src1, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)

    model = MadIPM.Optimizer()
    MOI.copy_to(model, src1)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ResultCount()) == 1

    src2 = MOI.Utilities.Model{Float64}()
    y = MOI.add_variable(src2)
    MOI.add_constraint(src2, y, MOI.GreaterThan(2.0))
    MOI.set(src2, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src2, MOI.ObjectiveFunction{MOI.VariableIndex}(), y)

    MOI.copy_to(model, src2)
    @test MOI.get(model, MOI.ResultCount()) == 0
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    @test_throws MOI.ResultIndexBoundsError MOI.get(model, MOI.ObjectiveValue())
end

function test_fixed_variable_models()
    model = MOI.instantiate(MadIPM.Optimizer, with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    MOI.add_constraint(model, x, MOI.LessThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    @test_throws ArgumentError MOI.optimize!(model)

    model2 = MOI.instantiate(MadIPM.Optimizer, with_bridge_type = Float64)
    MOI.set(model2, MOI.Silent(), true)
    y = MOI.add_variable(model2)
    MOI.add_constraint(model2, y, MOI.EqualTo(2.0))
    MOI.set(model2, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model2, MOI.ObjectiveFunction{MOI.VariableIndex}(), y)
    @test_throws ArgumentError MOI.optimize!(model2)
end

function test_quadratic_diagonal_objective_value()
    model = MOI.instantiate(MadIPM.Optimizer, with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(2.0))
    q = MOI.ScalarQuadraticFunction(
        MOI.ScalarQuadraticTerm.([2.0], [x], [x]),
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(q)}(), q)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 2.0 atol = 1e-6
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 4.0 atol = 1e-6
end

function test_runtests()
    excludes = [
        "test_model_copy_to_UnsupportedAttribute",
        r"^test_linear_integration$",
        # Currently not supported
        "test_linear_complex",
        "test_quadratic",
        "test_conic",
        "test_solve_VariableIndex_ConstraintDual_MIN_SENSE",
        "test_solve_VariableIndex_ConstraintDual_MAX_SENSE",
    ]
    model = MOI.instantiate(MadIPM.Optimizer, with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true) # comment this to enable output
    config = MOI.Test.Config(
        atol = 1e-6,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
            MOI.SolverVersion,
        ],
    )
    MOI.Test.runtests(model, config, exclude=excludes)
    return
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

end  # module

TestMOI.runtests()
