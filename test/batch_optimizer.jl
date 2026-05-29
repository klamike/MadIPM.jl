using BatchOptInterface
using JuMP

const BOI = BatchOptInterface

@testset "BatchOptInterface optimizer" begin
    model = MOI.instantiate(MadIPM.BatchOptimizer; with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    p, _ = MOI.add_constrained_variable(
        model,
        BOI.Batched(MOI.Parameter.([1.0, 2.0, 4.0])),
    )
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(
        model,
        x,
        BOI.Batched(MOI.LessThan.([2.0, 1.0, 0.3])),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.add_constraint(
        model,
        MOI.ScalarQuadraticFunction(
            [MOI.ScalarQuadraticTerm(1.0, x, p)],
            MOI.ScalarAffineTerm{Float64}[],
            0.0,
        ),
        MOI.GreaterThan(1.0),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ResultCount()) == 3
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test [MOI.get(model, BOI.BatchTerminationStatus(i)) for i in 1:3] ==
          fill(MOI.OPTIMAL, 3)
    @test [MOI.get(model, MOI.VariablePrimal(i), x) for i in 1:3] ≈
          [1.0, 0.5, 0.25] atol = 1e-6
    @test [MOI.get(model, MOI.VariablePrimal(i), p) for i in 1:3] ==
          [1.0, 2.0, 4.0]
end

@testset "JuMP with BatchOptInterface optimizer" begin
    model = Model(
        () -> MOI.instantiate(MadIPM.BatchOptimizer; with_bridge_type = Float64),
    )
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, p in BOI.BatchedParameter([1.0, 2.0, 4.0]))
    @objective(model, Min, x)
    @constraint(model, x * p >= 1)
    @constraint(model, x in BOI.Batched(MOI.LessThan.([2.0, 1.0, 0.3])))
    optimize!(model)

    @test result_count(model) == 3
    @test termination_status(model) == MOI.OPTIMAL
    @test [BOI.termination_status(model, i) for i in 1:3] == fill(MOI.OPTIMAL, 3)
    @test [value(x; result = i) for i in 1:3] ≈ [1.0, 0.5, 0.25] atol = 1e-6
    @test [value(p; result = i) for i in 1:3] == [1.0, 2.0, 4.0]
end

@testset "BatchOptInterface optimizer with MOI bridges" begin
    model = MOI.instantiate(MadIPM.BatchOptimizer; with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    p, _ = MOI.add_constrained_variable(
        model,
        BOI.Batched(MOI.Parameter.([1.0, 2.0])),
    )
    MOI.add_constraint(
        model,
        MOI.VectorOfVariables([x]),
        MOI.Nonnegatives(1),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarQuadraticFunction(
            [MOI.ScalarQuadraticTerm(1.0, x, p)],
            MOI.ScalarAffineTerm{Float64}[],
            0.0,
        ),
        MOI.GreaterThan(1.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.ResultCount()) == 2
    @test [MOI.get(model, BOI.BatchTerminationStatus(i)) for i in 1:2] ==
          fill(MOI.OPTIMAL, 2)
    @test [MOI.get(model, MOI.VariablePrimal(i), x) for i in 1:2] ≈
          [1.0, 0.5] atol = 1e-6
end
