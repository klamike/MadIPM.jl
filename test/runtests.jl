using Test

using MathOptInterface
using MadNLP
using MadIPM
using MadNLPTests
using NLPModels
import BatchQuadraticModels as BQM
using CUDA
using SparseArrays

function simple_lp()
    c = ones(2)
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2)
    Q = sparse(Int[], Int[], Float64[], 2, 2)
    data = BQM.QPData(
        A,
        c,
        Q;
        lcon = [1.0],
        ucon = [1.0],
        lvar = [0.0, 0.0],
        uvar = [Inf, Inf],
        c0 = 0.0,
    )
    return BQM.QuadraticModel(data)
end

function simple_qp_cross_term()
    A = sparse(Int[], Int[], Float64[], 0, 2)
    Q = sparse([2], [1], [1.0], 2, 2)
    return BQM.QuadraticModel(BQM.QPData(A, zeros(2), Q))
end

function bounded_lp()
    data = BQM.QPData(
        sparse(Int[], Int[], Float64[], 0, 1),
        [-1.0],
        sparse(Int[], Int[], Float64[], 1, 1);
        lvar = [0.0],
        uvar = [1.0],
    )
    return BQM.QuadraticModel(data)
end

@testset "Test with simple LP" begin
    qp = simple_lp()

    qp_solver = MadIPM.MPCSolver(
        qp;
        print_level=MadNLP.ERROR,
        regularization=MadIPM.NoRegularization(),
    )
    sol_ref = MadIPM.solve!(qp_solver)

    @testset "Standard formulation" begin
        new_qp, _ = MadIPM.standard_form(qp)
        solver = MadIPM.MPCSolver(new_qp; print_level=MadNLP.ERROR)
        sol = MadIPM.solve!(solver)
        @test sol.objective ≈ sol_ref.objective atol=1e-6
    end

    @testset "Step rule $rule" for rule in [
        MadIPM.AdaptiveStep(0.99),
        MadIPM.ConservativeStep(0.99),
        MadIPM.MehrotraAdaptiveStep(0.99),
    ]
        solver = MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, step_rule=rule)
        sol = MadIPM.solve!(solver)
        @test sol.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "NormalKKTSystem implementation" begin
        # Test
        linear_solver = MadNLP.LapackCPUSolver
        cb = MadNLP.create_callback(
            MadNLP.SparseCallback, qp,
        )
        kkt = MadNLP.create_kkt_system(
            MadIPM.NormalKKTSystem,
            cb,
            linear_solver;
        )
        MadNLPTests.test_kkt_system(kkt, cb)
    end

    @testset "Solve LP with NormalKKTSystem" begin
        solver = MadIPM.MPCSolver(
            qp;
            linear_solver=LDLSolver,
            print_level=MadNLP.ERROR,
            kkt_system=MadIPM.NormalKKTSystem,
            rethrow_error=true,
        )
        sol = MadIPM.solve!(solver)

        @test sol.status == MadNLP.SOLVE_SUCCEEDED
        @test sol.objective ≈ sol_ref.objective atol=1e-6
        @test sol.solution ≈ sol_ref.solution atol=1e-6
        @test sol.constraints ≈ sol_ref.constraints atol=1e-6
        @test sol.multipliers ≈ sol_ref.multipliers atol=1e-6
    end

    @testset "ScaledSparseKKTSystem" begin
        solver = MadIPM.MPCSolver(
            qp;
            print_level=MadNLP.ERROR,
            kkt_system=MadNLP.ScaledSparseKKTSystem,
            regularization=MadIPM.NoRegularization(),
        )
        sol = MadIPM.solve!(solver)
        @test sol.status == MadNLP.SOLVE_SUCCEEDED
        @test sol.objective ≈ sol_ref.objective atol=1e-6
        @test sol.solution ≈ sol_ref.solution atol=1e-6
    end

    @testset "Regularization $(reg)" for reg in [
        MadIPM.FixedRegularization(1e-8, -1e-9),
        MadIPM.AdaptiveRegularization(1e-8, -1e-9, 1e-9),
    ]
        solver = MadIPM.MPCSolver(
            qp;
            linear_solver=LDLSolver,
            print_level=MadNLP.ERROR,
            regularization=reg,
            rethrow_error=true,
        )
        sol = MadIPM.solve!(solver)
        @test sol.status == MadNLP.SOLVE_SUCCEEDED
        @test sol.objective ≈ sol_ref.objective atol=1e-6
        @test sol.solution ≈ sol_ref.solution atol=1e-6
    end
end

@testset "Quadratic cross term" begin
    qp = simple_qp_cross_term()
    x = [3.0, 5.0]
    g = zeros(2)
    NLPModels.grad!(qp, x, g)
    @test g ≈ [5.0, 3.0]
    @test NLPModels.obj(qp, x) ≈ 15.0
end

@testset "Fixed variable with MakeParameter" begin
    solver = MadIPM.MPCSolver(
        BQM.QuadraticModel(
            BQM.QPData(
                sparse([1, 1], [1, 2], [1.0, 1.0], 1, 3),
                [1.0, 1.0, 1.0],
                sparse(Int[], Int[], Float64[], 3, 3);
                lcon = [1.0],
                ucon = [Inf],
                lvar = [0.0, 0.0, 2.0],
                uvar = [Inf, Inf, 2.0],
            ),
        );
        print_level=MadNLP.ERROR,
        fixed_variable_treatment=MadNLP.MakeParameter,
        rethrow_error=true,
    )
    sol = MadIPM.solve!(solver)
    @test sol.status == MadNLP.SOLVE_SUCCEEDED
    @test sol.solution[3] == 2.0
end

@testset "Upper bound conversion" begin
    qp = bounded_lp()
    sol = MadIPM.solve!(MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, rethrow_error=true))
    @test sol.status == MadNLP.SOLVE_SUCCEEDED
    @test sol.solution[1] ≈ 1.0 atol=1e-6
    @test sol.objective ≈ -1.0 atol=1e-6
end

@testset "Trivial fixed solve unsupported" begin
    qp = BQM.QuadraticModel(
        BQM.QPData(
            sparse(Int[], Int[], Float64[], 0, 1),
            [1.0],
            sparse(Int[], Int[], Float64[], 1, 1);
            lvar = [2.0],
            uvar = [2.0],
            c0 = 3.0,
        ),
    )
    @test_throws ArgumentError MadIPM.MPCSolver(qp; print_level = MadNLP.ERROR, rethrow_error = true)
    @test_throws ArgumentError MadIPM.standard_form(qp)
end

@testset "Trivial fixed infeasible" begin
    qp = BQM.QuadraticModel(
        BQM.QPData(
            sparse([1], [1], [1.0], 1, 1),
            [0.0],
            sparse(Int[], Int[], Float64[], 1, 1);
            lcon = [3.0],
            ucon = [Inf],
            lvar = [2.0],
            uvar = [2.0],
        ),
    )
    sol = MadIPM.solve!(MadIPM.MPCSolver(qp; print_level = MadNLP.ERROR, rethrow_error = true))
    @test sol.status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
    @test sol.solution == [2.0]
end

@testset "Standard-form incremental update" begin
    qp = simple_lp()
    solver = MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, rethrow_error=true)
    sol1 = MadIPM.solve!(solver)
    @test sol1.status == MadNLP.SOLVE_SUCCEEDED

    MadIPM.update!(solver; c = [2.0, 3.0], lcon = [2.0], ucon = [2.0], c0 = 1.5)
    qp = solver.problem.original_nlp

    sol2 = MadIPM.solve!(solver)
    sol_ref = MadIPM.solve!(MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, rethrow_error=true))

    @test sol2.status == MadNLP.SOLVE_SUCCEEDED
    @test sol2.solution ≈ sol_ref.solution atol=1e-6
    @test sol2.objective ≈ sol_ref.objective atol=1e-6
    @test sol2.constraints ≈ sol_ref.constraints atol=1e-6
end

@testset "Standard-form update with non-unit scaling" begin
    qp = BQM.QuadraticModel(
        BQM.QPData(
            sparse([1, 1], [1, 2], [2.0, 8.0], 1, 2),
            [1.0, 10.0],
            sparse(Int[], Int[], Float64[], 2, 2);
            lcon = [8.0],
            ucon = [8.0],
            lvar = [0.0, 0.0],
            uvar = [Inf, Inf],
        ),
    )
    solver = MadIPM.MPCSolver(qp; print_level = MadNLP.ERROR, rethrow_error = true)
    sol1 = MadIPM.solve!(solver)
    @test sol1.status == MadNLP.SOLVE_SUCCEEDED

    MadIPM.update!(solver; c = [10.0, 1.0])
    sol2 = MadIPM.solve!(solver)
    sol_ref = MadIPM.solve!(MadIPM.MPCSolver(
        solver.problem.original_nlp;
        print_level = MadNLP.ERROR,
        rethrow_error = true,
    ))

    @test sol2.status == MadNLP.SOLVE_SUCCEEDED
    @test sol2.solution ≈ sol_ref.solution atol = 1e-6
    @test sol2.objective ≈ sol_ref.objective atol = 1e-6
end

@testset "Standard-form structural change requires rebuild" begin
    qp = simple_lp()
    solver = MadIPM.MPCSolver(qp; print_level = MadNLP.ERROR, rethrow_error = true)
    sol = MadIPM.solve!(solver)
    @test sol.status == MadNLP.SOLVE_SUCCEEDED

    @test_throws ArgumentError MadIPM.update!(solver; uvar = fill(1.0, length(qp.data.uvar)))
end

@testset "Float32 solve" begin
    A = sparse(Int[1], Int[1], Float32[1], 1, 1)
    data = BQM.LPData(
        A,
        Float32[1];
        lcon = Float32[1],
        ucon = Float32[1],
        lvar = Float32[0],
        uvar = Float32[Inf],
        c0 = 0f0,
    )
    model = BQM.LinearModel(data; x0 = Float32[1], y0 = Float32[0])
    solver = MadIPM.MPCSolver(model; tol = 1f-6, max_iter = 10, print_level = MadNLP.ERROR, rethrow_error = true)
    stats = MadIPM.solve!(solver)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
end

@testset "Standard-form start mapping" begin
    qp = BQM.QuadraticModel(
        BQM.QPData(
            sparse([1, 1, 2, 2], [1, 2, 2, 3], [1.0, 1.0, 1.0, -1.0], 2, 3),
            [1.0, -2.0, 0.5],
            sparse(Int[], Int[], Float64[], 3, 3);
            lcon = [1.0, -Inf],
            ucon = [3.0, 4.0],
            lvar = [0.0, -Inf, -1.0],
            uvar = [2.0, 5.0, Inf],
        );
        x0 = [0.25, -1.5, 2.0],
        y0 = [0.75, -0.2],
    )
    std, ws = MadIPM.standard_form(qp)
    @test MadIPM.recover_primal(ws, std.meta.x0) ≈ qp.meta.x0
    @test std.meta.y0[ws.con_start.row] ≈ qp.meta.y0
end

include("test_batch.jl")

if get(ENV, "MADIPM_SKIP_MOI", "") == ""
    @testset "MathOptInterface" begin
        include("MOI_wrapper.jl")
    end
end

if CUDA.functional()
    include("test_gpu.jl")
end
