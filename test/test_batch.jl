using Test
using MadIPM
using MadNLP
using BatchQuadraticModels
using SparseArrays
import BatchQuadraticModels as BQM

# Helpers --------------------------------------------------------------------

function _build_lp_batch(c_batch, lcon_batch, ucon_batch, lvar_batch, uvar_batch; A = sparse([1.0 1.0]))
    nbatch = size(c_batch, 2)
    qp_template = BQM.QuadraticModel(BQM.QPData(A, Vector(c_batch[:, 1]),
        sparse([0.0 0.0; 0.0 0.0]);
        lcon = Vector(lcon_batch[:, 1]),
        ucon = Vector(ucon_batch[:, 1]),
        lvar = Vector(lvar_batch[:, 1]),
        uvar = Vector(uvar_batch[:, 1]),
        c0 = 0.0))
    bqp = BQM.ObjRHSBatchQuadraticModel(qp_template, nbatch)
    bqp.c_batch    .= c_batch
    bqp.meta.lcon  .= lcon_batch
    bqp.meta.ucon  .= ucon_batch
    bqp.meta.lvar  .= lvar_batch
    bqp.meta.uvar  .= uvar_batch
    return bqp
end

# -- Tests --------------------------------------------------------------------

@testset "Batch LP via madipm_batch (default KKT)" begin
    A = sparse([1.0 1.0])
    nbatch = 3
    c    = [1.0  5.0 -2.0; 2.0  1.0  3.0]
    lcon = [1.0  1.0  1.0]
    ucon = [1.0  1.0  1.0]
    lvar = [0.0  0.0  0.0; 0.0  0.0  0.0]
    uvar = [1.0  1.0  1.0; 1.0  1.0  1.0]
    bqp = _build_lp_batch(c, lcon, ucon, lvar, uvar; A = A)

    stats = madipm_batch(bqp; print_level = MadNLP.ERROR)
    for j in 1:nbatch
        @test stats.status[j] == MadNLP.SOLVE_SUCCEEDED
    end
    @test stats.objective[1] ≈ 1.0  atol = 1e-5
    @test stats.objective[2] ≈ 1.0  atol = 1e-5
    @test stats.objective[3] ≈ -2.0 atol = 1e-5

    # Per-column comparison against scalar madipm.
    for j in 1:nbatch
        qp_j = BQM.QuadraticModel(BQM.QPData(A, Vector(bqp.c_batch[:, j]),
            sparse([0.0 0.0; 0.0 0.0]);
            lcon = [1.0], ucon = [1.0], lvar = [0.0, 0.0], uvar = [1.0, 1.0], c0 = 0.0))
        ref = madipm(qp_j; print_level = MadNLP.ERROR)
        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective[j]            ≈ ref.objective atol = 1e-5
        @test Vector(stats.solution[:, j])  ≈ ref.solution  atol = 1e-5
    end
end

@testset "Batch LP via batch NormalKKTSystem" begin
    A = sparse([1.0 1.0])
    nbatch = 3
    bqp = _build_lp_batch(
        [1.0  5.0 -2.0; 2.0  1.0  3.0],
        [1.0  1.0  1.0],
        [1.0  1.0  1.0],
        [0.0  0.0  0.0; 0.0  0.0  0.0],
        [1.0  1.0  1.0; 1.0  1.0  1.0];
        A = A,
    )
    stats = madipm_batch(bqp;
        print_level = MadNLP.ERROR,
        kkt_system  = MadIPM.NormalKKTSystem,
        linear_solver = MadNLP.LDLSolver,
    )
    for j in 1:nbatch
        @test stats.status[j] == MadNLP.SOLVE_SUCCEEDED
    end
    @test stats.objective[1] ≈ 1.0  atol = 1e-5
    @test stats.objective[2] ≈ 1.0  atol = 1e-5
    @test stats.objective[3] ≈ -2.0 atol = 1e-5
end

@testset "Batch LP with per-instance RHS" begin
    A = sparse([1.0 1.0])
    bqp = _build_lp_batch(
        [1.0  1.0; 1.0  1.0],
        [0.5  1.5],
        [0.5  1.5],
        [0.0  0.0; 0.0  0.0],
        [2.0  2.0; 2.0  2.0];
        A = A,
    )
    stats = madipm_batch(bqp; print_level = MadNLP.ERROR)
    @test all(stats.status .== MadNLP.SOLVE_SUCCEEDED)
    @test stats.objective[1] ≈ 0.5 atol = 1e-5
    @test stats.objective[2] ≈ 1.5 atol = 1e-5
end

@testset "Batch standard_form rejects mixed bound kinds" begin
    A = sparse([1.0 1.0])
    bqp = _build_lp_batch(
        [1.0  1.0; 1.0  1.0],
        [1.0  1.0],
        [1.0  1.0],
        [0.0  0.0; 0.0  0.0],
        [Inf  Inf; Inf  Inf];
        A = A,
    )
    # Flip one bound on column 2 to a finite uvar — kind mismatch.
    bqp.meta.uvar[1, 2] = 5.0
    @test_throws ArgumentError madipm_batch(bqp; print_level = MadNLP.ERROR)
end

@testset "Batch QP via BatchQuadraticModel (per-instance A and Q)" begin
    # Single constraint x1 + s_j*x2 = 1 with x_i >= 0, minimize x1 + x2.
    # Per-instance A coefficient on x2 (s_j) makes A vary across the batch.
    function make_qp(seed)
        A = sparse([1, 1], [1, 2], Float64[1.0, 1.0 + 0.25*seed])
        Q = sparse(Int[], Int[], Float64[], 2, 2)
        BQM.QuadraticModel(BQM.QPData(A, [1.0, 1.0], Q;
            lcon = [1.0], ucon = [1.0],
            lvar = [0.0, 0.0], uvar = [Inf, Inf], c0 = 0.0))
    end
    qps  = [make_qp(s) for s in 1:3]
    bnlp = BQM.BatchQuadraticModel(qps)
    stats = madipm_batch(bnlp; print_level = MadNLP.ERROR)
    for j in 1:3
        @test stats.status[j] == MadNLP.SOLVE_SUCCEEDED
        ref = madipm(qps[j]; print_level = MadNLP.ERROR)
        @test stats.objective[j]            ≈ ref.objective atol = 1e-5
        @test Vector(stats.solution[:, j])  ≈ ref.solution  atol = 1e-5
    end
end

@testset "Batch update! on UniformBatchMPCSolver" begin
    # Build solver once, mutate c via update!, re-solve, compare to a fresh solver.
    A = sparse([1.0 1.0])
    nbatch = 2
    bqp = _build_lp_batch(
        [1.0  1.0; 1.0  1.0],
        [1.0  1.0], [1.0  1.0],
        [0.0  0.0; 0.0  0.0],
        [1.0  1.0; 1.0  1.0]; A = A,
    )
    solver = MadIPM.UniformBatchMPCSolver(bqp; print_level = MadNLP.ERROR)
    new_c = [2.0  -1.0; 3.0   2.0]
    MadIPM.update!(solver; c = new_c)
    stats = MadIPM.solve!(solver)
    @test all(stats.status .== MadNLP.SOLVE_SUCCEEDED)

    # Reference: fresh solver with the new c.
    bqp_ref = _build_lp_batch(
        new_c, [1.0  1.0], [1.0  1.0],
        [0.0  0.0; 0.0  0.0], [1.0  1.0; 1.0  1.0]; A = A,
    )
    ref = madipm_batch(bqp_ref; print_level = MadNLP.ERROR)
    for j in 1:nbatch
        @test stats.objective[j] ≈ ref.objective[j] atol = 1e-5
    end
end

@testset "Batch LP recovers primal in original variable space" begin
    # min x1 + x2 s.t. x1 + x2 = 1, lvar = [0.2, 0.3], uvar = [0.8, 0.7].
    # Optimal at x = (0.2, 0.8) → 1.0 (LB binding for x1) — actually for these
    # bounds with constraint x1+x2=1 and minimizing x1+x2 the obj is fixed at
    # 1.0; we mainly check primal feasibility / recovery into orig space.
    A = sparse([1.0 1.0])
    bqp = _build_lp_batch(
        [1.0  1.0; 1.0  1.0],
        [1.0  1.0],
        [1.0  1.0],
        [0.2  0.3; 0.3  0.2],   # different lvar across batch (still VAR_LB_UB for both)
        [0.8  0.7; 0.7  0.8];
        A = A,
    )
    stats = madipm_batch(bqp; print_level = MadNLP.ERROR)
    @test all(stats.status .== MadNLP.SOLVE_SUCCEEDED)
    @test stats.objective[1] ≈ 1.0 atol = 1e-5
    @test stats.objective[2] ≈ 1.0 atol = 1e-5
    # Primal must respect the orig bounds.
    for j in 1:2
        @test all(stats.solution[:, j] .>= bqp.meta.lvar[:, j] .- 1e-5)
        @test all(stats.solution[:, j] .<= bqp.meta.uvar[:, j] .+ 1e-5)
        # Constraint feasibility: A x ≈ rhs.
        @test sum(stats.solution[:, j]) ≈ 1.0 atol = 1e-5
    end
end
