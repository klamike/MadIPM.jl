using Test
using MadIPM
using MadNLP
using BatchQuadraticModels
using SparseArrays
import BatchQuadraticModels as BQM

@testset "Batch LP via madipm_batch" begin
  # min c' x s.t. x1 + x2 = 1, 0 <= x <= 1
  A = sparse([1.0 1.0])
  qp_template = BQM.QuadraticModel(BQM.QPData(A, [1.0, 1.0], sparse([0.0 0.0; 0.0 0.0]);
    lcon = [1.0], ucon = [1.0],
    lvar = [0.0, 0.0], uvar = [1.0, 1.0], c0 = 0.0))
  nbatch = 3
  bqp = BQM.ObjRHSBatchQuadraticModel(qp_template, nbatch)
  bqp.c_batch .= [1.0 5.0 -2.0;
                  2.0 1.0  3.0]
  for j in 1:nbatch
    bqp.meta.lcon[:, j] .= [1.0]
    bqp.meta.ucon[:, j] .= [1.0]
    bqp.meta.lvar[:, j] .= [0.0, 0.0]
    bqp.meta.uvar[:, j] .= [1.0, 1.0]
  end

  stats = madipm_batch(bqp; print_level = MadNLP.ERROR)
  for j in 1:nbatch
    @test stats.status[j] == MadNLP.SOLVE_SUCCEEDED
  end
  @test stats.objective[1] ≈ 1.0 atol = 1e-5
  @test stats.objective[2] ≈ 1.0 atol = 1e-5
  @test stats.objective[3] ≈ -2.0 atol = 1e-5

  # Compare per-column to scalar madipm
  for j in 1:nbatch
    qp_j = BQM.QuadraticModel(BQM.QPData(A, Vector(bqp.c_batch[:, j]), sparse([0.0 0.0; 0.0 0.0]);
      lcon = [1.0], ucon = [1.0], lvar = [0.0, 0.0], uvar = [1.0, 1.0], c0 = 0.0))
    ref = madipm(qp_j; print_level = MadNLP.ERROR)
    @test ref.status == MadNLP.SOLVE_SUCCEEDED
    @test stats.objective[j] ≈ ref.objective atol = 1e-5
    @test Vector(stats.solution[:, j]) ≈ ref.solution atol = 1e-5
  end
end

@testset "Batch LP via batch NormalKKTSystem" begin
  A = sparse([1.0 1.0])
  qp_template = BQM.QuadraticModel(BQM.QPData(A, [1.0, 1.0], sparse([0.0 0.0; 0.0 0.0]);
    lcon = [1.0], ucon = [1.0], lvar = [0.0, 0.0], uvar = [1.0, 1.0], c0 = 0.0))
  nbatch = 3
  bqp = BQM.ObjRHSBatchQuadraticModel(qp_template, nbatch)
  bqp.c_batch .= [1.0 5.0 -2.0; 2.0 1.0 3.0]
  for j in 1:nbatch
    bqp.meta.lcon[:, j] .= [1.0]
    bqp.meta.ucon[:, j] .= [1.0]
    bqp.meta.lvar[:, j] .= [0.0, 0.0]
    bqp.meta.uvar[:, j] .= [1.0, 1.0]
  end
  stats = madipm_batch(bqp;
    print_level = MadNLP.ERROR,
    kkt_system = MadIPM.NormalKKTSystem,
    linear_solver = MadNLP.LDLSolver,
  )
  for j in 1:nbatch
    @test stats.status[j] == MadNLP.SOLVE_SUCCEEDED
  end
  @test stats.objective[1] ≈ 1.0 atol = 1e-5
  @test stats.objective[2] ≈ 1.0 atol = 1e-5
  @test stats.objective[3] ≈ -2.0 atol = 1e-5
end

@testset "Batch LP with per-instance RHS" begin
  # Vary lcon/ucon (both equal for equality), verify each column solves its own.
  A = sparse([1.0 1.0])
  qp_template = BQM.QuadraticModel(BQM.QPData(A, [1.0, 1.0], sparse([0.0 0.0; 0.0 0.0]);
    lcon = [0.5], ucon = [0.5], lvar = [0.0, 0.0], uvar = [2.0, 2.0], c0 = 0.0))
  nbatch = 2
  bqp = BQM.ObjRHSBatchQuadraticModel(qp_template, nbatch)
  bqp.c_batch .= [1.0 1.0; 1.0 1.0]
  bqp.meta.lcon .= [0.5 1.5]
  bqp.meta.ucon .= [0.5 1.5]
  for j in 1:nbatch
    bqp.meta.lvar[:, j] .= [0.0, 0.0]
    bqp.meta.uvar[:, j] .= [2.0, 2.0]
  end
  stats = madipm_batch(bqp; print_level = MadNLP.ERROR)
  @test all(stats.status .== MadNLP.SOLVE_SUCCEEDED)
  @test stats.objective[1] ≈ 0.5 atol = 1e-5
  @test stats.objective[2] ≈ 1.5 atol = 1e-5
end
