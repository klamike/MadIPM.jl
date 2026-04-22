using Adapt
using KernelAbstractions
using MadNLPGPU
using CUDA.CUSPARSE
import BatchQuadraticModels as BQM

@testset "MadIPMCUDA" begin
    qp = simple_lp()
    qp_gpu = adapt(CuArray, qp)

    for (kkt, algo) in ((MadNLP.ScaledSparseKKTSystem, MadNLP.LDL     ),
                        (MadNLP.SparseKKTSystem      , MadNLP.LDL     ),
                        (MadIPM.NormalKKTSystem      , MadNLP.CHOLESKY))
        solver = MadIPM.MPCSolver(
            qp_gpu;
            kkt_system=kkt,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=algo,
            print_level=MadNLP.ERROR,
        )
        results = MadIPM.solve!(solver)
        @test results.status == MadNLP.SOLVE_SUCCEEDED
    end

    solver = MadIPM.MPCSolver(
        qp_gpu;
        kkt_system=MadNLP.SparseKKTSystem,
        linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.LDL,
        print_level=MadNLP.ERROR,
    )
    results = MadIPM.solve!(solver)
    @test results.status == MadNLP.SOLVE_SUCCEEDED

    @test_throws MethodError BQM._copy_sparse_structure!(qp_gpu.data.A, Vector{Int}(undef, nnz(qp_gpu.data.A)), Vector{Int}(undef, nnz(qp_gpu.data.A)))
    @test_throws MethodError BQM._copy_sparse_values!(qp_gpu.data.A, Vector{Float64}(undef, nnz(qp_gpu.data.A)))
    jrows = CuArray{Int}(undef, nnz(qp_gpu.data.A))
    jcols = similar(jrows)
    jvals = CuArray{Float64}(undef, nnz(qp_gpu.data.A))
    BQM._copy_sparse_structure!(qp_gpu.data.A, jrows, jcols)
    BQM._copy_sparse_values!(qp_gpu.data.A, jvals)

    MadIPM.update!(solver; c = CuArray([2.0, 3.0]), lcon = CuArray([2.0]), ucon = CuArray([2.0]), c0 = 1.5)
    qp_gpu = solver.problem.original_nlp

    updated = MadIPM.solve!(solver)
    reference = MadIPM.solve!(MadIPM.MPCSolver(
        qp_gpu;
        kkt_system=MadNLP.SparseKKTSystem,
        linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.LDL,
        print_level=MadNLP.ERROR,
    ))
    @test updated.status == MadNLP.SOLVE_SUCCEEDED
    @test Array(updated.solution) ≈ Array(reference.solution) atol=1e-6
    @test updated.objective ≈ reference.objective atol=1e-6
end

@testset "MadIPMCUDA batch (madipm_batch on GPU)" begin
    using SparseArrays
    A = sparse([1.0 1.0])
    qp_template = BQM.QuadraticModel(BQM.QPData(A, [1.0, 1.0], sparse([0.0 0.0; 0.0 0.0]);
        lcon = [1.0], ucon = [1.0], lvar = [0.0, 0.0], uvar = [1.0, 1.0], c0 = 0.0))
    nbatch = 3
    bqp_cpu = BQM.ObjRHSBatchQuadraticModel(qp_template, nbatch)
    bqp_cpu.c_batch    .= [1.0  5.0 -2.0; 2.0  1.0  3.0]
    for j in 1:nbatch
        bqp_cpu.meta.lcon[:, j] .= [1.0]
        bqp_cpu.meta.ucon[:, j] .= [1.0]
        bqp_cpu.meta.lvar[:, j] .= [0.0, 0.0]
        bqp_cpu.meta.uvar[:, j] .= [1.0, 1.0]
    end
    bqp_gpu = adapt(CuArray, bqp_cpu)

    stats = madipm_batch(bqp_gpu;
        kkt_system    = MadIPM.NormalKKTSystem,
        linear_solver = MadNLPGPU.CUDSSSolver,
        cudss_algorithm = MadNLP.CHOLESKY,
        print_level   = MadNLP.ERROR,
    )
    for j in 1:nbatch
        @test stats.status[j] == MadNLP.SOLVE_SUCCEEDED
    end
    @test Array(stats.objective)[1] ≈ 1.0  atol = 1e-5
    @test Array(stats.objective)[2] ≈ 1.0  atol = 1e-5
    @test Array(stats.objective)[3] ≈ -2.0 atol = 1e-5

    # Same problem via SparseKKTSystem (exercises augmented batch KKT path on GPU).
    stats_sk = madipm_batch(bqp_gpu;
        kkt_system    = MadNLP.SparseKKTSystem,
        linear_solver = MadNLPGPU.CUDSSSolver,
        cudss_algorithm = MadNLP.LDL,
        print_level   = MadNLP.ERROR,
    )
    for j in 1:nbatch
        @test stats_sk.status[j] == MadNLP.SOLVE_SUCCEEDED
    end
    @test Array(stats_sk.objective)[1] ≈ 1.0  atol = 1e-5
    @test Array(stats_sk.objective)[2] ≈ 1.0  atol = 1e-5
    @test Array(stats_sk.objective)[3] ≈ -2.0 atol = 1e-5
end

@testset "MadIPMCUDA BatchQuadraticModel (per-instance A on GPU)" begin
    using SparseArrays
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
    bnlp_cpu = BQM.BatchQuadraticModel(qps)
    bnlp_gpu = adapt(CuArray, bnlp_cpu)

    stats = madipm_batch(bnlp_gpu;
        kkt_system    = MadIPM.NormalKKTSystem,
        linear_solver = MadNLPGPU.CUDSSSolver,
        cudss_algorithm = MadNLP.CHOLESKY,
        print_level   = MadNLP.ERROR,
    )
    for j in 1:3
        @test stats.status[j] == MadNLP.SOLVE_SUCCEEDED
    end
    obj_gpu = Array(stats.objective)
    sol_gpu = Array(stats.solution)
    for j in 1:3
        ref = madipm(qps[j]; print_level = MadNLP.ERROR)
        @test obj_gpu[j]            ≈ ref.objective atol = 1e-5
        @test Vector(sol_gpu[:, j]) ≈ ref.solution  atol = 1e-5
    end
end

@testset "MadIPMCUDA batch update! on GPU" begin
    using SparseArrays
    A = sparse([1.0 1.0])
    qp_template = BQM.QuadraticModel(BQM.QPData(A, [1.0, 1.0], sparse([0.0 0.0; 0.0 0.0]);
        lcon = [1.0], ucon = [1.0], lvar = [0.0, 0.0], uvar = [1.0, 1.0], c0 = 0.0))
    nbatch = 2
    bqp_cpu = BQM.ObjRHSBatchQuadraticModel(qp_template, nbatch)
    bqp_cpu.c_batch    .= [1.0  1.0; 1.0  1.0]
    for j in 1:nbatch
        bqp_cpu.meta.lcon[:, j] .= [1.0]
        bqp_cpu.meta.ucon[:, j] .= [1.0]
        bqp_cpu.meta.lvar[:, j] .= [0.0, 0.0]
        bqp_cpu.meta.uvar[:, j] .= [1.0, 1.0]
    end
    bqp_gpu = adapt(CuArray, bqp_cpu)
    solver = MadIPM.UniformBatchMPCSolver(bqp_gpu;
        kkt_system    = MadIPM.NormalKKTSystem,
        linear_solver = MadNLPGPU.CUDSSSolver,
        cudss_algorithm = MadNLP.CHOLESKY,
        print_level   = MadNLP.ERROR,
    )
    new_c = adapt(CuArray, [2.0  -1.0; 3.0   2.0])
    MadIPM.update!(solver; c = new_c)
    stats = MadIPM.solve!(solver)
    @test all(Array(stats.status) .== MadNLP.SOLVE_SUCCEEDED)
end
