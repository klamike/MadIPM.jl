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
