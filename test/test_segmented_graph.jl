using Test
using CUDA
using MadIPM
using MadNLP
using MadNLPGPU
using QuadraticModels
using BatchQuadraticModels: ObjRHSBatchQuadraticModel

const CUDAExt = Base.get_extension(MadIPM, :MadIPMCUDAExt)

function _segmented_graph_lp()
    QuadraticModel(
        [1.0, 1.0], Int[], Int[], Float64[];
        Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0],
        lvar=[0.0, 0.0], uvar=[Inf, Inf],
        x0=ones(2),
    )
end

@testset "Segmented CUDA Graph Capture" begin
    qps = [_segmented_graph_lp() for _ in 1:4]
    ref = [MadIPM.madipm(qp; print_level=MadNLP.ERROR) for qp in qps]

    cpu_bnlp = ObjRHSBatchQuadraticModel(qps)
    gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
    stats = MadIPM.madipm_batch(
        gpu_bnlp;
        print_level=MadNLP.ERROR,
        uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.LDL,
    )

    cache = CUDAExt._SEG_CACHE[]
    @test cache.valid
    @test cache.n_segments > 0
    @test length(cache.execs) == cache.n_segments

    CUDA.@allowscalar for i in eachindex(ref)
        @test stats[i].status == MadNLP.SOLVE_SUCCEEDED
        @test stats[i].objective ≈ ref[i].objective atol=1e-5
        @test Array(stats[i].solution) ≈ ref[i].solution atol=1e-5
    end
end
