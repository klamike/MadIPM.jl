#=
    Test script for segmented CUDA graph capture.

    Run on GPU machine:
        export JULIA_DEPOT_PATH=/tmp/mkjd3
        cd /tmp/mkworkspace/MadIPM.jl
        julia --project -e 'include("test/test_segmented_graph.jl")'
=#

using Test
using CUDA
using CUDSS
using KernelAbstractions
using MadNLPGPU
using MadIPM
using MadNLP
using QuadraticModels
using QuadraticModels: ObjRHSBatchQuadraticModel

# Access the CUDA extension internals
const CUDAExt = Base.get_extension(MadIPM, :MadIPMCUDAExt)

function simple_lp()
    QuadraticModel(
        [1.0, 1.0], Int[], Int[], Float64[];
        Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0],
        lvar=[0.0, 0.0], uvar=[Inf, Inf],
        x0=ones(2),
    )
end

function small_qp()
    n = 4
    c = [1.0, -2.0, 0.5, 1.0]
    Hrows = [1, 2, 3, 4]; Hcols = [1, 2, 3, 4]; Hvals = [2.0, 1.0, 3.0, 1.5]
    Arows = [1, 1, 2, 2]; Acols = [1, 2, 3, 4]; Avals = [1.0, 1.0, 1.0, 1.0]
    QuadraticModel(
        c, Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0, 0.5], ucon=[2.0, 1.5],
        lvar=zeros(n), uvar=fill(Inf, n),
        x0=ones(n),
    )
end

function solve_gpu_batch(qps; kwargs...)
    cpu_bnlp = ObjRHSBatchQuadraticModel(qps)
    gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
    return MadIPM.madipm_batch(
        gpu_bnlp;
        print_level=MadNLP.INFO,
        uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.LDL,
        kwargs...,
    )
end

@testset "Segmented CUDA Graph Capture" begin
    # Solve a batch of identical LPs (bs=4)
    qps = [simple_lp() for _ in 1:4]

    # CPU reference
    refs = [MadIPM.madipm(qp; print_level=MadNLP.ERROR) for qp in qps]
    for r in refs
        @test r.status == MadNLP.SOLVE_SUCCEEDED
    end

    # GPU batch solve — this exercises segmented graph capture
    println("\n=== GPU batch solve (bs=4) ===")
    stats = solve_gpu_batch(qps)

    # Check cache state
    cache = CUDAExt._SEG_CACHE[]
    println("\nSegmented graph cache:")
    println("  valid     = $(cache.valid)")
    println("  n_segments = $(cache.n_segments)")
    println("  na        = $(cache.na)")
    println("  #execs    = $(length(cache.execs))")
    println("  #cudss    = $(length(cache.cudss_calls))")

    @test cache.valid
    @test cache.n_segments == 4  # expect 4 segments (3 CUDSS breaks)
    @test length(cache.execs) == 4
    @test length(cache.cudss_calls) == 3

    # Verify correctness
    CUDA.@allowscalar for i in 1:4
        si = stats[i]
        @test si.status == MadNLP.SOLVE_SUCCEEDED
        @test si.objective ≈ refs[i].objective atol=1e-5
        @test Array(si.solution) ≈ refs[i].solution atol=1e-5
    end
    println("\nLP bs=4: PASSED")

    # Now test with QPs (bs=2)
    println("\n=== GPU batch solve QP (bs=2) ===")
    qps2 = [small_qp() for _ in 1:2]
    refs2 = [MadIPM.madipm(qp; print_level=MadNLP.ERROR) for qp in qps2]
    stats2 = solve_gpu_batch(qps2)

    CUDA.@allowscalar for i in 1:2
        si = stats2[i]
        @test si.status == MadNLP.SOLVE_SUCCEEDED
        @test si.objective ≈ refs2[i].objective atol=1e-6
        @test Array(si.solution) ≈ refs2[i].solution atol=1e-6
    end
    println("QP bs=2: PASSED")
end
