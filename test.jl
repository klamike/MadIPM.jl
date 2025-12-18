using Pkg; Pkg.instantiate()

using MadIPM, MadNLP
using QuadraticModels, NLPModels, ExaModels, ExaModelsPower
using MadNLPGPU, CUDA, CUDSS, KernelAbstractions
using PProf, NVTX
using Random, Distributions, SparseArrays, Memoize

const MODE = :profile;

const FLOAT_TYPE = Float64
const KWARGS = (
    print_level=MadNLP.ERROR,
    tol=1e-4,
    check_residual=true, tol_linear_solve=1e-3,
    cudss_ir=3,
    cudss_algorithm=MadNLP.LDL,
    regularization=MadIPM.FixedRegularization(1e-8, -1e-8),
    rethrow_error=true,
)

@memoize parse_data(case) = ExaModelsPower.parse_ac_power_data("pglib_opf_case$case.m")

function build_qps(case, batch_size; T=FLOAT_TYPE)
    data = parse_data(case);
    core = ExaCore(T);  # formulate on CPU
    dcopf, ~, ~, (pd, bs) = ExaModelsPower.build_dcopf(data; core);

    pd0 = dcopf.θ[1+pd.offset:pd.offset+pd.length];
    pds = [begin
               rng = Xoshiro(i)
               d_local = Uniform(0.95, 1.05)
               d_global = Uniform(0.6, 0.8)

               noise = similar(pd0)
               rand!(rng, d_local, noise)

               noise *= rand(rng, d_global)

               copy(pd0) .* noise
           end for i in 1:batch_size];

    bs0 = dcopf.θ[1+bs.offset:bs.offset+bs.length];
    bss = [begin
               rng = Xoshiro(i)
               d_local = Uniform(0.975, 1.025)

               noise = similar(bs0)
               rand!(rng, d_local, noise)

               copy(bs0) .* noise
           end for i in 1:batch_size];

    return [begin
                set_parameter!(core, pd, pds[i])  # changes rhs
                # set_parameter!(core, bs, bss[i])  # changes jac
                qp = QuadraticModel(dcopf, get_x0(dcopf))
                pqp, flag = MadIPM.presolve_qp(qp)
                @assert flag
                sqp = MadIPM.standard_form_qp(pqp)
                convert(QuadraticModel{Float64, CuVector{Float64}}, sqp)
            end for i in 1:batch_size]
end

function run_loop(qps)
    t = typeof(first(qps)).parameters
    stats = Vector{MadNLP.MadNLPExecutionStats{t[1],t[2]}}(undef, length(qps))
    for (i, qp) in enumerate(qps)
        stats[i] = MadIPM.madipm(qp; linear_solver=MadNLPGPU.CUDSSSolver, KWARGS...)
    end
    return stats
end

function run_batch(qps)
    return MadIPM.madipm(qps; KWARGS...)
end

function run_benchmark(case, batch_size; T=FLOAT_TYPE, warmup=false)
    NVTX.@range "Building" begin
        qps = build_qps(case, batch_size; T);
    end
    
    NVTX.@range "Loop" begin
        t_loop = @elapsed stats = run_loop(qps)
    end
    for (i, stat) in enumerate(stats)
        @assert stat.status == MadNLP.SOLVE_SUCCEEDED "Got $(stat.status) for sample $i"
    end

    NVTX.@range "Batch" begin
        t_batch = @elapsed batch_stats = run_batch(qps)
    end
    for (i, stat) in enumerate(batch_stats)
        @assert stat.status == MadNLP.SOLVE_SUCCEEDED "Got $(stats[i].status) for sample $i in batch"
        @assert stat.objective ≈ stats[i].objective atol=1e-3
    end
    
    if !warmup
        if t_loop > t_batch
            speedup = t_loop / t_batch
            @info "$case x $batch_size -- Batch is $(round(speedup, digits=2))x faster" t_loop t_batch (t_loop  - t_batch) (t_loop / batch_size) (t_batch / batch_size)
        else
            slowdown = t_batch / t_loop
            @error "$case x $batch_size -- Batch is $(round(slowdown, digits=2))x slower" t_loop t_batch (t_batch  - t_loop) (t_loop / batch_size) (t_batch / batch_size)
        end
    end
end

NVTX.@range "Precompile" begin
    run_benchmark("5_pjm", 1, warmup=true)
end

CASES = [
    # "14_ieee",
    # "30_ieee",
    # "57_ieee",
    "89_pegase",
    # "118_ieee",
    # "300_ieee",
    "1354_pegase",
    # "1888_rte",
    "2869_pegase",
    "6470_rte",
    "9241_pegase",
    # "13659_pegase",
]

BATCH_SIZES = [
    # 1,
    # 2,
    4,
    # 8,
    16,
    # 24,
    # 32,
    # 48,
    64,
    # 96,
    # 128,
    # 160,
    # 192,
    # 256,
    # 320,
    # 480,
    # 512,
    # 768,
    # 1024,
]

for case in CASES
    println()
    println()
    println("\t\tStarting $case")
    println()
    for batch_size in BATCH_SIZES
        try
            NVTX.@range "Warmup" begin
                run_benchmark(case, batch_size, warmup=true)
            end
            CUDA.@profile begin
                NVTX.@range "Benchmark" begin
                    run_benchmark(case, batch_size)
                end
            end
        catch e
            @error "Error for $case x $batch_size: $e"
            break
        end
    end
end


# case = CASES[1]
# batch_size=BATCH_SIZES[1]
# run_benchmark(case, batch_size, warmup=true)
# @pprof run_benchmark(case, batch_size)