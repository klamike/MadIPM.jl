# ============================================================================
# Netlib LP batch-solve benchmark for MadIPM.
#
# Each batch instance is the same Netlib LP but with its objective rotated by
# a different angle in a random 2D coordinate plane: `c_i = R(θ_i) · c`. All
# `(A, b, bounds)` are shared; only `c` varies across the batch.
#
# We compare `UniformBatchMPCSolver` (one batch call) against solving each
# rotation separately with the scalar `MPCSolver`, as a wall-clock speedup.
# ============================================================================

using LinearAlgebra
using Printf
using Random
using SparseArrays

using NLPModels
using BatchQuadraticModels
using MadIPM
using MadNLP
using MathOptInterface
using MathOptBenchmarkInstances
using QPSReader: QPSData

using Adapt
using CUDA
using MadNLPGPU

# Commercial / open-source barrier baselines via BQMSolvers threaded batch.
# Gurobi needs a license to be visible in this process — on the cluster,
# run `module load gurobi` before `julia` so `GRB_LICENSE_FILE` is set.
using BQMSolvers
using Gurobi
using HiGHS

const MOI = MathOptInterface
const SAF = MOI.ScalarAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const VI = MOI.VariableIndex

const COLLECTIONS = Dict(
    :netlib     => MathOptBenchmarkInstances.Netlib,
    :miplib     => MathOptBenchmarkInstances.MIPLIB2017,
    :mittelmann => MathOptBenchmarkInstances.MittelmannLP,
    :maros      => MathOptBenchmarkInstances.MarosMeszaros,
)

_resolve_collection(c::Symbol) = COLLECTIONS[c]
_resolve_collection(c::MathOptBenchmarkInstances.Dataset) = c

function load_instance(collection, name::AbstractString)
    problem, _ = MathOptBenchmarkInstances.read_instance(collection, name)
    return problem::QPSData
end
# Back-compat alias — existing helper scripts still call `load_netlib`.
load_netlib(name::AbstractString) = load_instance(MathOptBenchmarkInstances.Netlib, name)

function _miplib_mps_path(name::AbstractString)
    mps_path = joinpath(MathOptBenchmarkInstances.MPS_SCRATCH, "miplib2017",
                        "$(lowercase(name)).mps")
    ispath(mps_path) && return mps_path
    try
        MathOptBenchmarkInstances.read_instance(MathOptBenchmarkInstances.MIPLIB2017, String(name))
    catch
        # `read_instance` may fail while parsing unsupported MIP sections after
        # it has decompressed the source MPS into the scratch directory.
    end
    ispath(mps_path) || error("MIPLIB instance file is not available: $mps_path")
    return mps_path
end

function _add_bound!(model, x::VI, l::Float64, u::Float64)
    if isfinite(l) && isfinite(u)
        l == u ? MOI.add_constraint(model, x, MOI.EqualTo(l)) :
                 MOI.add_constraint(model, x, MOI.Interval(l, u))
    elseif isfinite(l)
        MOI.add_constraint(model, x, MOI.GreaterThan(l))
    elseif isfinite(u)
        MOI.add_constraint(model, x, MOI.LessThan(u))
    end
end

function _copy_objective!(dest, src, index_map)
    MOI.set(dest, MOI.ObjectiveSense(), MOI.get(src, MOI.ObjectiveSense()))
    F = MOI.get(src, MOI.ObjectiveFunctionType())
    if F <: VI || F <: SAF || F <: SQF
        f = MOI.get(src, MOI.ObjectiveFunction{F}())
        MOI.set(dest, MOI.ObjectiveFunction{F}(), MOI.Utilities.map_indices(index_map, f))
    else
        error("unsupported objective function type in relaxation: $F")
    end
end

function _moi_relaxation_model(mps_path::AbstractString)
    src = MOI.FileFormats.MPS.Model()
    MOI.read_from_file(src, mps_path)

    dest = MOI.Utilities.Model{Float64}()
    index_map = MOI.Utilities.IndexMap()
    for x in MOI.get(src, MOI.ListOfVariableIndices())
        y = MOI.add_variable(dest)
        index_map[x] = y
    end
    for x in MOI.get(src, MOI.ListOfVariableIndices())
        l, u = MOI.Utilities.get_bounds(src, Float64, x)
        _add_bound!(dest, index_map[x], l, u)
    end
    _copy_objective!(dest, src, index_map)

    dropped = Dict{String, Int}()
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if F <: VI
            continue  # variable bounds and integrality were handled above.
        elseif F <: SAF && S <: Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64},
                                      MOI.EqualTo{Float64}, MOI.Interval{Float64}}
            for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
                f = MOI.get(src, MOI.ConstraintFunction(), ci)
                s = MOI.get(src, MOI.ConstraintSet(), ci)
                MOI.add_constraint(dest, MOI.Utilities.map_indices(index_map, f), s)
            end
        else
            key = "$F-in-$S"
            dropped[key] = get(dropped, key, 0) +
                length(MOI.get(src, MOI.ListOfConstraintIndices{F, S}()))
        end
    end
    if !isempty(dropped)
        summary = join(("$key ($count)" for (key, count) in sort!(collect(dropped))), "; ")
        @info "Dropped non-LP relaxation constraints from $(basename(mps_path)): $summary"
    end
    return dest
end

function load_qp(collection, name::AbstractString)
    try
        return qps_to_qp(load_instance(collection, name))
    catch err
        collection == MathOptBenchmarkInstances.MIPLIB2017 || rethrow()
        mps_path = _miplib_mps_path(name)
        model = _moi_relaxation_model(mps_path)
        qp, _ = BatchQuadraticModels.qp_model(model)
        return qp
    end
end

# Build a `QuadraticModels.QuadraticModel`-compatible scalar QP from a QPSData.
# We stick to the `BatchQuadraticModels` export path so BQM owns the format.
function qps_to_qp(qps::QPSData)
    n = length(qps.c)
    m = length(qps.lcon)
    A = sparse(Vector{Int}(qps.arows), Vector{Int}(qps.acols),
               Vector{Float64}(qps.avals), m, n)
    Q = isempty(qps.qrows) ? sparse(Int[], Int[], Float64[], n, n) :
        sparse(Vector{Int}(qps.qrows), Vector{Int}(qps.qcols),
               Vector{Float64}(qps.qvals), n, n)
    data = BatchQuadraticModels.QPData(A, Vector{Float64}(qps.c), Q;
        c0   = qps.c0,
        lvar = Vector{Float64}(qps.lvar),
        uvar = Vector{Float64}(qps.uvar),
        lcon = Vector{Float64}(qps.lcon),
        ucon = Vector{Float64}(qps.ucon),
    )
    return BatchQuadraticModels.QuadraticModel(data)
end

# `R(θ)` acting in the (i,j) plane: `c_new[i] = cos θ·c[i] - sin θ·c[j]`,
# `c_new[j] = sin θ·c[i] + cos θ·c[j]`. Returns an `(n, bs)` matrix.
function rotated_costs(c::AbstractVector{T}, thetas::AbstractVector;
                        plane::Tuple{Int,Int}) where {T}
    n  = length(c)
    bs = length(thetas)
    i, j = plane
    C = Matrix{T}(undef, n, bs)
    for b in 1:bs
        cθ = cos(thetas[b]); sθ = sin(thetas[b])
        @inbounds for k in 1:n
            C[k, b] = c[k]
        end
        ci = c[i]; cj = c[j]
        C[i, b] = cθ * ci - sθ * cj
        C[j, b] = sθ * ci + cθ * cj
    end
    return C
end

function random_plane(n::Int; seed::Int = 42)
    rng = MersenneTwister(seed)
    i = rand(rng, 1:n)
    j = rand(rng, 1:n)
    while j == i
        j = rand(rng, 1:n)
    end
    return (i, j)
end

function build_batch(qp, thetas::AbstractVector; plane)
    bs   = length(thetas)
    c_mat = rotated_costs(qp.data.c, thetas; plane)
    # Replicate bounds and c0; A and Q are shared across the batch.
    n = qp.meta.nvar; m = qp.meta.ncon
    lvar = repeat(qp.meta.lvar, 1, bs)
    uvar = repeat(qp.meta.uvar, 1, bs)
    lcon = repeat(qp.meta.lcon, 1, bs)
    ucon = repeat(qp.meta.ucon, 1, bs)
    c0   = fill(qp.data.c0[], bs)
    return BatchQuadraticModels.ObjRHSBatchQuadraticModel(qp, bs;
        c = c_mat, c0 = c0, lvar = lvar, uvar = uvar, lcon = lcon, ucon = ucon,
    )
end

_ok(status) = status == MadNLP.SOLVE_SUCCEEDED || status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL

function solve_scalar!(qp; c::AbstractVector, opts, linear_solver)
    qp.data.c .= c
    return MadIPM.madipm(qp; linear_solver, opts...)
end

# GPU scalar path: `c_cols_gpu` is an `(n, bs)` device matrix of rotations.
# Per column we broadcast-copy into `qp_gpu.data.c` and resolve — no
# CPU↔GPU transfer inside the timed loop.
function solve_scalar_gpu!(qp_gpu, c_cols_gpu::AbstractMatrix, col::Int;
                            opts, linear_solver)
    qp_gpu.data.c .= @view c_cols_gpu[:, col]
    return MadIPM.madipm(qp_gpu; linear_solver, opts...)
end

function benchmark_instance(name::AbstractString; bs::Int = 8,
                             max_theta::Float64 = 0.1,
                             collection = MathOptBenchmarkInstances.Netlib)
    qp   = load_qp(collection, name)
    n, m = qp.meta.nvar, qp.meta.ncon
    plane  = random_plane(n)
    thetas = range(-max_theta, max_theta; length = bs)
    c_mat  = rotated_costs(qp.data.c, thetas; plane)  # CPU (n, bs) matrix

    bqp_cpu = build_batch(qp, thetas; plane)

    opts = (; kkt_system = MadNLP.ScaledSparseKKTSystem, cudss_ir = 0, max_iter = 1000, tol = 1e-6, print_level = MadNLP.ERROR)

    CUDA.functional() || error("benchmark requires a functional CUDA device")
    qp_gpu     = adapt(CuArray, qp)
    bqp_gpu    = adapt(CuArray, bqp_cpu)
    ls_gpu     = MadNLPGPU.CUDSSSolver
    ls_cpu     = MadNLP.UmfpackSolver
    base_c_gpu = copy(qp_gpu.data.c)
    base_c_cpu = copy(qp.data.c)
    c_cols_gpu = CuArray(c_mat)

    # Warm-up all three paths to eat one-time JIT / first-kernel costs.
    let
        warm = MadIPM.UniformBatchMPCSolver(bqp_gpu; linear_solver = ls_gpu, opts...)
        MadIPM.solve!(warm); CUDA.synchronize()
        solve_scalar_gpu!(qp_gpu, c_cols_gpu, 1; opts, linear_solver = ls_gpu)
        CUDA.synchronize()
        qp_gpu.data.c .= base_c_gpu
        solve_scalar!(qp; c = view(c_mat, :, 1), opts, linear_solver = ls_cpu)
        qp.data.c .= base_c_cpu
    end

    # --- GPU batch ---
    GC.gc(true); GC.gc(true); CUDA.reclaim()
    gpu_batch_stats = nothing
    t_gpu_batch = CUDA.@elapsed begin
        bs_solver = MadIPM.UniformBatchMPCSolver(bqp_gpu;
            linear_solver = ls_gpu, opts...)
        gpu_batch_stats = MadIPM.solve!(bs_solver)
    end

    # --- GPU scalar (sequential) ---
    GC.gc(true); GC.gc(true); CUDA.reclaim()
    gpu_scalar_stats = Vector{Any}(undef, bs)
    t_gpu_scalar = CUDA.@elapsed for b in 1:bs
        gpu_scalar_stats[b] = solve_scalar_gpu!(qp_gpu, c_cols_gpu, b;
            opts, linear_solver = ls_gpu)
    end
    qp_gpu.data.c .= base_c_gpu

    # --- CPU scalar (sequential) ---
    GC.gc(true)
    cpu_scalar_stats = Vector{Any}(undef, bs)
    t_cpu_scalar = @elapsed for b in 1:bs
        cpu_scalar_stats[b] = solve_scalar!(qp;
            c = view(c_mat, :, b), opts, linear_solver = ls_cpu)
    end
    qp.data.c .= base_c_cpu

    # --- GPU scalar (threaded) ---
    # Per-task qp_gpu copy so each thread mutates its own `data.c`. Indexing
    # by `b` (not threadid) avoids the interactive-pool threadid > nthreads
    # pitfall.
    GC.gc(true); GC.gc(true); CUDA.reclaim()
    qp_gpu_pool = [adapt(CuArray, qp) for _ in 1:bs]
    gpu_thrd_stats = Vector{Any}(undef, bs)
    t_gpu_thrd = CUDA.@elapsed begin
        Threads.@threads for b in 1:bs
            gpu_thrd_stats[b] = solve_scalar_gpu!(qp_gpu_pool[b], c_cols_gpu, b;
                opts, linear_solver = ls_gpu)
        end
        CUDA.synchronize()
    end
    qp_gpu_pool = nothing

    # --- CPU scalar (threaded) ---
    GC.gc(true)
    qp_cpu_pool = [deepcopy(qp) for _ in 1:bs]
    cpu_thrd_stats = Vector{Any}(undef, bs)
    t_cpu_thrd = @elapsed Threads.@threads for b in 1:bs
        cpu_thrd_stats[b] = solve_scalar!(qp_cpu_pool[b];
            c = view(c_mat, :, b), opts, linear_solver = ls_cpu)
    end
    qp_cpu_pool = nothing

    # --- Gurobi barrier, threaded batch via BQMSolvers ---
    # `Method=2` = barrier; `Crossover=0` skips the simplex polish; `Presolve=0`
    # disables presolve so we compare raw barrier iteration cost. `Threads=1`
    # keeps each solver single-threaded so outer Julia threads aren't
    # oversubscribing.
    GC.gc(true)
    t_gurobi_thrd = @elapsed gurobi_thrd_stats = BQMSolvers.gurobi(bqp_cpu;
        Method = 2, Crossover = 0, Presolve = 0, Threads = 1, OutputFlag = 0)

    # --- HiGHS barrier (IPM), threaded batch via BQMSolvers ---
    # `solver = "ipm"` picks HiGHS's interior-point method; `run_crossover
    # = "off"` skips the simplex polish; `presolve = "off"` disables HiGHS's
    # presolve so we measure raw barrier iteration cost.
    GC.gc(true)
    t_highs_thrd = @elapsed highs_thrd_stats = BQMSolvers.highs(bqp_cpu;
        solver = "ipm", run_crossover = "off", presolve = "off",
        threads = 1, output_flag = false)

    # --- Stats / statuses ---
    gpu_batch_ok  = all(_ok, gpu_batch_stats.status)
    gpu_scalar_ok = all(s -> _ok(s.status), gpu_scalar_stats)
    cpu_scalar_ok = all(s -> _ok(s.status), cpu_scalar_stats)
    gpu_thrd_ok   = all(s -> _ok(s.status), gpu_thrd_stats)
    cpu_thrd_ok   = all(s -> _ok(s.status), cpu_thrd_stats)
    gurobi_thrd_ok = all(s -> s.status == :acceptable, gurobi_thrd_stats)
    highs_thrd_ok  = all(s -> s.status == :acceptable, highs_thrd_stats)

    _reason(vec_stats) = let idx = findfirst(s -> !_ok(s.status), vec_stats)
        idx === nothing ? MadNLP.SOLVE_SUCCEEDED : vec_stats[idx].status
    end
    gpu_batch_reason  = gpu_batch_ok ? MadNLP.SOLVE_SUCCEEDED :
        gpu_batch_stats.status[findfirst(!_ok, gpu_batch_stats.status)]
    gpu_scalar_reason = _reason(gpu_scalar_stats)
    cpu_scalar_reason = _reason(cpu_scalar_stats)
    gpu_thrd_reason   = _reason(gpu_thrd_stats)
    cpu_thrd_reason   = _reason(cpu_thrd_stats)
    _reason_sym(vec_stats) = let idx = findfirst(s -> s.status != :acceptable, vec_stats)
        idx === nothing ? :acceptable : vec_stats[idx].status
    end
    gurobi_thrd_reason = _reason_sym(gurobi_thrd_stats)
    highs_thrd_reason  = _reason_sym(highs_thrd_stats)

    t_gpu_batch_r  = gpu_batch_ok  ? t_gpu_batch  : Inf
    t_gpu_scalar_r = gpu_scalar_ok ? t_gpu_scalar : Inf
    t_cpu_scalar_r = cpu_scalar_ok ? t_cpu_scalar : Inf
    t_gpu_thrd_r   = gpu_thrd_ok   ? t_gpu_thrd   : Inf
    t_cpu_thrd_r   = cpu_thrd_ok   ? t_cpu_thrd   : Inf
    t_gurobi_thrd_r = gurobi_thrd_ok ? t_gurobi_thrd : Inf
    t_highs_thrd_r  = highs_thrd_ok  ? t_highs_thrd  : Inf
    _sp(a, b) = isinf(a) || isinf(b) ? NaN : a / b
    speedup_vs_gpu_scalar  = _sp(t_gpu_scalar_r,   t_gpu_batch_r)
    speedup_vs_cpu_scalar  = _sp(t_cpu_scalar_r,   t_gpu_batch_r)
    speedup_vs_gpu_thrd    = _sp(t_gpu_thrd_r,     t_gpu_batch_r)
    speedup_vs_cpu_thrd    = _sp(t_cpu_thrd_r,     t_gpu_batch_r)
    speedup_vs_gurobi_thrd = _sp(t_gurobi_thrd_r,  t_gpu_batch_r)
    speedup_vs_highs_thrd  = _sp(t_highs_thrd_r,   t_gpu_batch_r)

    return (; name, n, m, bs,
              t_gpu_batch = t_gpu_batch_r, t_gpu_scalar = t_gpu_scalar_r, t_cpu_scalar = t_cpu_scalar_r,
              t_gpu_thrd = t_gpu_thrd_r, t_cpu_thrd = t_cpu_thrd_r,
              t_gurobi_thrd = t_gurobi_thrd_r, t_highs_thrd = t_highs_thrd_r,
              speedup_vs_gpu_scalar, speedup_vs_cpu_scalar, speedup_vs_gpu_thrd, speedup_vs_cpu_thrd,
              speedup_vs_gurobi_thrd, speedup_vs_highs_thrd,
              gpu_batch_reason, gpu_scalar_reason, cpu_scalar_reason, gpu_thrd_reason, cpu_thrd_reason,
              gurobi_thrd_reason, highs_thrd_reason)
end

# ---------- driver ----------

# Cheap size probe: read the QPSData once, report (n, m). We reuse the
# cached file on subsequent `read_instance` calls so the full bench just
# re-reads what's already on disk.
function _skip_reason(err)
    msg = sprint(showerror, err)
    if occursin("No such file or directory", msg)
        return "missing instance file"
    elseif occursin("Un-recognized section header: INDICATORS", msg)
        return "unsupported MPS section: INDICATORS"
    elseif occursin("Un-recognized section header: LAZYCONS", msg)
        return "unsupported MPS section: LAZYCONS"
    else
        return first(split(msg, '\n'))
    end
end

function _instance_sizes(collection, names)
    sizes = Tuple{String, Int, Int}[]
    skipped = Dict{String, Int}()
    for name in names
        try
            qp = load_qp(collection, name)
            push!(sizes, (name, qp.meta.nvar, qp.meta.ncon))
        catch err
            reason = _skip_reason(err)
            skipped[reason] = get(skipped, reason, 0) + 1
        end
    end
    if !isempty(skipped)
        summary = join(("$reason ($count)" for (reason, count) in sort!(collect(skipped))), "; ")
        @info "Skipped $(sum(values(skipped))) instance(s) during size probe: $summary"
    end
    return sort!(sizes; by = t -> t[2] + t[3])
end

# Back-compat alias.
_netlib_sizes(names) = _instance_sizes(MathOptBenchmarkInstances.Netlib, names)

function main(; bs = 8, max_theta = 0.1,
                collection = :netlib,
                max_size::Int = typemax(Int), limit::Int = typemax(Int))
    col       = _resolve_collection(collection)
    println("# MadIPM × $(nameof(typeof(col))) / $(col) rotated-objective batch benchmark (bs = $bs, max_size = $max_size, threads = $(Threads.nthreads()))")
    flush(stdout)
    all_names = MathOptBenchmarkInstances.list_instances(col)
    sorted    = _instance_sizes(col, all_names)
    filtered  = Iterators.take((t for t in sorted if (t[2] + t[3]) <= max_size), limit)

    @printf "%-14s %6s %6s %3s %11s %11s %11s %11s %11s %14s %14s %7s %7s %7s %7s %7s %7s  %-22s %-22s %-22s %-22s %-22s %-22s %-22s\n" "name" "n" "m" "bs" "gpu_batch[s]" "gpu_scal[s]" "cpu_scal[s]" "gpu_thrd[s]" "cpu_thrd[s]" "gurobi_thrd[s]" "highs_thrd[s]" "sp_gpu" "sp_cpu" "sp_gpuT" "sp_cpuT" "sp_grbT" "sp_hgsT" "gpu_batch_status" "gpu_scal_status" "cpu_scal_status" "gpu_thrd_status" "cpu_thrd_status" "gurobi_thrd_status" "highs_thrd_status"
    for (name, n, m) in filtered
        try
            r = benchmark_instance(name; bs, max_theta, collection = col)
            @printf "%-14s %6d %6d %3d %11.3f %11.3f %11.3f %11.3f %11.3f %14.3f %14.3f %6.2fx %6.2fx %6.2fx %6.2fx %6.2fx %6.2fx  %-22s %-22s %-22s %-22s %-22s %-22s %-22s\n" r.name r.n r.m r.bs r.t_gpu_batch r.t_gpu_scalar r.t_cpu_scalar r.t_gpu_thrd r.t_cpu_thrd r.t_gurobi_thrd r.t_highs_thrd r.speedup_vs_gpu_scalar r.speedup_vs_cpu_scalar r.speedup_vs_gpu_thrd r.speedup_vs_cpu_thrd r.speedup_vs_gurobi_thrd r.speedup_vs_highs_thrd string(r.gpu_batch_reason) string(r.gpu_scalar_reason) string(r.cpu_scalar_reason) string(r.gpu_thrd_reason) string(r.cpu_thrd_reason) string(r.gurobi_thrd_reason) string(r.highs_thrd_reason)
            flush(stdout)
        catch err
            @printf "%-14s FAIL\n" name
            showerror(stdout, err, catch_backtrace()); println(); flush(stdout)
        end
    end
    return nothing
end

function _parse_cli(args)
    opts = Dict{Symbol, Any}()
    for arg in args
        if startswith(arg, "--collection=")
            opts[:collection] = Symbol(last(split(arg, '='; limit = 2)))
        elseif startswith(arg, "--bs=")
            opts[:bs] = parse(Int, last(split(arg, '='; limit = 2)))
        elseif startswith(arg, "--max-theta=")
            opts[:max_theta] = parse(Float64, last(split(arg, '='; limit = 2)))
        elseif startswith(arg, "--max-size=")
            opts[:max_size] = parse(Int, last(split(arg, '='; limit = 2)))
        elseif startswith(arg, "--limit=")
            opts[:limit] = parse(Int, last(split(arg, '='; limit = 2)))
        else
            error("unknown argument: $arg")
        end
    end
    return opts
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; _parse_cli(ARGS)...)
end
