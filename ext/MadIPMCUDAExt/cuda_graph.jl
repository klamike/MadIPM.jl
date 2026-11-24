# Graph mode: record one iteration of the uniform batch MPC solver (mpc_step!
# and the termination criteria) as a CUDA graph and replay it, so that an
# iteration costs one graph launch instead of hundreds of kernel launches.
#
# Graph mode is off by default; the solver option `cuda_graph = true` turns it
# on. It requires a batch linear solver whose refactorization and solves can
# be recorded (MadIPM.graph_capturable; for cuDSS, `cudss_pivoting = false`)
# and a regularization that does not change its host scalars from one
# iteration to the next. Otherwise the solver runs plain iterations.
#
# The iteration is recorded on the current stream, which must be the stream
# the cuDSS handle was built on: cuDSS keeps the stream of its handle, so
# recording on another stream would run cuDSS eagerly instead of recording it.
# The cuDSS calls are not recorded inline with the rest of the iteration:
#
# - pass 1 records the iteration with an empty placeholder node for each cuDSS
#   call, and collects the calls;
# - each cuDSS call is then recorded on its own;
# - pass 2 records the iteration again, with each cuDSS recording added as a
#   child graph where its call is made.
#
# The iteration is recorded again whenever MadIPM.recording_key of the linear
# solver changes (for cuDSS, the ubatch_mask of the active instances).
#
# Memory: only the executable graph is kept during the solve; the recorded
# graphs are destroyed once it is instantiated. The executable still holds
# the device memory of the temporaries allocated during the recording (they
# become allocation nodes, served from the device's graph memory pool, apart
# from the pool plain iterations use). Before a new recording the previous
# executable is destroyed and the graph memory pool trimmed. When a recording
# or a replay runs out of memory, graph mode releases its graph and turns
# itself off for that solve, and the solve goes on with plain iterations. The
# graph is released when the solve ends, so it never holds memory past it.

# A recorded iteration replays the host scalars it was recorded with: the graph
# is only used when no scalar argument of the iteration changes from one
# iteration to the next (AdaptiveRegularization updates its deltas on the host).
_graph_ok(solver) = solver.opt.cuda_graph &&
    MadIPM.graph_capturable(solver.kkt.batch_solver) &&
    (solver.opt.regularization isa MadIPM.FixedRegularization ||
     solver.opt.regularization isa MadIPM.NoRegularization)

const CC = CUDA.CUDACore

# a cuDSS call of the iteration, found in pass 1
struct CudssCall
    call::Function
    graph::CC.CUgraph   # the graph its placeholder was added to
end

mutable struct GraphState
    exec::Union{Nothing, CUDA.CuGraphExec}
    graph::Union{Nothing, CUDA.CuGraph}
    key::Any                # recording_key of the linear solver at recording
    ncaptures::Int
    ninstantiations::Int
    nlaunch::Int
    t_capture::Float64
    t_instantiate::Float64
    t_launch::Float64
    disabled::Bool          # out of memory in graph mode: plain iterations
end
GraphState() = GraphState(nothing, nothing, nothing, 0, 0, 0, 0.0, 0.0, 0.0, false)

const GRAPH_STATES = WeakKeyDict{Any, GraphState}()   # weak: dropped with the solver

"""
    graph_stats(solver)

The graph-mode counters and timings of `solver`, or `nothing` if it never
recorded an iteration.
"""
graph_stats(solver) = get(GRAPH_STATES, solver, nothing)

const _SEGMENTING = Ref(false)        # a recording with separate cuDSS calls is running
const _CUDSS_CALLS = CudssCall[]      # pass 1: the cuDSS calls, in call order
const _CHILDREN = CUDA.CuGraph[]      # pass 2: the cuDSS recordings, in call order
const _NEXT_CHILD = Ref(0)

# Run a cuDSS call of the iteration, or, while the iteration is recorded, add
# its node to the recording (a placeholder in pass 1, its recording in pass 2).
function _cudss_call!(call::Function)
    _SEGMENTING[] || return call()
    st = CUDA.stream()
    status = Ref{CC.CUstreamCaptureStatus}()
    id = Ref{CC.cuuint64_t}()
    g = Ref{CC.CUgraph}()
    deps = Ref{Ptr{CC.CUgraphNode}}(C_NULL)
    n = Ref{Csize_t}(0)
    CC.cuStreamGetCaptureInfo_v3(st, status, id, g, deps, C_NULL, n)
    depv = n[] == 0 ? CC.CUgraphNode[] : copy(unsafe_wrap(Array, deps[], Int(n[])))
    node = Ref{CC.CUgraphNode}()
    if isempty(_CHILDREN)
        CC.cuGraphAddEmptyNode(node, g[], depv, length(depv))
        push!(_CUDSS_CALLS, CudssCall(call, g[]))
    else
        _NEXT_CHILD[] += 1
        child = _CHILDREN[_NEXT_CHILD[]]
        CC.cuGraphAddChildGraphNode(node, g[], depv, length(depv), child.handle)
    end
    CC.cuStreamUpdateCaptureDependencies_v2(st, node, C_NULL, 1, CC.CU_STREAM_SET_CAPTURE_DEPENDENCIES)
    return
end

# run the finalizer now (it destroys the graph and is not run again)
_destroy!(g) = (g === nothing || finalize(g); nothing)

# Destroy the recording of `gs` now (not at the next GC) and give the memory
# of its allocation nodes back to the device.
function _release!(gs::GraphState)
    gs.exec === nothing && gs.graph === nothing && return
    CUDA.synchronize()
    _destroy!(gs.exec); _destroy!(gs.graph)
    gs.exec = nothing; gs.graph = nothing
    CC.cuDeviceGraphMemTrim(CUDA.device().handle)
    return
end

function _plain_iteration!(solver)
    invoke(MadIPM.mpc_step!, Tuple{MadIPM.AbstractBatchMPCSolver}, solver)
    MadIPM.update_termination_criteria!(solver)
    return
end

# Record the iteration and instantiate it. cuDSS allocates its buffers in its
# first factorization, which cannot be recorded: the starting point
# (initialize!) factorizes the KKT system once before the first recording.
function _capture_iteration!(solver)
    st = CUDA.stream()
    gs = get!(GraphState, GRAPH_STATES, solver)
    gs.key = MadIPM.recording_key(solver.kkt.batch_solver)
    record() = CUDA.capture(() -> _plain_iteration!(solver); stream = st)
    # the recording runs the host side of the iteration, which counts it
    k = copy(solver.batch_cnt.k)
    t0 = time()
    # release the previous recording first: its memory is not needed anymore
    _release!(gs)
    g1 = nothing
    _SEGMENTING[] = true
    try
        # pass 1: find the cuDSS calls of the iteration
        empty!(_CUDSS_CALLS); empty!(_CHILDREN)
        g1 = record()
        calls = [c.call for c in _CUDSS_CALLS if c.graph == g1.handle]
        _destroy!(g1); g1 = nothing
        # record each cuDSS call on its own
        _SEGMENTING[] = false
        for c in calls
            push!(_CHILDREN, CUDA.capture(c; stream = st))
        end
        # pass 2: the iteration, with the cuDSS recordings in place of the calls
        _SEGMENTING[] = true
        copyto!(solver.batch_cnt.k, k)
        _NEXT_CHILD[] = 0
        gs.graph = record()
        _NEXT_CHILD[] == length(calls) || error("the iteration made a different number of cuDSS calls")
    finally
        _SEGMENTING[] = false
        # the parent graph holds copies of the cuDSS recordings
        _destroy!(g1); foreach(_destroy!, _CHILDREN)
        empty!(_CHILDREN); empty!(_CUDSS_CALLS)
    end
    gs.ncaptures += 1
    gs.t_capture += time() - t0
    t0 = time()
    gs.exec = CUDA.instantiate(gs.graph)
    _destroy!(gs.graph); gs.graph = nothing   # the executable does not need it
    gs.ninstantiations += 1
    gs.t_instantiate += time() - t0
    return
end

_is_oom(e) = e isa CUDA.OutOfGPUMemoryError ||
    occursin(r"Out of GPU memory|ALLOC_FAILED|OUT_OF_MEMORY"i, sprint(showerror, e))

function _graph_mpc_step!(solver)
    st = CUDA.stream()
    gs = get!(GraphState, GRAPH_STATES, solver)
    ls = solver.kkt.batch_solver
    MadIPM.prepare_active!(ls, MadIPM.active_view(solver.batch_views))
    gs.disabled && return _plain_iteration!(solver)
    k = copy(solver.batch_cnt.k)
    try
        if gs.exec === nothing || gs.key != MadIPM.recording_key(ls)
            # recording runs the iteration's host side, and the replay below
            # runs its device side
            _capture_iteration!(solver)
        else
            MadIPM.increment_k!(solver)
        end
        t0 = time()
        CUDA.launch(gs.exec, st)
        gs.t_launch += time() - t0; gs.nlaunch += 1
    catch e
        # the recording and the memory its replays hold do not always fit
        # where plain iterations do: release them and go on without graphs
        # (a failed recording or launch ran nothing on the device)
        _is_oom(e) || rethrow()
        gs.disabled = true
        _release!(gs)
        GC.gc(true); CUDA.reclaim()
        copyto!(solver.batch_cnt.k, k)
        return _plain_iteration!(solver)
    end
    return
end

function MadIPM.mpc_step!(
    solver::MadIPM.UniformBatchMPCSolver{T, MT},
) where {T, MT <: CuMatrix{T}}
    _graph_ok(solver) || return invoke(MadIPM.mpc_step!, Tuple{MadIPM.AbstractBatchMPCSolver}, solver)
    return _graph_mpc_step!(solver)
end

# graph mode: the termination criteria are computed inside the graph, at the
# end of each iteration, so the loop only reads the termination flags back.
# The graph is released when the solve ends (it is recorded again by the next
# solve), so that no graph memory outlives the solve. An out-of-memory error
# anywhere in the loop releases the graph and finishes the solve with plain
# iterations from the current iterate.
function MadIPM.mpc!(solver::MadIPM.UniformBatchMPCSolver{T, MT}) where {T, MT <: CuMatrix{T}}
    _graph_ok(solver) || return invoke(MadIPM.mpc!, Tuple{MadIPM.AbstractBatchMPCSolver}, solver)
    gs = get!(GraphState, GRAPH_STATES, solver)
    gs.disabled = false
    try
        MadIPM.update_termination_criteria!(solver)
        while true
            MadNLP.print_iter(solver)
            if MadIPM.update_termination_status!(solver)
                MadIPM.update_active_set!(solver)
                MadIPM.active_batch_size(solver) == 0 && return
                MadIPM._update_active_mask!(solver)
            end
            _graph_mpc_step!(solver)
        end
    catch e
        _is_oom(e) || rethrow()
        gs.disabled = true
        _release!(gs)
        GC.gc(true); CUDA.reclaim()
        return invoke(MadIPM.mpc!, Tuple{MadIPM.AbstractBatchMPCSolver}, solver)
    finally
        _release!(gs)
    end
end
