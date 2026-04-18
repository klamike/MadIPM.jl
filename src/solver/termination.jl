# Termination criteria.
# Stays specialized: scalar checks scalar status; batch evaluates the
# branching tree on device into a (1, bs) status code matrix and aggregates.

# ---------- scalar ----------

function update_termination_criteria!(solver::MPCSolver{T}) where {T}
    problem = solver.problem
    state = solver.state
    dobj = -dot(state.y, state.rhs)
    state.inf_pr = MadNLP.get_inf_pr(state.c) / max(one(T), state.norm_b)
    state.inf_du = norm(MadNLP.primal(state.f) .+ state.jacl .- MadNLP.full(state.zl), Inf) / max(one(T), state.norm_c)
    state.inf_compl = _xz_sum(solver) / max(one(T), state.norm_c)
    state.best_complementarity = min(state.best_complementarity, state.inf_compl)

    if max(state.inf_pr, state.inf_du, state.inf_compl) <= problem.opt.tol
        state.status = MadNLP.SOLVE_SUCCEEDED
    elseif ((state.inf_compl > problem.opt.divergence_tol * state.best_complementarity) &&
            (dobj > max(problem.opt.divergence_scale * abs(state.obj_val), one(T))))
        state.status = MadNLP.INFEASIBLE_PROBLEM_DETECTED
    elseif state.obj_val < -problem.opt.divergence_tol * max(problem.opt.divergence_scale * abs(dobj), one(T))
        state.status = MadNLP.DIVERGING_ITERATES
    elseif state.cnt.k >= problem.opt.max_iter
        state.status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    elseif time() - state.cnt.start_time >= problem.opt.max_wall_time
        state.status = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
    end
    return
end

# ---------- batch ----------

function compute_term_gpu!(ws::UniformBatchWorkspace{T}, opt) where T
    ds = T(opt.divergence_scale)
    tol = T(opt.tol)
    div_tol = T(opt.divergence_tol)
    Int_ERROR = Int(MadNLP.INTERNAL_ERROR)
    Int_SOLVED = Int(MadNLP.SOLVE_SUCCEEDED)
    Int_INFEASIBLE = Int(MadNLP.INFEASIBLE_PROBLEM_DETECTED)
    Int_DIVERGING = Int(MadNLP.DIVERGING_ITERATES)
    Int_REGULAR = Int(MadNLP.REGULAR)
    @. ws._term_gpu = ifelse(
        ws._ls_error > zero(Int32),
        Int_ERROR,
        ifelse(
            max(ws.inf_pr, ws.inf_du, ws.inf_compl) <= tol,
            Int_SOLVED,
            ifelse(
                (ws.inf_compl > div_tol * ws.best_complementarity) &
                (ws.dual_obj > max(ds * abs(ws.obj_val), one(T))),
                Int_INFEASIBLE,
                ifelse(
                    ws.obj_val < -(div_tol * max(ds * abs(ws.dual_obj), one(T))),
                    Int_DIVERGING,
                    Int_REGULAR,
                ),
            ),
        ),
    )
    minimum!(ws._any_nonregular_gpu, ws._term_gpu)
end

function update_termination_criteria!(batch_solver::AbstractBatchMPCSolver{T}) where T
    problem = batch_solver.problem
    state = batch_solver.state
    ws = state.workspace
    opt = problem.opt
    x, xl, xu = state.x, state.xl, state.xu
    zl, zu = state.zl, state.zu
    nlb, nub = state.d.nlb, state.d.nub

    get_inf_pr!(ws.inf_pr, MadNLP.full(state.c))
    @. ws.inf_pr /= max(one(T), ws.norm_b)

    get_inf_du!(ws.inf_du, MadNLP.full(state.f), MadNLP.full(zl),
                MadNLP.full(zu), MadNLP.full(state.jacl))
    @. ws.inf_du /= max(one(T), ws.norm_c)

    get_inf_compl!(ws.inf_compl, x, xl, zl, xu, zu,
        ws.sum_lb, ws.sum_ub, nlb, nub)
    @. ws.inf_compl /= max(one(T), ws.norm_c)
    @. ws.best_complementarity = min(ws.best_complementarity, ws.inf_compl)

    dual_objective!(ws.dual_obj, MadNLP.full(state.y), MadNLP.full(state.rhs),
        lower(zl), lower(xl), upper(zu), upper(xu),
        ws.sum_lb, ws.sum_ub, nlb, nub)

    compute_term_gpu!(ws, opt)
    return
end

function update_termination_status!(batch_solver::AbstractBatchMPCSolver)
    problem = batch_solver.problem
    state = batch_solver.state
    ws = state.workspace
    opt = problem.opt
    bcnt = state.cnt
    bs = problem.batch_size
    Int_REGULAR = Int64(Int(MadNLP.REGULAR))

    walltime_hit = time() - bcnt.start_time >= opt.max_wall_time
    max_iter_hit = walltime_hit ? false :
        any(ws.status[i] == MadNLP.REGULAR && bcnt.k[i] >= opt.max_iter for i in 1:bs)

    if !walltime_hit && !max_iter_hit
        copyto!(ws._any_nonregular_cpu, ws._any_nonregular_gpu)
        ws._any_nonregular_cpu[1] == Int_REGULAR && return false
    end

    copyto!(ws._term_cpu, ws._term_gpu)
    @inbounds for i in 1:bs
        ws.status[i] != MadNLP.REGULAR && continue
        code = MadNLP.Status(ws._term_cpu[i])
        if code != MadNLP.REGULAR
            ws.status[i] = code
        elseif bcnt.k[i] >= opt.max_iter
            ws.status[i] = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        elseif walltime_hit
            ws.status[i] = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
        end
    end
    return true
end


function dual_objective!(dual_obj, y_vals, rhs_vals, zl_r, xl_r, zu_r, xu_r,
                         sum_lb, sum_ub, nlb, nub)
    T = eltype(dual_obj)
    batch_mapreduce!(*, +, zero(T), dual_obj, y_vals, rhs_vals)
    dual_obj .*= -one(T)
    if nlb > 0
        batch_mapreduce!(*, +, zero(T), sum_lb, zl_r, xl_r)
        dual_obj .+= sum_lb
    end
    if nub > 0
        batch_mapreduce!(*, +, zero(T), sum_ub, zu_r, xu_r)
        dual_obj .-= sum_ub
    end
    return dual_obj
end
