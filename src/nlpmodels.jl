# TODO: split up MadNLPSolver similarly with problem/state

function MadNLP.eval_f_wrapper(solver::MPCSolver, x::MadNLP.PrimalVector{T}) where {T}
    problem = solver.problem
    state = solver.state
    nlp = problem.nlp
    MadNLP.@trace(problem.logger, "Evaluating objective.")
    state.cnt.eval_function_time += @elapsed begin
        sense = NLPModels.get_minimize(nlp) ? one(T) : -one(T)
        obj_val = sense * MadNLP._eval_f_wrapper(problem.cb, MadNLP.variable(x))
    end
    state.cnt.obj_cnt += 1
    if state.cnt.obj_cnt == 1 && !MadNLP.is_valid(obj_val)
        throw(MadNLP.InvalidNumberException(:obj))
    end
    return obj_val
end

function MadNLP.eval_grad_f_wrapper!(solver::MPCSolver, f::MadNLP.PrimalVector{T}, x::MadNLP.PrimalVector{T}) where {T}
    problem = solver.problem
    state = solver.state
    nlp = problem.nlp
    MadNLP.@trace(problem.logger, "Evaluating objective gradient.")
    state.cnt.eval_function_time += @elapsed MadNLP._eval_grad_f_wrapper!(problem.cb, MadNLP.variable(x), MadNLP.variable(f))
    if !NLPModels.get_minimize(nlp)
        MadNLP.variable(f) .*= -one(T)
    end
    state.cnt.obj_grad_cnt += 1
    if state.cnt.obj_grad_cnt == 1 && !MadNLP.is_valid(MadNLP.full(f))
        throw(MadNLP.InvalidNumberException(:grad))
    end
    return f
end

function MadNLP.eval_cons_wrapper!(solver::MPCSolver, c::AbstractVector{T}, x::MadNLP.PrimalVector{T}) where {T}
    problem = solver.problem
    state = solver.state
    MadNLP.@trace(problem.logger, "Evaluating constraints.")
    state.cnt.eval_function_time += @elapsed MadNLP._eval_cons_wrapper!(problem.cb, MadNLP.variable(x), c)
    c .-= state.rhs
    state.cnt.con_cnt += 1
    if state.cnt.con_cnt == 1 && !MadNLP.is_valid(c)
        throw(MadNLP.InvalidNumberException(:cons))
    end
    return c
end

function MadNLP.eval_jac_wrapper!(solver::MPCSolver, kkt::MadNLP.AbstractKKTSystem, x::MadNLP.PrimalVector{T}) where {T}
    problem = solver.problem
    state = solver.state
    jac = MadNLP.get_jacobian(kkt)
    MadNLP.@trace(problem.logger, "Evaluating constraint Jacobian.")
    state.cnt.eval_function_time += @elapsed MadNLP._eval_jac_wrapper!(problem.cb, MadNLP.variable(x), jac)
    MadNLP.compress_jacobian!(kkt)
    state.cnt.con_jac_cnt += 1
    if state.cnt.con_jac_cnt == 1 && !MadNLP.is_valid(jac)
        throw(MadNLP.InvalidNumberException(:jac))
    end
    return jac
end

function MadNLP.eval_lag_hess_wrapper!(solver::MPCSolver, kkt::MadNLP.AbstractKKTSystem, x::MadNLP.PrimalVector{T}, l::AbstractVector{T}; is_resto = false) where {T}
    problem = solver.problem
    state = solver.state
    hess = MadNLP.get_hessian(kkt)
    scale = (NLPModels.get_minimize(problem.nlp) ? one(T) : -one(T)) * (is_resto ? zero(T) : one(T))
    MadNLP.@trace(problem.logger, "Evaluating Lagrangian Hessian.")
    state.cnt.eval_function_time += @elapsed MadNLP._eval_lag_hess_wrapper!(problem.cb, MadNLP.variable(x), l, hess; obj_weight = scale)
    MadNLP.compress_hessian!(kkt)
    state.cnt.lag_hess_cnt += 1
    if state.cnt.lag_hess_cnt == 1 && !MadNLP.is_valid(hess)
        throw(MadNLP.InvalidNumberException(:hess))
    end
    return hess
end
