# ============================================================================
# `MadNLP.eval_*_wrapper[!]` specializations for `MPCSolver`.
#
# Each wrapper: traces, accumulates `eval_function_time`, forwards to the
# callback, flips sign on max-sense problems (internal math is always min),
# validates on first call, and bumps the counter.
# ============================================================================

macro _check_nan_once(cnt_field, tag, val)
    quote
        if $(esc(:state)).cnt.$(cnt_field) == 1 && !MadNLP.is_valid($(esc(val)))
            throw(MadNLP.InvalidNumberException($(QuoteNode(tag))))
        end
    end
end

function MadNLP.eval_f_wrapper(solver::MPCSolver,
                               x::MadNLP.PrimalVector{T}) where {T}
    problem, state = solver.problem, solver.state
    sense = NLPModels.get_minimize(problem.nlp) ? one(T) : -one(T)
    MadNLP.@trace(problem.logger, "Evaluating objective.")

    state.cnt.eval_function_time += @elapsed begin
        obj_val = sense * MadNLP._eval_f_wrapper(problem.cb, MadNLP.variable(x))
    end
    state.cnt.obj_cnt += 1
    @_check_nan_once(obj_cnt, :obj, obj_val)
    return obj_val
end

function MadNLP.eval_grad_f_wrapper!(solver::MPCSolver,
                                      f::MadNLP.PrimalVector{T},
                                      x::MadNLP.PrimalVector{T}) where {T}
    problem, state = solver.problem, solver.state
    MadNLP.@trace(problem.logger, "Evaluating objective gradient.")

    state.cnt.eval_function_time += @elapsed MadNLP._eval_grad_f_wrapper!(
        problem.cb, MadNLP.variable(x), MadNLP.variable(f))
    NLPModels.get_minimize(problem.nlp) || (MadNLP.variable(f) .*= -one(T))
    state.cnt.obj_grad_cnt += 1
    @_check_nan_once(obj_grad_cnt, :grad, MadNLP.full(f))
    return f
end

function MadNLP.eval_cons_wrapper!(solver::MPCSolver,
                                    c::AbstractVector{T},
                                    x::MadNLP.PrimalVector{T}) where {T}
    problem, state = solver.problem, solver.state
    MadNLP.@trace(problem.logger, "Evaluating constraints.")

    state.cnt.eval_function_time += @elapsed MadNLP._eval_cons_wrapper!(
        problem.cb, MadNLP.variable(x), c)
    c .-= state.rhs
    state.cnt.con_cnt += 1
    @_check_nan_once(con_cnt, :cons, c)
    return c
end

function MadNLP.eval_jac_wrapper!(solver::MPCSolver,
                                   kkt::MadNLP.AbstractKKTSystem,
                                   x::MadNLP.PrimalVector{T}) where {T}
    problem, state = solver.problem, solver.state
    jac = MadNLP.get_jacobian(kkt)
    MadNLP.@trace(problem.logger, "Evaluating constraint Jacobian.")

    state.cnt.eval_function_time += @elapsed MadNLP._eval_jac_wrapper!(
        problem.cb, MadNLP.variable(x), jac)
    MadNLP.compress_jacobian!(kkt)
    state.cnt.con_jac_cnt += 1
    @_check_nan_once(con_jac_cnt, :jac, jac)
    return jac
end

function MadNLP.eval_lag_hess_wrapper!(solver::MPCSolver,
                                        kkt::MadNLP.AbstractKKTSystem,
                                        x::MadNLP.PrimalVector{T},
                                        l::AbstractVector{T};
                                        is_resto = false) where {T}
    problem, state = solver.problem, solver.state
    hess = MadNLP.get_hessian(kkt)
    scale = (NLPModels.get_minimize(problem.nlp) ? one(T) : -one(T)) *
            (is_resto ? zero(T) : one(T))
    MadNLP.@trace(problem.logger, "Evaluating Lagrangian Hessian.")

    state.cnt.eval_function_time += @elapsed MadNLP._eval_lag_hess_wrapper!(
        problem.cb, MadNLP.variable(x), l, hess; obj_weight = scale)
    MadNLP.compress_hessian!(kkt)
    state.cnt.lag_hess_cnt += 1
    @_check_nan_once(lag_hess_cnt, :hess, hess)
    return hess
end
