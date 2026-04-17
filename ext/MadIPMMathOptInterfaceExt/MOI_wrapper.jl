
Base.@kwdef struct OptimizerConfig{AT, KKT, LS, REG <: MadIPM.AbstractRegularization, STEP <: MadIPM.AbstractStepRule, ALG}
    array_type::AT = Vector{Float64}
    tol::Float64 = 1e-8
    max_iter::Int = 3000
    print_level::MadNLP.LogLevels = MadNLP.INFO
    rethrow_error::Bool = false
    kkt_system::KKT = MadNLP.SparseKKTSystem
    linear_solver::LS = nothing
    regularization::REG = MadIPM.FixedRegularization(1e-10, 1e-10)
    step_rule::STEP = MadIPM.AdaptiveStep(0.99)
    cudss_algorithm::ALG = nothing
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    config::OptimizerConfig
    silent::Bool
    solver::Union{Nothing, MadNLP.AbstractMadNLPSolver}
    qp::Union{Nothing, MadIPM.QuadraticModel}
    stats::Union{
        Nothing,
        MadNLP.MadNLPExecutionStats{Float64, <:AbstractVector{Float64}},
    }
    Optimizer() = new(OptimizerConfig(), false, nothing, nothing, nothing)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "MadIPM"
MOI.is_empty(optimizer::Optimizer) = isnothing(optimizer.solver) && isnothing(optimizer.qp)

function MOI.empty!(optimizer::Optimizer)
    optimizer.solver = nothing
    optimizer.qp = nothing
    optimizer.stats = nothing
    return
end

function _override(config::OptimizerConfig, sym::Symbol, value)
    return OptimizerConfig(;
        (field === sym ? (field => value) : (field => getfield(config, field)) for field in fieldnames(OptimizerConfig))...
    )
end

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    sym = Symbol(param.name)
    hasfield(OptimizerConfig, sym) || throw(ArgumentError("Unsupported MadIPM optimizer attribute `$(param.name)`"))
    optimizer.config = _override(optimizer.config, sym, value)
    return
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    sym = Symbol(param.name)
    hasfield(OptimizerConfig, sym) || throw(ArgumentError("Unsupported MadIPM optimizer attribute `$(param.name)`"))
    return getfield(optimizer.config, sym)
end

###
### MOI.Silent
###

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

###
### MOI.AbstractModelAttribute
###

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{<:Union{VI, SAF, SQF}},
    },
)
    return true
end

function MOI.get(
    model::Optimizer,
    ::MOI.ObjectiveSense,
)
    isnothing(model.qp) && return MOI.MIN_SENSE
    qp = model.qp
    return (qp.meta.minimize) ? MOI.MIN_SENSE : MOI.MAX_SENSE
end


###
### MOI.AbstractVariableAttribute
###

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

###
### `supports_constraint`
###

MOI.supports_constraint(::Optimizer, ::Type{VI}, ::Type{<:_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{SAF}, ::Type{<:_SCALAR_SETS}) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    dest.solver = nothing
    dest.stats = nothing
    dest.qp, index_map = BatchQuadraticModels.qp_model(src)
    if dest.config.array_type != Vector{Float64}
        dest.qp = Adapt.adapt(dest.config.array_type, dest.qp)
    end
    return index_map
end

function MOI.optimize!(model::Optimizer)
    config = model.config
    linear_solver = isnothing(config.linear_solver) ? MadNLP.default_sparse_solver(model.qp) : config.linear_solver
    print_level = model.silent ? MadNLP.ERROR : config.print_level
    model.solver = MadIPM.MPCSolver(
        model.qp;
        tol = config.tol,
        max_iter = config.max_iter,
        print_level = print_level,
        rethrow_error = config.rethrow_error,
        kkt_system = config.kkt_system,
        linear_solver = linear_solver,
        regularization = config.regularization,
        step_rule = config.step_rule,
        cudss_algorithm = config.cudss_algorithm,
    )
    model.stats = _host_stats(MadIPM.solve!(model.solver))
    return
end

function _host_stats(stats::MadNLP.MadNLPExecutionStats{T}) where {T}
    solution = Vector{T}(stats.solution)
    constraints = Vector{T}(stats.constraints)
    multipliers = Vector{T}(stats.multipliers)
    multipliers_L = Vector{T}(stats.multipliers_L)
    multipliers_U = Vector{T}(stats.multipliers_U)
    return MadNLP.MadNLPExecutionStats(
        stats.options,
        stats.status,
        solution,
        stats.objective,
        constraints,
        stats.dual_feas,
        stats.primal_feas,
        multipliers,
        multipliers_L,
        multipliers_U,
        stats.iter,
        stats.counters,
    )
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    isnothing(optimizer.stats) && return 0.0
    return optimizer.stats.counters.total_time
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    isnothing(optimizer.stats) && return string(MOI.OPTIMIZE_NOT_CALLED)
    return string(optimizer.stats.status)
end

struct RawStatus <: MOI.AbstractModelAttribute
    name::Symbol
end

MOI.is_set_by_optimize(::RawStatus) = true

function MOI.get(optimizer::Optimizer, attr::RawStatus)
    isnothing(optimizer.stats) && error("Raw status is only available after optimize! is called.")
    return getfield(optimizer.stats, attr.name)
end

function _termination_status(status::MadNLP.Status)
    status === MadNLP.SOLVE_SUCCEEDED && return MOI.OPTIMAL
    status === MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL && return MOI.ALMOST_OPTIMAL
    status === MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL && return MOI.SLOW_PROGRESS
    status === MadNLP.DIVERGING_ITERATES && return MOI.INFEASIBLE_OR_UNBOUNDED
    status === MadNLP.INFEASIBLE_PROBLEM_DETECTED && return MOI.INFEASIBLE
    status === MadNLP.MAXIMUM_ITERATIONS_EXCEEDED && return MOI.ITERATION_LIMIT
    status === MadNLP.MAXIMUM_WALLTIME_EXCEEDED && return MOI.TIME_LIMIT
    status === MadNLP.INITIAL && return MOI.OPTIMIZE_NOT_CALLED
    status === MadNLP.RESTORATION_FAILED && return MOI.NUMERICAL_ERROR
    status === MadNLP.INVALID_NUMBER_DETECTED && return MOI.INVALID_MODEL
    status === MadNLP.ERROR_IN_STEP_COMPUTATION && return MOI.NUMERICAL_ERROR
    status === MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM && return MOI.INVALID_MODEL
    status === MadNLP.USER_REQUESTED_STOP && return MOI.INTERRUPTED
    status === MadNLP.INTERNAL_ERROR && return MOI.OTHER_ERROR
    status === MadNLP.INVALID_NUMBER_OBJECTIVE && return MOI.INVALID_MODEL
    status === MadNLP.INVALID_NUMBER_GRADIENT && return MOI.INVALID_MODEL
    status === MadNLP.INVALID_NUMBER_CONSTRAINTS && return MOI.INVALID_MODEL
    status === MadNLP.INVALID_NUMBER_JACOBIAN && return MOI.INVALID_MODEL
    status === MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN && return MOI.INVALID_MODEL
    return MOI.OTHER_ERROR
end

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    if isnothing(optimizer.stats)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return _termination_status(optimizer.stats.status)
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.objective
end

function _result_status(optimizer::Optimizer, result_index)
    result_index > MOI.get(optimizer, MOI.ResultCount()) && return MOI.NO_SOLUTION
    term = MOI.get(optimizer, MOI.TerminationStatus())
    term == MOI.OPTIMAL && return MOI.FEASIBLE_POINT
    term == MOI.INFEASIBLE && return MOI.INFEASIBLE_POINT
    return MOI.NO_SOLUTION
end

MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus) = _result_status(optimizer, attr.result_index)
MOI.get(optimizer::Optimizer, attr::MOI.DualStatus) = _result_status(optimizer, attr.result_index)

function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.solution[vi.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:_SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    return MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:_SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    return model.stats.constraints[c.value+1]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.VariableIndex,S},
) where {S<:_SCALAR_SETS}
    MOI.check_result_index_bounds(model, attr)
    col = c.value
    dual = if S <: MOI.LessThan
        -model.stats.multipliers_U[col]
    elseif S <: MOI.GreaterThan
        model.stats.multipliers_L[col]
    else
        model.stats.multipliers_L[col] - model.stats.multipliers_U[col]
    end
    return dual
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S},
) where {S<:_SCALAR_SETS}
    MOI.check_result_index_bounds(model, attr)
    return -model.stats.multipliers[c.value+1]
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = isnothing(optimizer.stats) ? 0 : 1
