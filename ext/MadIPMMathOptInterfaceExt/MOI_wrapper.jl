# ============================================================================
# MOI `Optimizer` wrapping `MadIPM.MPCSolver`.
# ============================================================================

Base.@kwdef struct OptimizerConfig{AT, KKT, LS, REG, STEP, ALG}
    array_type::AT          = Vector{Float64}
    tol::Float64            = 1e-8
    max_iter::Int           = 3000
    print_level::MadNLP.LogLevels = MadNLP.INFO
    rethrow_error::Bool     = false
    kkt_system::KKT         = MadNLP.SparseKKTSystem
    linear_solver::LS       = nothing
    regularization::REG     = MadIPM.FixedRegularization(1e-10, 1e-10)
    step_rule::STEP         = MadIPM.AdaptiveStep(0.99)
    cudss_algorithm::ALG    = nothing
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    config::OptimizerConfig
    silent::Bool
    solver::Union{Nothing, MadNLP.AbstractMadNLPSolver}
    qp::Union{Nothing, MadIPM.QuadraticModel}
    stats::Union{Nothing,
                 MadNLP.MadNLPExecutionStats{Float64, <:AbstractVector{Float64}}}

    Optimizer() = new(OptimizerConfig(), false, nothing, nothing, nothing)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "MadIPM"
MOI.is_empty(opt::Optimizer) = opt.solver === nothing && opt.qp === nothing

function MOI.empty!(opt::Optimizer)
    opt.solver = nothing
    opt.qp     = nothing
    opt.stats  = nothing
    return nothing
end

# ---------- `RawOptimizerAttribute` → `OptimizerConfig` ----------

function _override(config::OptimizerConfig, sym::Symbol, value)
    return OptimizerConfig(;
        (f === sym ? (f => value) : (f => getfield(config, f))
            for f in fieldnames(OptimizerConfig))...)
end

function _check_config_field(name)
    hasfield(OptimizerConfig, Symbol(name)) ||
        throw(ArgumentError("Unsupported MadIPM optimizer attribute `$(name)`"))
end

function MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value)
    _check_config_field(param.name)
    opt.config = _override(opt.config, Symbol(param.name), value)
    return nothing
end

function MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute)
    _check_config_field(param.name)
    return getfield(opt.config, Symbol(param.name))
end

# ---------- silent / sense / supported constraints ----------

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(opt::Optimizer, ::MOI.Silent, v::Bool) = (opt.silent = v; nothing)
MOI.get(opt::Optimizer, ::MOI.Silent) = opt.silent

MOI.supports(::Optimizer,
    ::Union{MOI.ObjectiveSense, MOI.ObjectiveFunction{<:Union{VI, SAF, SQF}}}) = true

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) =
    model.qp === nothing ? MOI.MIN_SENSE :
    (model.qp.meta.minimize ? MOI.MIN_SENSE : MOI.MAX_SENSE)

MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}) = true

MOI.supports_constraint(::Optimizer, ::Type{VI},  ::Type{<:_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{SAF}, ::Type{<:_SCALAR_SETS}) = true

# ---------- copy_to / optimize! ----------

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    dest.solver = nothing
    dest.stats  = nothing
    dest.qp, index_map = BatchQuadraticModels.qp_model(src)
    dest.config.array_type === Vector{Float64} ||
        (dest.qp = Adapt.adapt(dest.config.array_type, dest.qp))
    return index_map
end

function MOI.optimize!(model::Optimizer)
    cfg         = model.config
    ls          = something(cfg.linear_solver, MadNLP.default_sparse_solver(model.qp))
    print_level = model.silent ? MadNLP.ERROR : cfg.print_level

    model.solver = MadIPM.MPCSolver(
        model.qp;
        tol             = cfg.tol,
        max_iter        = cfg.max_iter,
        print_level     = print_level,
        rethrow_error   = cfg.rethrow_error,
        kkt_system      = cfg.kkt_system,
        linear_solver   = ls,
        regularization  = cfg.regularization,
        step_rule       = cfg.step_rule,
        cudss_algorithm = cfg.cudss_algorithm,
    )
    model.stats = _host_stats(MadIPM.solve!(model.solver))
    return nothing
end

# Materialize all device-backed arrays onto the host so MOI getters return
# plain `Vector`s regardless of the solver's `array_type`.
function _host_stats(s::MadNLP.MadNLPExecutionStats{T}) where {T}
    return MadNLP.MadNLPExecutionStats(
        s.options, s.status,
        Vector{T}(s.solution), s.objective, Vector{T}(s.constraints),
        s.dual_feas, s.primal_feas,
        Vector{T}(s.multipliers),
        Vector{T}(s.multipliers_L), Vector{T}(s.multipliers_U),
        s.iter, s.counters,
    )
end

# ---------- status getters ----------

MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) =
    opt.stats === nothing ? 0.0 : opt.stats.counters.total_time

MOI.get(opt::Optimizer, ::MOI.RawStatusString) =
    opt.stats === nothing ? string(MOI.OPTIMIZE_NOT_CALLED) : string(opt.stats.status)

struct RawStatus <: MOI.AbstractModelAttribute
    name::Symbol
end
MOI.is_set_by_optimize(::RawStatus) = true

function MOI.get(opt::Optimizer, attr::RawStatus)
    opt.stats === nothing &&
        error("MadIPM: raw status available only after `optimize!`.")
    return getfield(opt.stats, attr.name)
end

# ---------- termination-status map ----------

const _TERMINATION_MAP = Dict{MadNLP.Status, MOI.TerminationStatusCode}(
    MadNLP.SOLVE_SUCCEEDED                      => MOI.OPTIMAL,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL           => MOI.ALMOST_OPTIMAL,
    MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL   => MOI.SLOW_PROGRESS,
    MadNLP.DIVERGING_ITERATES                   => MOI.INFEASIBLE_OR_UNBOUNDED,
    MadNLP.INFEASIBLE_PROBLEM_DETECTED          => MOI.INFEASIBLE,
    MadNLP.MAXIMUM_ITERATIONS_EXCEEDED          => MOI.ITERATION_LIMIT,
    MadNLP.MAXIMUM_WALLTIME_EXCEEDED            => MOI.TIME_LIMIT,
    MadNLP.INITIAL                              => MOI.OPTIMIZE_NOT_CALLED,
    MadNLP.RESTORATION_FAILED                   => MOI.NUMERICAL_ERROR,
    MadNLP.INVALID_NUMBER_DETECTED              => MOI.INVALID_MODEL,
    MadNLP.ERROR_IN_STEP_COMPUTATION            => MOI.NUMERICAL_ERROR,
    MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM        => MOI.INVALID_MODEL,
    MadNLP.USER_REQUESTED_STOP                  => MOI.INTERRUPTED,
    MadNLP.INTERNAL_ERROR                       => MOI.OTHER_ERROR,
    MadNLP.INVALID_NUMBER_OBJECTIVE             => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_GRADIENT              => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_CONSTRAINTS           => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_JACOBIAN              => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN    => MOI.INVALID_MODEL,
)

_termination_status(s::MadNLP.Status) = get(_TERMINATION_MAP, s, MOI.OTHER_ERROR)

MOI.get(opt::Optimizer, ::MOI.TerminationStatus) =
    opt.stats === nothing ? MOI.OPTIMIZE_NOT_CALLED :
                            _termination_status(opt.stats.status)

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return opt.stats.objective
end

function _result_status(opt::Optimizer, result_index)
    result_index > MOI.get(opt, MOI.ResultCount()) && return MOI.NO_SOLUTION
    term = MOI.get(opt, MOI.TerminationStatus())
    term === MOI.OPTIMAL    && return MOI.FEASIBLE_POINT
    term === MOI.INFEASIBLE && return MOI.INFEASIBLE_POINT
    return MOI.NO_SOLUTION
end

MOI.get(opt::Optimizer, attr::MOI.PrimalStatus) = _result_status(opt, attr.result_index)
MOI.get(opt::Optimizer, attr::MOI.DualStatus)   = _result_status(opt, attr.result_index)

# ---------- solution getters ----------

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(opt, attr)
    return opt.stats.solution[vi.value]
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal,
                  c::MOI.ConstraintIndex{VI, <:_SCALAR_SETS})
    MOI.check_result_index_bounds(model, attr)
    return MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(c.value))
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal,
                  c::MOI.ConstraintIndex{SAF, <:_SCALAR_SETS})
    MOI.check_result_index_bounds(model, attr)
    return model.stats.constraints[c.value + 1]
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual,
                  c::MOI.ConstraintIndex{VI, S}) where {S <: _SCALAR_SETS}
    MOI.check_result_index_bounds(model, attr)
    col = c.value
    if S <: MOI.LessThan
        return -model.stats.multipliers_U[col]
    elseif S <: MOI.GreaterThan
        return  model.stats.multipliers_L[col]
    else
        return model.stats.multipliers_L[col] - model.stats.multipliers_U[col]
    end
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual,
                  c::MOI.ConstraintIndex{SAF, S}) where {S <: _SCALAR_SETS}
    MOI.check_result_index_bounds(model, attr)
    return -model.stats.multipliers[c.value + 1]
end

MOI.get(opt::Optimizer, ::MOI.ResultCount) = opt.stats === nothing ? 0 : 1
