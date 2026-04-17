#=
    Barrier update
=#

abstract type AbstractBarrierUpdate end
struct Mehrotra <: AbstractBarrierUpdate end

#=
    Step rule for next iterate
=#

abstract type AbstractStepRule end

@kwdef struct ConservativeStep{T} <: AbstractStepRule
    tau::T = T(0.995)
end

@kwdef struct AdaptiveStep{T} <: AbstractStepRule
    tau_min::T = T(0.99)
end

@kwdef struct MehrotraAdaptiveStep{T} <: AbstractStepRule
    gamma_f::T = T(0.99)
end

#=
    Primal-dual regularization for KKT system
=#

abstract type AbstractRegularization end

struct NoRegularization <: AbstractRegularization end

struct FixedRegularization{T} <: AbstractRegularization
    delta_p::T
    delta_d::T
end

struct AdaptiveRegularization{T} <: AbstractRegularization
    init_delta_p::T
    init_delta_d::T
    delta_min::T
end

#=
    Utils for linear solvers
=#

function is_factorized(::MadNLP.AbstractLinearSolver)
    return true # assume the system is factorized by default
end
function is_factorized(lin_solver::MadNLP.LDLSolver)
    return LDLFactorizations.factorized(lin_solver.inner)
end
function is_factorized(lin_solver::MadNLP.CHOLMODSolver)
    return issuccess(lin_solver.inner)
end


#=
    Options
=#

Base.@kwdef struct IPMOptions <: MadNLP.AbstractOptions
    tol::Float64 = 1e-8
    kkt_system::Type = MadNLP.SparseKKTSystem
    linear_solver::Type
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
    rethrow_error::Bool = false
    max_iter::Int = 3000
    max_wall_time::Float64 = 1e6
    divergence_tol::Float64 = 1e4
    divergence_scale::Float64 = 10.0
    kappa_d::Float64 = 1e-5
    fixed_variable_treatment::Type = kkt_system <: MadNLP.SparseCondensedKKTSystem ? MadNLP.RelaxBound : MadNLP.MakeParameter
    equality_treatment::Type = kkt_system <: MadNLP.SparseCondensedKKTSystem ? MadNLP.RelaxEquality : MadNLP.EnforceEquality
    scaling::Bool = true
    nlp_scaling_max_gradient::Float64 = 100.0
    bound_push::Float64 = 1e-2
    bound_fac::Float64 = 1e-2
    bound_relax_factor::Float64 = 1e-12
    s_max::Float64 = 100.0
    mu_init::Float64 = 1e-1
    mu_min::Float64 = 1e-12
    mu_superlinear_decrease_power::Float64 = 1.5
    tau_min::Float64 = 0.99
    tol_linear_solve::Float64 = 1e-8
    check_residual::Bool = false
end

IPMOptions(nlp::NLPModels.AbstractNLPModel; linear_solver = MadNLP.default_sparse_solver(nlp), kwargs...) =
    IPMOptions(; linear_solver = linear_solver, kwargs...)

function load_options(
    nlp;
    regularization::AbstractRegularization = FixedRegularization(1e-10, 1e-10),
    step_rule::AbstractStepRule = AdaptiveStep(0.99),
    barrier_update::AbstractBarrierUpdate = Mehrotra(),
    cudss_algorithm = nothing,
    kwargs...,
)
    opt_ipm = IPMOptions(nlp; kwargs...)
    opt_linear_solver = MadNLP.default_options(opt_ipm.linear_solver)
    if !isnothing(cudss_algorithm)
        opt_linear_solver.cudss_algorithm = cudss_algorithm
    end

    logger = MadNLP.MadNLPLogger(
        print_level = opt_ipm.print_level,
        file_print_level = opt_ipm.file_print_level,
        file = opt_ipm.output_file == "" ? nothing : open(opt_ipm.output_file, "w+"),
    )
    MadNLP.@trace(logger, "Logger is initialized.")

    return (
        interior_point = opt_ipm,
        linear_solver = opt_linear_solver,
        logger = logger,
        regularization = regularization,
        step_rule = step_rule,
        barrier_update = barrier_update,
    )
end

function coo_to_csr(
    n_rows,
    n_cols,
    Ai::AbstractVector{Ti},
    Aj::AbstractVector{Ti},
    Ax::AbstractVector{Tv},
) where {Tv, Ti}
    @assert length(Ai) == length(Aj) == length(Ax)
    nnz = length(Ai)
    Bp = zeros(Ti, n_rows+1)
    Bj = zeros(Ti, nnz)
    Bx = zeros(Tv, nnz)

    nnz = length(Ai)
    @inbounds for n in 1:nnz
        Bp[Ai[n]] += 1
    end

    # cumsum the nnz per row to get Bp
    cumsum = 1
    @inbounds for i in 1:n_rows
        tmp = Bp[i]
        Bp[i] = cumsum
        cumsum += tmp
    end
    Bp[n_rows+1] = nnz + 1

    @inbounds for n in 1:nnz
        i = Ai[n]
        dest = Bp[i]
        Bj[dest] = Aj[n]
        Bx[dest] = Ax[n]
        Bp[i] += 1
    end

    last = 1
    @inbounds for i in 1:n_rows+1
        tmp = Bp[i]
        Bp[i] = last
        last = tmp
    end

    return (Bp, Bj, Bx)
end

function coo_to_csr(A::MadNLP.SparseMatrixCOO)
    return coo_to_csr(
        A.m, A.n, A.I, A.J, A.V,
    )
end

function build_normal_system(
    n_rows,
    n_cols,
    Jtp::AbstractVector{Ti},
    Jtj::AbstractVector{Ti},
) where {Ti}
    Cp = zeros(Ti, n_rows + 1)
    xb = zeros(UInt8, n_cols)

    # Count nonzeros per rows
    nnz = 0
    @inbounds for i in 1:n_rows
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            xb[j] = UInt8(1)
        end
        # JᵀJ is symmetric, store only lower triangular part
        for j in i:n_rows
            for c in Jtp[j]:Jtp[j+1]-1
                k = Jtj[c]
                if xb[k] == 1
                    nnz += 1
                    Cp[i] += 1
                    break
                end
            end
        end
        # Reset to 0
        for c in Jtp[i]:Jtp[i+1]-1
            xb[Jtj[c]] = UInt8(0)
        end
    end
    # cumsum the nnz per row to get Bp
    cumsum = 1
    @inbounds for i in 1:n_rows
        tmp = Cp[i]
        Cp[i] = cumsum
        cumsum += tmp
    end
    Cp[n_rows+1] = nnz + 1

    Cj = zeros(Ti, nnz)
    cnt = 0
    @inbounds for i in 1:n_rows
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            xb[j] = UInt8(1)
        end
        # JᵀJ is symmetric, store only lower triangular part
        for j in i:n_rows
            for c in Jtp[j]:Jtp[j+1]-1
                k = Jtj[c]
                if xb[k] == 1
                    cnt += 1
                    Cj[cnt] = j
                    break
                end
            end
        end
        for c in Jtp[i]:Jtp[i+1]-1
            xb[Jtj[c]] = UInt8(0)
        end
    end

    return (Cp, Cj)
end

function assemble_normal_system!(
    n_rows,
    n_cols,
    Jtp::AbstractVector{Ti},
    Jtj::AbstractVector{Ti},
    Jtx::AbstractVector{Tv},
    Cp::AbstractVector{Ti},
    Cj::AbstractVector{Ti},
    Cx::AbstractVector{Tv},
    Dx::AbstractVector{Tv},
) where {Ti, Tv}
    buffer = zeros(Tv, n_cols)
    @inbounds for i in 1:n_rows
        # Read row i
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            buffer[j] = Jtx[c] * Dx[j]
        end
        for c in Cp[i]:Cp[i+1]-1
            j = Cj[c]
            Cx[c] = Tv(0)
            for d in Jtp[j]:Jtp[j+1]-1
                k = Jtj[d]
                Cx[c] += buffer[k] * Jtx[d]
            end
        end
        # Reset buffer
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            buffer[j] = Tv(0)
        end
    end
end

sparse_csc_format(::Type{<:Array}) = SparseArrays.SparseMatrixCSC
_colptr(A::SparseArrays.SparseArrays.SparseMatrixCSC) = A.colptr
_rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval
_nzval(A::SparseArrays.SparseMatrixCSC) = A.nzval

