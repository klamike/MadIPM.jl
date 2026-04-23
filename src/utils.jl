# ---------- barrier / step-length policy types ----------

"""
    AbstractBarrierUpdate

Strategy for updating the central-path parameter μ between iterations.
Mehrotra's predictor-corrector is the only implementation today; kept
dispatchable in case alternatives are added.
"""
abstract type AbstractBarrierUpdate end

"""
    Mehrotra()

Predictor-corrector barrier update: computes μ_affine from an affine step,
sets μ_new = max((μ_affine / μ_curr)^3, σ_min) · μ_curr.
"""
struct Mehrotra <: AbstractBarrierUpdate end

"""
    AbstractStepRule

How the fraction-to-boundary ratio τ ∈ (0,1) is picked each iteration.
Smaller τ is more conservative (slower but safer); larger τ cuts corners
closer to the boundary.
"""
abstract type AbstractStepRule end

"""
    ConservativeStep(tau = 0.995)

Fixed τ = `tau` on every iteration.
"""
@kwdef struct ConservativeStep{T} <: AbstractStepRule
    tau::T = T(0.995)
end

"""
    AdaptiveStep(tau_min = 0.99)

τ ramps from `tau_min` toward 1 as μ shrinks — tighter near the solution.
"""
@kwdef struct AdaptiveStep{T} <: AbstractStepRule
    tau_min::T = T(0.99)
end

"""
    MehrotraAdaptiveStep(gamma_f = 0.99)

Mehrotra's heuristic: derives τ from the complementarity drop achievable
under the affine step. More aggressive than `AdaptiveStep` in good-condition
regions.
"""
@kwdef struct MehrotraAdaptiveStep{T} <: AbstractStepRule
    gamma_f::T = T(0.99)
end

# ---------- primal-dual regularization ----------

"""
    AbstractRegularization

KKT-system regularization: adds `+δ_p I` to the primal block and `-δ_d I` to
the dual block before factoring, to keep factorization stable when the
optimum sits on a degenerate face.
"""
abstract type AbstractRegularization end

"""
    NoRegularization()

Disable regularization. Only safe when the KKT system is guaranteed
non-singular (e.g. `NormalKKTSystem` on a well-conditioned LP).
"""
struct NoRegularization <: AbstractRegularization end

"""
    FixedRegularization(delta_p, delta_d)

Apply a constant `(δ_p, δ_d)` on every iteration. Robust but slightly slower
near convergence.
"""
struct FixedRegularization{T} <: AbstractRegularization
    delta_p::T
    delta_d::T
end

"""
    AdaptiveRegularization(init_delta_p, init_delta_d, delta_min)

Start at `(init_delta_p, init_delta_d)` and shrink toward `delta_min` as the
residual improves; inflate on factorization failures.
"""
struct AdaptiveRegularization{T} <: AbstractRegularization
    init_delta_p::T
    init_delta_d::T
    delta_min::T
end

# ---------- linear-solver introspection ----------

# Assume `factorize!(...)` succeeds unless the solver exposes a check.
is_factorized(::MadNLP.AbstractLinearSolver) = true
is_factorized(ls::MadNLP.LDLSolver)     = LDLFactorizations.factorized(ls.inner)
is_factorized(ls::MadNLP.CHOLMODSolver) = issuccess(ls.inner)

# ---------- IPM solver options ----------

"""
    IPMOptions

Collected options for the MPC solver — tolerances, KKT system, step/bound
policies, and logging. Constructed indirectly by `load_options(nlp; ...)`
which also builds the linear-solver options and logger.
"""
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
    fixed_variable_treatment::Type = kkt_system <: MadNLP.SparseCondensedKKTSystem ? MadNLP.RelaxBound : MadNLP.MakeParameter
    equality_treatment::Type = kkt_system <: MadNLP.SparseCondensedKKTSystem ? MadNLP.RelaxEquality : MadNLP.EnforceEquality
    bound_push::Float64 = 1e-2
    bound_fac::Float64 = 1e-2
    mu_init::Float64 = 1e-1
    mu_min::Float64 = 1e-12
    tau_min::Float64 = 0.99
    tol_linear_solve::Float64 = 1e-8
    check_residual::Bool = false
end

IPMOptions(nlp::NLPModels.AbstractNLPModel; linear_solver = MadNLP.default_sparse_solver(nlp), kwargs...) =
    IPMOptions(; linear_solver = linear_solver, kwargs...)

"""
    load_options(nlp; regularization, step_rule, barrier_update, cudss_algorithm, kwargs...)

Build the full option bundle consumed by `MPCSolver` / `UniformBatchMPCSolver`:
IPM options (from `IPMOptions`), linear-solver options (from the backend's
`default_options`), the logger, and the three policy objects.
"""
function load_options(
    nlp;
    regularization::AbstractRegularization = FixedRegularization(1e-10, 1e-10),
    step_rule::AbstractStepRule = AdaptiveStep(0.99),
    barrier_update::AbstractBarrierUpdate = Mehrotra(),
    scaling::AbstractScaling = RuizScaling(),
    cudss_algorithm = nothing,
    cudss_ir = nothing,
    kwargs...,
)
    opt_ipm = IPMOptions(nlp; kwargs...)
    opt_linear_solver = MadNLP.default_options(opt_ipm.linear_solver)
    if !isnothing(cudss_algorithm)
        opt_linear_solver.cudss_algorithm = cudss_algorithm
    end
    # CUDSS-LDL on the unreduced augmented system is fragile without iterative
    # refinement (e.g. Netlib `forplan`); Ruiz scaling helps but doesn't make
    # the factor itself more accurate. Default to a modest IR budget when the
    # linear solver is CUDSS — cheap and broadly stabilizing.
    if hasfield(typeof(opt_linear_solver), :cudss_ir)
        opt_linear_solver.cudss_ir = isnothing(cudss_ir) ? 5 : cudss_ir
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
        scaling = scaling,
    )
end

# ---------- sparse-matrix helpers ----------
# COO → CSR conversion and JᵀJ (normal-system) structure building. Used by
# `NormalKKTSystem`'s build path; GPU ext overrides with CUSPARSE / KA kernels.

"""
    coo_to_csr(n_rows, n_cols, Ai, Aj, Ax) -> (Bp, Bj, Bx)

Convert a COO triple to a CSR row-pointer / col-index / value triple. Rows
are not sorted; use the output as-is for symbolic normal-system construction.
"""
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

    @inbounds for n in 1:nnz
        Bp[Ai[n]] += 1
    end

    # cumsum per-row counts to get Bp
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

coo_to_csr(A::MadNLP.SparseMatrixCOO) = coo_to_csr(A.m, A.n, A.I, A.J, A.V)

"""
    build_normal_system(n_rows, n_cols, Jtp, Jtj) -> (Cp, Cj)

Precompute the symbolic sparsity pattern of `C = J D J'` (only its lower
triangle) from the CSR structure of `Jᵀ` (`Jtp`, `Jtj`). Called once at
`NormalKKTSystem` build time; `assemble_normal_system!` fills the values.
"""
function build_normal_system(
    n_rows,
    n_cols,
    Jtp::AbstractVector{Ti},
    Jtj::AbstractVector{Ti},
) where {Ti}
    Cp = zeros(Ti, n_rows + 1)
    xb = zeros(UInt8, n_cols)

    # Count nonzeros per row (only below-diagonal since JᵀJ is symmetric).
    nnz = 0
    @inbounds for i in 1:n_rows
        for c in Jtp[i]:Jtp[i+1]-1
            xb[Jtj[c]] = UInt8(1)
        end
        for j in i:n_rows
            for c in Jtp[j]:Jtp[j+1]-1
                if xb[Jtj[c]] == 1
                    nnz += 1
                    Cp[i] += 1
                    break
                end
            end
        end
        for c in Jtp[i]:Jtp[i+1]-1
            xb[Jtj[c]] = UInt8(0)
        end
    end
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
            xb[Jtj[c]] = UInt8(1)
        end
        for j in i:n_rows
            for c in Jtp[j]:Jtp[j+1]-1
                if xb[Jtj[c]] == 1
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

"""
    assemble_normal_system!(n_rows, n_cols, Jtp, Jtj, Jtx, Cp, Cj, Cx, Dx)

Fill `Cx` with the values of `C = J D J'` (lower triangle, pattern from
`build_normal_system`). `Dx::Vector` is the diagonal scaling.
"""
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
        # Materialize row i of (D·Jᵀ) into `buffer`.
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            buffer[j] = Jtx[c] * Dx[j]
        end
        # Dot with rows j ≥ i of Jᵀ to fill the lower-triangle slots.
        for c in Cp[i]:Cp[i+1]-1
            j = Cj[c]
            Cx[c] = Tv(0)
            for d in Jtp[j]:Jtp[j+1]-1
                k = Jtj[d]
                Cx[c] += buffer[k] * Jtx[d]
            end
        end
        for c in Jtp[i]:Jtp[i+1]-1
            buffer[Jtj[c]] = Tv(0)
        end
    end
end

# ---------- CSC accessor shims ----------
# Single source of truth so scalar/GPU can share the same normal-system code
# regardless of the concrete CSC type. GPU ext overrides with CuSparseMatrixCSC.

sparse_csc_format(::Type{<:Array}) = SparseArrays.SparseMatrixCSC
_colptr(A::SparseArrays.SparseMatrixCSC) = A.colptr
_rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval
_nzval(A::SparseArrays.SparseMatrixCSC)  = A.nzval
