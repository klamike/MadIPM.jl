# Predictor-corrector RHS assembly. Each function writes the step-system
# right-hand side `p` from the current iterate; `set_predictive_rhs!` /
# `set_correction_rhs!` take the KKT system as an extra dispatch argument so
# new KKT systems can specialize without touching the solver loop. Today all
# KKT systems (scalar `AbstractKKTSystem` + batch `AbstractBatchKKTSystem`)
# share the same Mehrotra formula and route to `_set_*_impl!`.

# Initialize the primal RHS (p-side zero, d-side = -c): used by the starting
# point solve that sets `x` from current constraints.
function set_initial_primal_rhs!(s::MaybeBatchMPCSolver{T}) where {T}
    p = _p(s)
    fill!(MadNLP.full(p), zero(T))
    MadNLP.dual(p) .= .-_c(s)
    return
end

# Initialize the dual RHS (p-side = -∇f, d-side zero): used by the starting
# point solve that sets `(y, z)` from the current gradient.
function set_initial_dual_rhs!(s::MaybeBatchMPCSolver{T}) where {T}
    p = _p(s)
    fill!(MadNLP.full(p), zero(T))
    MadNLP.primal(p) .= .-MadNLP.primal(_f(s))
    return
end

# Predictor (affine) RHS: `∇f - ∇c'y - z` on primal, `-c` on dual, bound
# complementarity `(xl - x) * zl` on the lower-bound slice.
set_predictive_rhs!(s::MaybeBatchMPCSolver{T}, ::MadNLP.AbstractKKTSystem) where {T} =
    _set_predictive_rhs_impl!(s)
set_predictive_rhs!(s::UniformBatchMPCSolver{T}, ::AbstractBatchKKTSystem) where {T} =
    _set_predictive_rhs_impl!(s)

@inline function _set_predictive_rhs_impl!(s::MaybeBatchMPCSolver{T}) where {T}
    p = _p(s)
    fill!(MadNLP.full(p), zero(T))
    MadNLP.primal(p)  .= .-MadNLP.primal(_f(s)) .+ MadNLP.full(_zl(s)) .- _jacl(s)
    MadNLP.dual(p)    .= .-_c(s)
    MadNLP.dual_lb(p) .= (_xl_r(s) .- _x_lr(s)) .* _zl_r(s)
    return
end

# Corrector RHS: same as predictor plus the Mehrotra correction
# `μ - (Δx_aff .* Δz_aff)` on the bound-complementarity slice.
set_correction_rhs!(s::MaybeBatchMPCSolver{T}, ::MadNLP.AbstractKKTSystem, mu, correction_lb) where {T} =
    _set_correction_rhs_impl!(s, mu, correction_lb)
set_correction_rhs!(s::UniformBatchMPCSolver{T}, ::AbstractBatchKKTSystem, mu, correction_lb) where {T} =
    _set_correction_rhs_impl!(s, mu, correction_lb)

@inline function _set_correction_rhs_impl!(s::MaybeBatchMPCSolver{T}, mu, correction_lb) where {T}
    p = _p(s)
    MadNLP.primal(p)  .= .-MadNLP.primal(_f(s)) .+ MadNLP.full(_zl(s)) .- _jacl(s)
    MadNLP.dual(p)    .= .-_c(s)
    MadNLP.dual_lb(p) .= (_xl_r(s) .- _x_lr(s)) .* _zl_r(s) .+ mu .- correction_lb
    return
end

# Stash the affine-step's complementarity product for the corrector to reuse.
function get_correction!(s::MaybeBatchMPCSolver, correction_lb)
    correction_lb .= _dx_lr(s) .* _dz_lb(s)
    return
end

# Assemble the predictor RHS and solve the KKT system for the affine step.
function affine_direction!(s::MaybeBatchMPCSolver)
    set_predictive_rhs!(s, _kkt(s))
    solve_system!(_d(s), s, _p(s))
    return
end

# Assemble the corrector RHS (μ and the stashed affine complementarity) and
# solve for the full Mehrotra step.
function mehrotra_correction_direction!(s::MaybeBatchMPCSolver)
    set_correction_rhs!(s, _kkt(s), _mu(s), _correction_lb(s))
    solve_system!(_d(s), s, _p(s))
    return
end
