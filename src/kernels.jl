# IPM kernels.
#
# Many kernels are unified across the scalar `MPCSolver` and the batched
# `UniformBatchMPCSolver` via dispatch on `AnyMPCSolver{T}` (defined in
# src/batch/structure.jl). The accessors (`_x_lr`, `_zl_r`, `_alpha_p`,
# `_del_w`, `_mu`, ...) live alongside each solver type and return the right
# shape — `T`/`Vector` on scalar, `Matrix(1,bs)`/`Matrix(dim,bs)` on batch —
# so the same broadcast expressions work on both.
#
# A handful of operations stay specialized:
#   * `set_aug_diagonal_reg!` — KKT-system specific (different fields per
#     KKT type) and batch has masked / unmasked variants for the active set.
#   * `_xz_sum` / `get_complementarity_measure` /
#     `get_affine_complementarity_measure` — scalar uses `mapreduce`, batch
#     uses `batch_mapreduce!`; signatures differ.
#   * `update_step!` for `MehrotraAdaptiveStep` — scalar's tight scalar-state
#     formulation vs batch's per-column kernel.
#
# This file is loaded *after* the batch infrastructure so that
# `AnyMPCSolver{T}` is in scope.

# ---------- unified IPM direction kernels ----------

function affine_direction!(s::AnyMPCSolver)
    set_predictive_rhs!(s, _kkt(s))
    solve_system!(_d(s), s, _p(s))
    return
end

function mehrotra_correction_direction!(s::AnyMPCSolver)
    set_correction_rhs!(s, _kkt(s), _mu(s), _correction_lb(s))
    solve_system!(_d(s), s, _p(s))
    return
end

# ---------- unified IPM RHS / correction kernels ----------

function set_initial_primal_rhs!(s::AnyMPCSolver{T}) where {T}
    p = _p(s)
    fill!(MadNLP.full(p), zero(T))
    MadNLP.dual(p) .= .-_c(s)
    return
end

function set_initial_dual_rhs!(s::AnyMPCSolver{T}) where {T}
    p = _p(s)
    fill!(MadNLP.full(p), zero(T))
    MadNLP.primal(p) .= .-MadNLP.primal(_f(s))
    return
end

function set_predictive_rhs!(s::AnyMPCSolver{T}, ::MadNLP.AbstractKKTSystem) where {T}
    _set_predictive_rhs_impl!(s)
end
function set_predictive_rhs!(s::AbstractBatchMPCSolver{T}, ::AbstractBatchKKTSystem) where {T}
    _set_predictive_rhs_impl!(s)
end

@inline function _set_predictive_rhs_impl!(s::AnyMPCSolver{T}) where {T}
    p = _p(s)
    fill!(MadNLP.full(p), zero(T))
    MadNLP.primal(p)  .= .-MadNLP.primal(_f(s)) .+ MadNLP.full(_zl(s)) .- _jacl(s)
    MadNLP.dual(p)    .= .-_c(s)
    MadNLP.dual_lb(p) .= (_xl_r(s) .- _x_lr(s)) .* _zl_r(s)
    return
end

function set_correction_rhs!(s::AnyMPCSolver{T}, ::MadNLP.AbstractKKTSystem, mu, correction_lb) where {T}
    _set_correction_rhs_impl!(s, mu, correction_lb)
end
function set_correction_rhs!(s::AbstractBatchMPCSolver{T}, ::AbstractBatchKKTSystem, mu, correction_lb) where {T}
    _set_correction_rhs_impl!(s, mu, correction_lb)
end

@inline function _set_correction_rhs_impl!(s::AnyMPCSolver{T}, mu, correction_lb) where {T}
    p = _p(s)
    MadNLP.primal(p)  .= .-MadNLP.primal(_f(s)) .+ MadNLP.full(_zl(s)) .- _jacl(s)
    MadNLP.dual(p)    .= .-_c(s)
    MadNLP.dual_lb(p) .= (_xl_r(s) .- _x_lr(s)) .* _zl_r(s) .+ mu .- correction_lb
    return
end

function get_correction!(s::AnyMPCSolver, correction_lb)
    correction_lb .= _dx_lr(s) .* _dz_lb(s)
    return
end

# ---------- KKT-system-specific augmented-diagonal setup (scalar) ----------

function set_aug_diagonal_reg!(kkt::MadNLP.AbstractKKTSystem{T}, solver::MPCSolver{T}) where {T}
    state = solver.state
    fill!(kkt.reg, state.del_w)
    fill!(kkt.du_diag, state.del_c)
    kkt.l_diag .= .-state.x_lr
    copyto!(kkt.l_lower, state.zl_r)
    copyto!(kkt.pr_diag, kkt.reg)
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    return
end

function set_aug_diagonal_reg!(kkt::MadNLP.ScaledSparseKKTSystem{T}, solver::MPCSolver{T}) where {T}
    state = solver.state
    fill!(kkt.reg, state.del_w)
    fill!(kkt.du_diag, state.del_c)
    kkt.l_diag .= state.x_lr
    copyto!(kkt.l_lower, state.zl_r)
    MadNLP._set_aug_diagonal!(kkt)
    return
end

# ---------- complementarity measures (scalar — batch lives in batch/madipm) ----------

function _xz_sum(solver::MPCSolver)
    x = solver.state.x_lr
    isempty(x) && return zero(eltype(x))
    return mapreduce(*, +, x, solver.state.zl_r; init = zero(eltype(x)))
end

get_complementarity_measure(solver::MPCSolver) =
    isempty(solver.state.x_lr) ? zero(eltype(solver.state.y)) :
        _xz_sum(solver) / length(solver.state.x_lr)

function update_barrier!(rule::Mehrotra, solver::MPCSolver{T}, mu_affine) where {T}
    problem = solver.problem
    state   = solver.state
    mu_curr = get_complementarity_measure(solver)
    sigma = if problem.nlb > 0
        iszero(mu_curr) ? one(T) : clamp((mu_affine / mu_curr)^3, T(1e-6), T(10))
    else
        one(T)
    end
    state.mu = max(T(problem.opt.mu_min), sigma * mu_curr)
    return mu_curr
end

function get_affine_complementarity_measure(solver::MPCSolver, alpha_p, alpha_d)
    state = solver.state
    isempty(state.x_lr) && return zero(eltype(state.x_lr))
    return mapreduce(
        (x, dx, z, dz) -> (x + alpha_p * dx) * (z + alpha_d * dz),
        +,
        state.x_lr, state.dx_lr, state.zl_r, MadNLP.dual_lb(state.d);
        init = zero(eltype(state.x_lr)),
    ) / length(state.x_lr)
end

# ---------- step rules (scalar) ----------

function get_alpha_max(v::AbstractVector{T}, dv, tau::T) where {T}
    return mapreduce(
        (dvi, vi, i) -> ((dvi < 0 ? (-vi) * tau / dvi : Inf), i),
        (a, b) -> a[1] < b[1] ? a : b,
        dv, v, eachindex(v);
        init = (one(T), 0),
    )
end

function get_fraction_to_boundary_step(solver::MPCSolver, tau)
    state = solver.state
    alpha_x, _ = get_alpha_max(state.x_lr, state.dx_lr, tau)
    alpha_z, _ = get_alpha_max(state.zl_r, MadNLP.dual_lb(state.d), tau)
    return alpha_x, alpha_z
end

get_tau(rule::ConservativeStep, solver::MPCSolver) = rule.tau
get_tau(rule::AdaptiveStep, solver::MPCSolver) =
    max(one(typeof(solver.state.mu)) - solver.state.mu, typeof(solver.state.mu)(rule.tau_min))

function update_step!(rule::Union{ConservativeStep, AdaptiveStep}, solver::MPCSolver)
    state = solver.state
    tau = get_tau(rule, solver)
    state.alpha_p, state.alpha_d = get_fraction_to_boundary_step(solver, tau)
    return
end

function update_step!(rule::MehrotraAdaptiveStep, solver::MPCSolver)
    state = solver.state
    T = eltype(state.y)
    gamma_a = one(T) / (one(T) - T(rule.gamma_f))
    d_zl = MadNLP.dual_lb(state.d)
    alpha_x, i_x = get_alpha_max(state.x_lr, state.dx_lr, one(T))
    alpha_z, i_z = get_alpha_max(state.zl_r, d_zl, one(T))

    mu_full = get_affine_complementarity_measure(solver, alpha_x, alpha_z) / gamma_a
    alpha_p = one(T)
    alpha_d = one(T)

    if alpha_x < one(T)
        tmp = mu_full / (state.zl_r[i_x] + alpha_z * d_zl[i_x])
        alpha_p = (state.x_lr[i_x] - tmp) / (-state.dx_lr[i_x])
    end
    if alpha_z < one(T)
        tmp = mu_full / (state.x_lr[i_z] + alpha_x * state.dx_lr[i_z])
        alpha_d = -(state.zl_r[i_z] - tmp) / d_zl[i_z]
    end

    state.alpha_p = max(alpha_p, T(rule.gamma_f) * alpha_x)
    state.alpha_d = max(alpha_d, T(rule.gamma_f) * alpha_z)
    return
end

# ---------- regularization (helpers shared by scalar and batch) ----------

_init_reg(::NoRegularization, ::Type{T}) where {T} = (one(T), zero(T))
_init_reg(r::FixedRegularization, ::Type{T}) where {T} = (one(T), T(r.delta_d))
_init_reg(r::AdaptiveRegularization, ::Type{T}) where {T} = (T(r.init_delta_p), T(r.init_delta_d))

_update_reg(::NoRegularization, ::Type{T}, _, _) where {T} = (zero(T), zero(T))
_update_reg(r::FixedRegularization, ::Type{T}, _, _) where {T} = (T(r.delta_p), T(r.delta_d))
_update_reg(r::AdaptiveRegularization, ::Type{T}, dw, dc) where {T} =
    (max(dw / T(10), T(r.delta_min)), min(dc / T(10), -T(r.delta_min)))

function init_regularization!(solver::AnyMPCSolver, reg::AbstractRegularization)
    T = eltype(_y(solver))
    dw, dc = _init_reg(reg, T)
    _assign_del!(solver, dw, dc)
    return
end

function update_regularization!(solver::AnyMPCSolver, reg::AbstractRegularization)
    T = eltype(_y(solver))
    _apply_reg_update!(solver, reg, T)
    return
end

@inline function _apply_reg_update!(s::MPCSolver, reg, ::Type{T}) where T
    s.state.del_w, s.state.del_c = _update_reg(reg, T, s.state.del_w, s.state.del_c)
    return
end

@inline _assign_del!(s::MPCSolver, dw, dc) = (s.state.del_w = dw; s.state.del_c = dc; nothing)
@inline _assign_del!(s::AbstractBatchMPCSolver, dw, dc) = (fill!(s.del_w, dw); fill!(s.del_c, dc); nothing)
