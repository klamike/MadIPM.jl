# Primal-dual regularization init / update.
# Unified across scalar and batch via `_init_reg`/`_update_reg` helpers
# plus `_assign_del!` and `_apply_reg_update!` setters that dispatch on
# storage shape (scalar T vs (1, bs) matrix).

# ---------- shared helpers ----------

_init_reg(::NoRegularization, ::Type{T}) where {T} = (one(T), zero(T))
_init_reg(r::FixedRegularization, ::Type{T}) where {T} = (one(T), T(r.delta_d))
_init_reg(r::AdaptiveRegularization, ::Type{T}) where {T} = (T(r.init_delta_p), T(r.init_delta_d))

_update_reg(::NoRegularization, ::Type{T}, _, _) where {T} = (zero(T), zero(T))
_update_reg(r::FixedRegularization, ::Type{T}, _, _) where {T} = (T(r.delta_p), T(r.delta_d))
_update_reg(r::AdaptiveRegularization, ::Type{T}, dw, dc) where {T} =
    (max(dw / T(10), T(r.delta_min)), min(dc / T(10), -T(r.delta_min)))

# ---------- unified entry points ----------

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

# ---------- scalar setters ----------

@inline _assign_del!(s::MPCSolver, dw, dc) = (s.state.del_w = dw; s.state.del_c = dc; nothing)

@inline function _apply_reg_update!(s::MPCSolver, reg, ::Type{T}) where T
    s.state.del_w, s.state.del_c = _update_reg(reg, T, s.state.del_w, s.state.del_c)
    return
end

# ---------- batch setters ----------
# Mask against `active_mask` so converged instances retain their values, and
# evaluate per-element so AdaptiveRegularization decays each instance's own
# state rather than mutating the shared reg struct.

@inline _assign_del!(s::AbstractBatchMPCSolver, dw, dc) = (fill!(s.del_w, dw); fill!(s.del_c, dc); nothing)

@inline function _apply_reg_update!(s::AbstractBatchMPCSolver, ::NoRegularization, ::Type{T}) where T
    mask = s.workspace.active_mask
    @. s.del_w = ifelse(mask == one(T), zero(T), s.del_w)
    @. s.del_c = ifelse(mask == one(T), zero(T), s.del_c)
    return
end
@inline function _apply_reg_update!(s::AbstractBatchMPCSolver, r::FixedRegularization, ::Type{T}) where T
    mask = s.workspace.active_mask
    dp, dd = T(r.delta_p), T(r.delta_d)
    @. s.del_w = ifelse(mask == one(T), dp, s.del_w)
    @. s.del_c = ifelse(mask == one(T), dd, s.del_c)
    return
end
@inline function _apply_reg_update!(s::AbstractBatchMPCSolver, r::AdaptiveRegularization, ::Type{T}) where T
    mask = s.workspace.active_mask
    dmin = T(r.delta_min)
    @. s.del_w = ifelse(mask == one(T), max(s.del_w / T(10), dmin), s.del_w)
    @. s.del_c = ifelse(mask == one(T), min(s.del_c / T(10), -dmin), s.del_c)
    return
end
