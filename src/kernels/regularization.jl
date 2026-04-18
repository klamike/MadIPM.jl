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

# ---------- scalar setters ----------

@inline _assign_del!(s::MPCSolver, dw, dc) = (s.state.del_w = dw; s.state.del_c = dc; nothing)

@inline function _apply_reg_update!(s::MPCSolver, reg, ::Type{T}) where T
    s.state.del_w, s.state.del_c = _update_reg(reg, T, s.state.del_w, s.state.del_c)
    return
end

# ---------- batch setters ----------

@inline _assign_del!(s::AbstractBatchMPCSolver, dw, dc) = (fill!(s.state.del_w, dw); fill!(s.state.del_c, dc); nothing)

@inline function _apply_reg_update!(s::AbstractBatchMPCSolver, ::NoRegularization, ::Type{T}) where T
    state = s.state
    mask = state.workspace.active_mask
    @. state.del_w = ifelse(mask == one(T), zero(T), state.del_w)
    @. state.del_c = ifelse(mask == one(T), zero(T), state.del_c)
    return
end
@inline function _apply_reg_update!(s::AbstractBatchMPCSolver, r::FixedRegularization, ::Type{T}) where T
    state = s.state
    mask = state.workspace.active_mask
    dp, dd = T(r.delta_p), T(r.delta_d)
    @. state.del_w = ifelse(mask == one(T), dp, state.del_w)
    @. state.del_c = ifelse(mask == one(T), dd, state.del_c)
    return
end
@inline function _apply_reg_update!(s::AbstractBatchMPCSolver, r::AdaptiveRegularization, ::Type{T}) where T
    state = s.state
    mask = state.workspace.active_mask
    dmin = T(r.delta_min)
    @. state.del_w = ifelse(mask == one(T), max(state.del_w / T(10), dmin), state.del_w)
    @. state.del_c = ifelse(mask == one(T), min(state.del_c / T(10), -dmin), state.del_c)
    return
end
