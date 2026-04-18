# IPM RHS / correction kernels — unified across scalar and batch via
# `AnyMPCSolver` dispatch. Scalar's `_xl_r(s) = zero(T)` makes
# `(_xl_r(s) .- _x_lr(s))` collapse to `-_x_lr(s)` with no allocation.

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
