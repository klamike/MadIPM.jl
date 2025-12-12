function _madnlp_unsafe_column_wrap(mat::MT, n, shift, ::Type{VT}) where {T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}}
    return unsafe_wrap(VT, pointer(mat, shift), n)
end

struct BatchPrimalVector{T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}, VI}
    values::MT  # matrix of size (nx+ns, batch_size)
    values_lr::Vector{MadNLP.SubVector{T, VT, VI}}  # vector of subviews for lower bounds
    values_ur::Vector{MadNLP.SubVector{T, VT, VI}}  # vector of subviews for upper bounds
    x::Vector{VT}  # vector of unsafe views for x components
    s::Vector{VT}  # vector of unsafe views for s components
    full_views::Vector{VT}  # vector of unsafe views for full columns
    variable_view::SubArray{T, 2, MT, <:Tuple}  # view of variable part across all batch elements
    slack_view::SubArray{T, 2, MT, <:Tuple}  # view of slack part across all batch elements
    batch_size::Int
    nx::Int
    ns::Int
end

function BatchPrimalVector(
    ::Type{MT}, ::Type{VT}, nx::Int, ns::Int, batch_size::Int, ind_lb, ind_ub
) where {T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}}
    values = MT(undef, nx+ns, batch_size)
    fill!(values, zero(T))
    
    VI = typeof(ind_lb)
    x_views = Vector{VT}(undef, batch_size)
    s_views = Vector{VT}(undef, batch_size)
    full_views = Vector{VT}(undef, batch_size)
    values_lr_views = Vector{MadNLP.SubVector{T, VT, VI}}(undef, batch_size)
    values_ur_views = Vector{MadNLP.SubVector{T, VT, VI}}(undef, batch_size)
    
    for i in 1:batch_size
        col_start = (i-1)*(nx+ns) + 1
        full_views[i] = _madnlp_unsafe_column_wrap(values, nx+ns, col_start, VT)
        x_views[i] = _madnlp_unsafe_column_wrap(values, nx, col_start, VT)
        s_views[i] = _madnlp_unsafe_column_wrap(values, ns, col_start + nx, VT)
        values_lr_views[i] = view(x_views[i], ind_lb)
        values_ur_views[i] = view(x_views[i], ind_ub)
    end
    
    variable_view = view(values, 1:nx, :)
    slack_view = view(values, (nx+1):(nx+ns), :)
    
    return BatchPrimalVector(values, values_lr_views, values_ur_views, x_views, s_views, full_views, variable_view, slack_view, batch_size, nx, ns)
end

MadNLP.full(bpv::BatchPrimalVector) = bpv.values
MadNLP.full(bpv::BatchPrimalVector, i::Int) = bpv.full_views[i]
MadNLP.primal(bpv::BatchPrimalVector) = bpv.values
MadNLP.primal(bpv::BatchPrimalVector, i::Int) = MadNLP.full(bpv, i)
MadNLP.variable(bpv::BatchPrimalVector) = bpv.variable_view
MadNLP.variable(bpv::BatchPrimalVector, i::Int) = bpv.x[i]
MadNLP.slack(bpv::BatchPrimalVector) = bpv.slack_view
MadNLP.slack(bpv::BatchPrimalVector, i::Int) = bpv.s[i]

struct BatchUnreducedKKTVector{T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}, VI}
    values::MT
    x::Vector{VT}
    xp::Vector{VT}
    xp_lr::Vector{MadNLP.SubVector{T, VT, VI}}
    xp_ur::Vector{MadNLP.SubVector{T, VT, VI}}
    xl::Vector{VT}
    xzl::Vector{VT}
    xzu::Vector{VT}
    full_views::Vector{VT}
    primal_view::SubArray{T, 2, MT, <:Tuple}
    dual_view::SubArray{T, 2, MT, <:Tuple}
    primal_dual_view::SubArray{T, 2, MT, <:Tuple}
    dual_lb_view::SubArray{T, 2, MT, <:Tuple}
    dual_ub_view::SubArray{T, 2, MT, <:Tuple}
    batch_size::Int
    n::Int
    m::Int
    nlb::Int
    nub::Int
end

function BatchUnreducedKKTVector(
    ::Type{MT}, ::Type{VT}, n::Int, m::Int, nlb::Int, nub::Int, batch_size::Int, ind_lb, ind_ub
) where {T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}}
    values = MT(undef, n+m+nlb+nub, batch_size)
    fill!(values, zero(T))
    
    VI = typeof(ind_lb)
    x_views = Vector{VT}(undef, batch_size)
    xp_views = Vector{VT}(undef, batch_size)
    xl_views = Vector{VT}(undef, batch_size)
    xzl_views = Vector{VT}(undef, batch_size)
    xzu_views = Vector{VT}(undef, batch_size)
    full_views = Vector{VT}(undef, batch_size)
    xp_lr_views = Vector{MadNLP.SubVector{T, VT, VI}}(undef, batch_size)
    xp_ur_views = Vector{MadNLP.SubVector{T, VT, VI}}(undef, batch_size)
    
    for i in 1:batch_size
        col_start = (i-1)*(n+m+nlb+nub) + 1
        full_views[i] = _madnlp_unsafe_column_wrap(values, n+m+nlb+nub, col_start, VT)
        x_views[i] = _madnlp_unsafe_column_wrap(values, n+m, col_start, VT)
        xp_views[i] = _madnlp_unsafe_column_wrap(values, n, col_start, VT)
        xl_views[i] = _madnlp_unsafe_column_wrap(values, m, col_start + n, VT)
        xzl_views[i] = _madnlp_unsafe_column_wrap(values, nlb, col_start + n + m, VT)
        xzu_views[i] = _madnlp_unsafe_column_wrap(values, nub, col_start + n + m + nlb, VT)
        xp_lr_views[i] = view(xp_views[i], ind_lb)
        xp_ur_views[i] = view(xp_views[i], ind_ub)
    end
    
    primal_view = view(values, 1:n, :)
    dual_view = view(values, (n+1):(n+m), :)
    primal_dual_view = view(values, 1:(n+m), :)
    dual_lb_view = view(values, (n+m+1):(n+m+nlb), :)
    dual_ub_view = view(values, (n+m+nlb+1):(n+m+nlb+nub), :)
    
    return BatchUnreducedKKTVector(
        values, x_views, xp_views, xp_lr_views, xp_ur_views,
        xl_views, xzl_views, xzu_views, full_views,
        primal_view, dual_view, primal_dual_view, dual_lb_view, dual_ub_view,
        batch_size, n, m, nlb, nub
    )
end

MadNLP.full(buktv::BatchUnreducedKKTVector) = buktv.values
MadNLP.full(buktv::BatchUnreducedKKTVector, i::Int) = buktv.full_views[i]
MadNLP.primal(buktv::BatchUnreducedKKTVector) = buktv.primal_view
MadNLP.primal(buktv::BatchUnreducedKKTVector, i::Int) = buktv.xp[i]
MadNLP.dual(buktv::BatchUnreducedKKTVector) = buktv.dual_view
MadNLP.dual(buktv::BatchUnreducedKKTVector, i::Int) = buktv.xl[i]
MadNLP.primal_dual(buktv::BatchUnreducedKKTVector) = buktv.primal_dual_view
MadNLP.primal_dual(buktv::BatchUnreducedKKTVector, i::Int) = buktv.x[i]
MadNLP.dual_lb(buktv::BatchUnreducedKKTVector) = buktv.dual_lb_view
MadNLP.dual_lb(buktv::BatchUnreducedKKTVector, i::Int) = buktv.xzl[i]
MadNLP.dual_ub(buktv::BatchUnreducedKKTVector) = buktv.dual_ub_view
MadNLP.dual_ub(buktv::BatchUnreducedKKTVector, i::Int) = buktv.xzu[i]

struct ViewBasedPrimalVector{T, VT<:AbstractVector{T}} <: MadNLP.AbstractPrimalVector{T, VT}
    values::VT
    values_lr
    values_ur
    x
    s
end

function ViewBasedPrimalVector(values::VT, nx::Int, ns::Int, ind_lb, ind_ub) where {T, VT <: AbstractVector{T}}
    x = MadNLP._madnlp_unsafe_wrap(values, nx)
    s = MadNLP._madnlp_unsafe_wrap(values, ns, nx+1)
    values_lr = view(values, ind_lb)
    values_ur = view(values, ind_ub)
    return ViewBasedPrimalVector{T, VT}(values, values_lr, values_ur, x, s)
end

MadNLP.full(rhs::ViewBasedPrimalVector) = rhs.values
MadNLP.primal(rhs::ViewBasedPrimalVector) = rhs.values
MadNLP.variable(rhs::ViewBasedPrimalVector) = rhs.x
MadNLP.slack(rhs::ViewBasedPrimalVector) = rhs.s

struct ViewBasedUnreducedKKTVector{T, VT<:AbstractVector{T}} <: MadNLP.AbstractUnreducedKKTVector{T, VT}
    values::VT
    x
    xp
    xp_lr
    xp_ur
    xl
    xzl
    xzu
end

function ViewBasedUnreducedKKTVector(values::VT, n::Int, m::Int, nlb::Int, nub::Int, ind_lb, ind_ub) where {T, VT <: AbstractVector{T}}
    x = MadNLP._madnlp_unsafe_wrap(values, n + m)  # Primal-Dual
    xp = MadNLP._madnlp_unsafe_wrap(values, n)  # Primal
    xl = MadNLP._madnlp_unsafe_wrap(values, m, n+1)  # Dual
    xzl = MadNLP._madnlp_unsafe_wrap(values, nlb, n + m + 1)  # Lower bound
    xzu = MadNLP._madnlp_unsafe_wrap(values, nub, n + m + nlb + 1)  # Upper bound
    xp_lr = view(xp, ind_lb)
    xp_ur = view(xp, ind_ub)
    return ViewBasedUnreducedKKTVector{T, VT}(values, x, xp, xp_lr, xp_ur, xl, xzl, xzu)
end

MadNLP.full(rhs::ViewBasedUnreducedKKTVector) = rhs.values
MadNLP.primal(rhs::ViewBasedUnreducedKKTVector) = rhs.xp
MadNLP.dual(rhs::ViewBasedUnreducedKKTVector) = rhs.xl
MadNLP.primal_dual(rhs::ViewBasedUnreducedKKTVector) = rhs.x
MadNLP.dual_lb(rhs::ViewBasedUnreducedKKTVector) = rhs.xzl
MadNLP.dual_ub(rhs::ViewBasedUnreducedKKTVector) = rhs.xzu
