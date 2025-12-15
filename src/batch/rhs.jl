struct BatchPrimalVector{T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}, VI}
    values::MT
    values_lr::Vector{MadNLP.SubVector{T, VT, VI}}
    values_ur::Vector{MadNLP.SubVector{T, VT, VI}}
    x::Vector{VT}
    s::Vector{VT}
    full_views::Vector{VT}
    variable_view::SubArray{T, 2, MT, <:Tuple}
    slack_view::SubArray{T, 2, MT, <:Tuple}
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
    row_stride = nx + ns

    full_views = _init_batch_views!(Vector{VT}(undef, batch_size), values, nx+ns, 0, row_stride)
    x_views    = _init_batch_views!(Vector{VT}(undef, batch_size), values, nx,    0, row_stride)
    s_views    = _init_batch_views!(Vector{VT}(undef, batch_size), values, ns,   nx, row_stride)

    values_lr_views = _init_subviews!(Vector{MadNLP.SubVector{T, VT, VI}}(undef, batch_size), x_views, ind_lb)
    values_ur_views = _init_subviews!(Vector{MadNLP.SubVector{T, VT, VI}}(undef, batch_size), x_views, ind_ub)

    variable_view = view(values, 1:nx, :)
    slack_view = view(values, (nx+1):(nx+ns), :)

    return BatchPrimalVector(
        values, values_lr_views, values_ur_views,
        x_views, s_views, full_views,
        variable_view, slack_view,
        batch_size, nx, ns
    )
end

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
    primal_dual_buffer::VT
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
    row_stride = n + m + nlb + nub

    full_views = _init_batch_views!(Vector{VT}(undef, batch_size), values, n+m+nlb+nub, 0,         row_stride)
    x_views    = _init_batch_views!(Vector{VT}(undef, batch_size), values, n+m,         0,         row_stride)
    xp_views   = _init_batch_views!(Vector{VT}(undef, batch_size), values, n,           0,         row_stride)
    xl_views   = _init_batch_views!(Vector{VT}(undef, batch_size), values, m,           n,         row_stride)
    xzl_views  = _init_batch_views!(Vector{VT}(undef, batch_size), values, nlb,         n+m,       row_stride)
    xzu_views  = _init_batch_views!(Vector{VT}(undef, batch_size), values, nub,         n+m+nlb,   row_stride)

    xp_lr_views = _init_subviews!(Vector{MadNLP.SubVector{T, VT, VI}}(undef, batch_size), xp_views, ind_lb)
    xp_ur_views = _init_subviews!(Vector{MadNLP.SubVector{T, VT, VI}}(undef, batch_size), xp_views, ind_ub)

    primal_view      = view(values, 1:n, :)
    dual_view        = view(values, (n+1):(n+m), :)
    primal_dual_view = view(values, 1:(n+m), :)
    dual_lb_view     = view(values, (n+m+1):(n+m+nlb), :)
    dual_ub_view     = view(values, (n+m+nlb+1):(n+m+nlb+nub), :)

    primal_dual_buffer = VT(undef, (n+m) * batch_size)
    fill!(primal_dual_buffer, zero(T))

    return BatchUnreducedKKTVector(
        values, x_views, xp_views, xp_lr_views, xp_ur_views,
        xl_views, xzl_views, xzu_views, full_views,
        primal_view, dual_view, primal_dual_view, dual_lb_view, dual_ub_view,
        primal_dual_buffer,
        batch_size, n, m, nlb, nub
    )
end

MadNLP.primal_dual(buktv::BatchUnreducedKKTVector) = vec(buktv.primal_dual_view)

@inline function _primal_vector_view(batch::BatchPrimalVector, i::Int, ind_lb, ind_ub)
    MadNLP.PrimalVector(batch.full_views[i], batch.nx, batch.ns, ind_lb, ind_ub)
end

@inline function _unreduced_kkt_vector_view(batch::BatchUnreducedKKTVector, i::Int, ind_lb, ind_ub)
    MadNLP.UnreducedKKTVector(batch.full_views[i], batch.n, batch.m, batch.nlb, batch.nub, ind_lb, ind_ub)
end