struct BatchUnreducedKKTVector{T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}, VI}
    values::MT
    views::Vector{VT}
    primal_dual_view::SubArray{T, 2, MT, <:Tuple}
    primal_dual_buffer::VT
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

    row_stride = n+m+nlb+nub
    views = _init_batch_views!(Vector{VT}(undef, batch_size), values, row_stride, 0, row_stride)

    primal_dual_view = view(values, 1:(n+m), :)
    primal_dual_buffer = VT(undef, (n+m) * batch_size)
    fill!(primal_dual_buffer, zero(T))

    return BatchUnreducedKKTVector(values, views, primal_dual_view, primal_dual_buffer, n, m, nlb, nub)
end

MadNLP.primal_dual(buktv::BatchUnreducedKKTVector) = vec(buktv.primal_dual_view)

function _unreduced_kkt_vector_view(batch::BatchUnreducedKKTVector, i::Int, ind_lb, ind_ub)
    MadNLP.UnreducedKKTVector(batch.views[i], batch.n, batch.m, batch.nlb, batch.nub, ind_lb, ind_ub)
end