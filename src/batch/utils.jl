@inline function _madnlp_unsafe_column_wrap(mat::MT, n, shift, ::Type{VT}) where {T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}}
    return unsafe_wrap(VT, pointer(mat, shift), n)
end

@inline function _alloc_batch_buffer(sym::Symbol, shared::Tuple, size::Int, batch_size::Int, ::Type{VT}, ::Type{MT}) where {VT, MT}
    sym in shared ? VT(undef, size) : MT(undef, size, batch_size)
end

@inline function _batch_view(buffer, sym::Symbol, shared::Tuple, size::Int, i::Int, ::Type{VT}) where VT
    sym in shared ? buffer : _madnlp_unsafe_column_wrap(buffer, size, (i-1)*size + 1, VT)
end

@inline function _init_batch_views!(
    views::Vector{VT}, values::MT, view_size::Int, view_offset::Int, row_stride::Int
) where {T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}}
    for i in 1:length(views)
        col_start = (i-1)*row_stride + 1 + view_offset
        views[i] = _madnlp_unsafe_column_wrap(values, view_size, col_start, VT)
    end
    return views
end

@inline function _init_subviews!(subviews::Vector, base_views::Vector, indices)
    for i in 1:length(subviews)
        subviews[i] = view(base_views[i], indices)
    end
    return subviews
end

@inline function _scalar_view(v::Vector{T}, i::Int) where T
    MadNLP._madnlp_unsafe_wrap(v, 1, i)
end

@inline function _matrix_column_view(mat::MT, size::Int, i::Int, ::Type{VT}) where {T, MT <: AbstractMatrix{T}, VT}
    _madnlp_unsafe_column_wrap(mat, size, (i-1)*size + 1, VT)
end
