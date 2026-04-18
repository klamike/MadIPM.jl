# Batched KKT / primal vectors for the std-form batch solver. Variables have
# only lower bounds and constraints are equalities — no upper-bound duals, no
# inequality slacks. `upper`, `dual_ub`, `xp_ur` accessors return empty views
# so any MadNLP-side hook that iterates them is a no-op.

struct BatchUnreducedKKTVector{T, MT<:AbstractMatrix{T}, VI, SV, IV}
    values::MT
    n::Int
    m::Int
    nlb::Int
    ind_lb::VI
    _primal::SV
    _dual::SV
    _primal_dual::SV
    _dual_lb::SV
    _empty::SV
    _xp_lr::IV
end

function BatchUnreducedKKTVector(
    ::Type{MT}, ::Type{VT},
    n::Int, m::Int, nlb::Int, batch_size::Int,
    ind_lb,
) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    total = n + m + nlb
    values = MT(undef, total, batch_size)
    fill!(values, zero(T))

    primal = view(values, 1:n, :)
    xp_lr  = view(values, ind_lb, :)
    empty  = view(values, 1:0, :)
    SV = typeof(primal); IV = typeof(xp_lr)
    return BatchUnreducedKKTVector{T, MT, typeof(ind_lb), SV, IV}(
        values, n, m, nlb, ind_lb,
        primal,
        view(values, n+1:n+m, :),
        view(values, 1:n+m, :),
        view(values, n+m+1:n+m+nlb, :),
        empty,
        xp_lr,
    )
end

# Compat shims for nativebatch-style code that still reads nub/ind_ub.
Base.getproperty(bv::BatchUnreducedKKTVector, s::Symbol) =
    s === :nub ? 0 :
    s === :ind_ub ? getfield(bv, :ind_lb)[1:0] :
    getfield(bv, s)

MadNLP.full(bv::BatchUnreducedKKTVector)        = bv.values
MadNLP.primal(bv::BatchUnreducedKKTVector)      = bv._primal
MadNLP.dual(bv::BatchUnreducedKKTVector)        = bv._dual
MadNLP.primal_dual(bv::BatchUnreducedKKTVector) = bv._primal_dual
MadNLP.dual_lb(bv::BatchUnreducedKKTVector)     = bv._dual_lb
MadNLP.dual_ub(bv::BatchUnreducedKKTVector)     = bv._empty
xp_lr(bv::BatchUnreducedKKTVector)              = bv._xp_lr
xp_ur(bv::BatchUnreducedKKTVector)              = bv._empty

struct BatchPrimalVector{T, MT<:AbstractMatrix{T}, VI, SV, IV}
    values::MT
    nx::Int
    ns::Int
    ind_lb::VI
    _variable::SV
    _slack::SV
    _lower::IV
    _empty::IV
end

function BatchPrimalVector(
    ::Type{MT}, ::Type{VT},
    nx::Int, ns::Int, batch_size::Int,
    ind_lb,
) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    total = nx + ns
    values = MT(undef, total, batch_size)
    fill!(values, zero(T))

    variable = view(values, 1:nx, :)
    lower    = view(values, ind_lb, :)
    empty    = view(values, ind_lb[1:0], :)
    SV = typeof(variable); IV = typeof(lower)
    return BatchPrimalVector{T, MT, typeof(ind_lb), SV, IV}(
        values, nx, ns, ind_lb,
        variable,
        view(values, nx+1:nx+ns, :),
        lower,
        empty,
    )
end

MadNLP.variable(bpv::BatchPrimalVector) = bpv._variable
MadNLP.slack(bpv::BatchPrimalVector)    = bpv._slack
lower(bpv::BatchPrimalVector)           = bpv._lower
upper(bpv::BatchPrimalVector)           = bpv._empty
MadNLP.full(bpv::BatchPrimalVector)     = bpv.values
MadNLP.primal(bpv::BatchPrimalVector)   = bpv.values
