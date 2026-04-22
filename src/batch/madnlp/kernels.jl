# ============================================================================
# Per-instance infinity-norm residuals for the batched termination check.
#
#   inf_pr    = ‖c(x) − rhs‖∞
#   inf_du    = ‖∇f − ∇c'y − z_l + z_u‖∞
#   inf_compl = max_i |x − xl|·zl or |xu − x|·zu
# ============================================================================

get_inf_pr!(inf_pr, c) =
    batch_mapreduce!(abs, max, zero(eltype(inf_pr)), inf_pr, c)

get_inf_du!(inf_du, f_vals, zl_vals, zu_vals, jacl_vals) =
    batch_mapreduce!(
        (f, zl, zu, jl) -> abs(f - zl + zu + jl),
        max, zero(eltype(inf_du)),
        inf_du, f_vals, zl_vals, zu_vals, jacl_vals,
    )

function get_inf_compl!(inf_compl, x, xl, zl, xu, zu, sum_lb, sum_ub, nlb, nub)
    T = eltype(inf_compl)
    nlb > 0 ?
        batch_mapreduce!((x, xl, z) -> abs(x - xl) * z, max, zero(T),
                         sum_lb, lower(x), lower(xl), lower(zl)) :
        fill!(sum_lb, zero(T))
    nub > 0 ?
        batch_mapreduce!((xu, x, z) -> abs(xu - x) * z, max, zero(T),
                         sum_ub, upper(xu), upper(x), upper(zu)) :
        fill!(sum_ub, zero(T))
    @. inf_compl = max(sum_lb, sum_ub)
    return inf_compl
end
