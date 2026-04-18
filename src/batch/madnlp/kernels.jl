function get_inf_pr!(inf_pr, c)
    batch_mapreduce!(abs, max, zero(eltype(inf_pr)), inf_pr, c)
    return inf_pr
end

function get_inf_du!(inf_du, f_vals, zl_vals, jacl_vals)
    batch_mapreduce!((f, zl, jl) -> abs(f - zl + jl), max, zero(eltype(inf_du)),
                     inf_du, f_vals, zl_vals, jacl_vals)
    return inf_du
end

function get_inf_compl_lb!(inf_compl, x, xl, zl, sum_lb, nlb)
    T = eltype(inf_compl)
    if nlb > 0
        batch_mapreduce!((x, xl, z) -> abs(x - xl) * z, max, zero(T),
                         sum_lb, lower(x), lower(xl), lower(zl))
    else
        fill!(sum_lb, zero(T))
    end
    @. inf_compl = sum_lb
    return inf_compl
end
