# Fused `update_termination_criteria!` (src/batch/madipm/solver.jl): one
# kernel, one block per instance, computes every reduction of the generic
# method and the termination code, instead of about 65 launches:
#
#   get_inf_pr!, get_inf_du!, get_inf_compl!        -> inf_pr, inf_du, inf_compl
#   best_complementarity update                      -> best_complementarity
#   dual_objective!                                  -> dual_obj
#   update_primal_infeasibility_certificate!         -> primal_cert_{res,margin}
#   update_dual_infeasibility_certificate!           -> dual_cert_{res,bound,margin}
#   compute_term_gpu!                                -> _term_gpu
#
# Each thread accumulates its rows of each vector (one pass per vector), the
# block reduces the partial results, and thread 1 combines them as the generic
# broadcasts do.

function _termination_kernel(cons, primal_dual, lb, ub, primal, ws, opt)
    T = eltype(cons.c)
    j = blockIdx().x
    @inbounds begin
        # constraint rows: c, y and rhs
        max_c = max_y = max_rhs = max_c_rhs = zero(T)  # |c|, |y|, |rhs|, |c + rhs|
        y_rhs = zero(T)                                # y' rhs
        @block_stride for i in 1:size(cons.c, 1)
            c = cons.c[i, j]
            y = cons.y[i, j]
            rhs = cons.rhs[i, j]
            max_c = max(max_c, abs(c))
            y_rhs += y * rhs
            max_y = max(max_y, abs(y))
            max_rhs = max(max_rhs, abs(rhs))
            max_c_rhs = max(max_c_rhs, abs(c + rhs))
        end
        # all rows of the full primal-dual vectors
        max_du = zero(T)       # |f - zl + zu + jacl|
        max_pcert = zero(T)    # |jacl - zl + zu|
        @block_stride for i in 1:size(primal_dual.f, 1)
            zl = primal_dual.zl[i, j]
            zu = primal_dual.zu[i, j]
            jacl = primal_dual.jacl[i, j]
            max_du = max(max_du, abs(primal_dual.f[i, j] - zl + zu + jacl))
            max_pcert = max(max_pcert, abs(jacl - zl + zu))
        end
        # lower bounds
        max_compl_l = max_zl = max_xl = max_ray_l = zero(T)  # |x - xl| zl, |zl|, |xl|, max(0, -x)
        zl_xl = zero(T)                                      # zl' xl
        @block_stride for i in 1:size(lb.x, 1)
            x = lb.x[i, j]
            xl = lb.xl[i, j]
            zl = lb.zl[i, j]
            max_compl_l = max(max_compl_l, abs(x - xl) * zl)
            zl_xl += zl * xl
            max_zl = max(max_zl, abs(zl))
            max_xl = max(max_xl, abs(xl))
            max_ray_l = max(max_ray_l, max(zero(T), -x))
        end
        # upper bounds
        max_compl_u = max_zu = max_xu = max_ray_u = zero(T)  # |xu - x| zu, |zu|, |xu|, max(0, x)
        zu_xu = zero(T)                                      # zu' xu
        @block_stride for i in 1:size(ub.x, 1)
            x = ub.x[i, j]
            xu = ub.xu[i, j]
            zu = ub.zu[i, j]
            max_compl_u = max(max_compl_u, abs(xu - x) * zu)
            zu_xu += zu * xu
            max_zu = max(max_zu, abs(zu))
            max_xu = max(max_xu, abs(xu))
            max_ray_u = max(max_ray_u, max(zero(T), x))
        end
        # primal variables (dual infeasibility certificate)
        max_x = max_f = zero(T)  # |x|, |f|
        f_x = zero(T)            # f' x
        @block_stride for i in 1:size(primal.x, 1)
            x = primal.x[i, j]
            f = primal.f[i, j]
            max_x = max(max_x, abs(x))
            f_x += f * x
            max_f = max(max_f, abs(f))
        end

        max_c = _block_max(max_c)
        y_rhs = _block_sum(y_rhs)
        max_y = _block_max(max_y)
        max_rhs = _block_max(max_rhs)
        max_c_rhs = _block_max(max_c_rhs)
        max_du = _block_max(max_du)
        max_pcert = _block_max(max_pcert)
        max_compl_l = _block_max(max_compl_l)
        zl_xl = _block_sum(zl_xl)
        max_zl = _block_max(max_zl)
        max_xl = _block_max(max_xl)
        max_ray_l = _block_max(max_ray_l)
        max_compl_u = _block_max(max_compl_u)
        zu_xu = _block_sum(zu_xu)
        max_zu = _block_max(max_zu)
        max_xu = _block_max(max_xu)
        max_ray_u = _block_max(max_ray_u)
        max_x = _block_max(max_x)
        f_x = _block_sum(f_x)
        max_f = _block_max(max_f)

        if threadIdx().x == 1

            # get_inf_pr!, get_inf_du!, get_inf_compl! and the scaling by the norms
            norm_c = max(one(T), ws.norm_c[1, j])
            inf_pr = max_c / max(one(T), ws.norm_b[1, j])
            inf_du = max_du / norm_c
            inf_compl = max(max_compl_l, max_compl_u) / norm_c
            ws.inf_pr[1, j] = inf_pr
            ws.inf_du[1, j] = inf_du
            ws.inf_compl[1, j] = inf_compl
            ws.best_complementarity[1, j] = min(ws.best_complementarity[1, j], inf_compl)

            # dual_objective!
            dual_obj = -y_rhs + zl_xl - zu_xu
            ws.dual_obj[1, j] = dual_obj

            # update_primal_infeasibility_certificate!
            cert_norm = max(one(T), max_y, max_zl, max_zu)
            data_norm = max(one(T), max_rhs, max_xl, max_xu)
            primal_cert_res = max_pcert / cert_norm
            primal_cert_margin = dual_obj / (cert_norm * data_norm)
            ws.primal_cert_res[1, j] = primal_cert_res
            ws.primal_cert_margin[1, j] = primal_cert_margin

            # update_dual_infeasibility_certificate! (ray = x / x_norm)
            if opt.islp
                x_norm = max(one(T), max_x)
                dual_cert_res = max_c_rhs / x_norm
                dual_cert_bound = max(max_ray_l, max_ray_u) / x_norm
                dual_cert_margin = -(f_x / x_norm) / max(one(T), max_f)
            else
                dual_cert_res = typemax(T)
                dual_cert_bound = typemax(T)
                dual_cert_margin = -typemax(T)
            end
            ws.dual_cert_res[1, j] = dual_cert_res
            ws.dual_cert_bound[1, j] = dual_cert_bound
            ws.dual_cert_margin[1, j] = dual_cert_margin

            # compute_term_gpu!
            obj_val = ws.obj_val[1, j]
            ws.term[1, j] = if ws.ls_error[1, j] > zero(Int32)
                Int64(Int(MadNLP.INTERNAL_ERROR))
            elseif max(inf_pr, inf_du, inf_compl) <= opt.tol
                Int64(Int(MadNLP.SOLVE_SUCCEEDED))
            elseif opt.certificates & (primal_cert_res <= opt.pcert_tol) & (primal_cert_margin > opt.pcert_tol)
                Int64(Int(MadNLP.INFEASIBLE_PROBLEM_DETECTED))
            elseif opt.certificates & (dual_cert_res <= opt.dcert_tol) & (dual_cert_bound <= opt.dcert_tol) &
                   (dual_cert_margin > opt.dcert_tol)
                Int64(Int(MadNLP.DIVERGING_ITERATES))
            elseif (inf_compl > opt.div_tol * ws.best_complementarity[1, j]) &
                   (dual_obj > max(opt.div_scale * abs(obj_val), one(T)))
                Int64(Int(MadNLP.INFEASIBLE_PROBLEM_DETECTED))
            elseif obj_val < -(opt.div_tol * max(opt.div_scale * abs(dual_obj), one(T)))
                Int64(Int(MadNLP.DIVERGING_ITERATES))
            else
                Int64(Int(MadNLP.REGULAR))
            end
        end
    end
    return nothing
end

function MadIPM.update_termination_criteria!(
    s::MadIPM.UniformBatchMPCSolver{T, MT},
) where {T, MT <: CuMatrix{T}}
    ws = s.workspace
    opt = s.opt
    full = MadNLP.full
    cons = (; c = full(s.c), y = full(s.y), rhs = full(s.rhs))
    primal_dual = (; f = full(s.f), zl = full(s.zl), zu = full(s.zu), jacl = full(s.jacl))
    primal = (; x = MadNLP.primal(s.x), f = MadNLP.primal(s.f))
    outputs = (;
        ws.norm_b, ws.norm_c, ws.best_complementarity, ws.inf_pr, ws.inf_du, ws.inf_compl, ws.dual_obj,
        ws.primal_cert_res, ws.primal_cert_margin, ws.dual_cert_res, ws.dual_cert_bound, ws.dual_cert_margin,
        ls_error = ws._ls_error, ws.obj_val, term = ws._term_gpu,
    )
    options = (;
        islp = s.bcb.nlp.meta.islp,
        certificates = opt.certificate_termination,
        tol = T(opt.tol),
        div_tol = T(opt.divergence_tol),
        div_scale = T(opt.divergence_scale),
        pcert_tol = T(opt.primal_infeasibility_cert_tol),
        dcert_tol = T(opt.dual_infeasibility_cert_tol),
    )
    _launch_per_instance(_termination_kernel, s,
        cons, primal_dual, _lower_rows(s), _upper_rows(s), primal, outputs, options)
    minimum!(ws._any_nonregular_gpu, ws._term_gpu)
    return
end
