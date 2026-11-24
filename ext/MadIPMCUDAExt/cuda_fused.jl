# Fused kernels for the uniform batch MPC iteration on CUDA. Each kernel
# replaces a sequence of broadcasts, reductions and small kernels of the
# generic batch code by one launch; most use one block per instance, with
# block reductions as in MadNLP#736. Each is switched by a FUSED_* flag (on by
# default) and falls back to the generic method when it does not apply.
#
# Fused termination criteria: one kernel computes, per instance (one block per
# column), every reduction of `update_termination_criteria!` (infeasibilities,
# complementarity, dual objective and both infeasibility certificates) and the
# termination code, instead of about 65 separate launches.

const FUSED_TERMINATION = Ref(true)

# threads of the one-block-per-instance kernels: enough to cover the longest
# vector of the problem, up to 1024
_fused_threads(s) = min(1024, max(64, 32 * cld(s.kkt.n_tot + s.kkt.m, 32 * 8)))

# launch a one-block-per-instance kernel with at most the threads its
# registers allow
function _launch_per_instance(f, s, args...)
    k = @cuda launch = false f(args...)
    threads = min(_fused_threads(s), CUDA.maxthreads(k))
    threads = max(32, 32 * (threads ÷ 32))
    k(args...; threads, blocks = s.batch_size)
    return
end

@inline _bred(op, v, neutral) = CUDACore.reduce_block(op, v, neutral, Val(true))

function _term_fused_kernel(
    c, y, rhs, f, zl, zu, jl, x_l, xl_l, zl_l, x_u, xu_u, zu_u, xp, fp,
    norm_b, norm_c, best, inf_pr, inf_du, inf_compl, dual_obj,
    pc_res, pc_margin, dc_res, dc_bound, dc_margin, ls_error, obj_val, term,
    islp::Bool, cert_enabled::Bool, tol::T, div_tol::T, ds::T, pcert_tol::T, dcert_tol::T,
) where {T}
    j = blockIdx().x
    tid = threadIdx().x
    nt = blockDim().x
    o = zero(T)
    @inbounds begin
        # constraint rows
        a_pr = o; a_yr = o; a_y = o; a_r = o; a_cr = o
        i = tid
        while i <= size(c, 1)
            ci = c[i, j]; yi = y[i, j]; ri = rhs[i, j]
            a_pr = max(a_pr, abs(ci)); a_yr += yi * ri; a_y = max(a_y, abs(yi))
            a_r = max(a_r, abs(ri)); a_cr = max(a_cr, abs(ci + ri))
            i += nt
        end
        # all primal-dual rows of the full vectors
        a_du = o; a_pcr = o
        i = tid
        while i <= size(f, 1)
            zli = zl[i, j]; zui = zu[i, j]; jli = jl[i, j]
            a_du = max(a_du, abs(f[i, j] - zli + zui + jli)); a_pcr = max(a_pcr, abs(jli - zli + zui))
            i += nt
        end
        # lower bounds
        a_cl = o; a_zxl = o; a_zl = o; a_xl = o; a_rl = o
        i = tid
        while i <= size(x_l, 1)
            xi = x_l[i, j]; xli = xl_l[i, j]; zi = zl_l[i, j]
            a_cl = max(a_cl, abs(xi - xli) * zi); a_zxl += zi * xli; a_zl = max(a_zl, abs(zi))
            a_xl = max(a_xl, abs(xli)); a_rl = max(a_rl, max(o, -xi))
            i += nt
        end
        # upper bounds
        a_cu = o; a_zxu = o; a_zu = o; a_xu = o; a_ru = o
        i = tid
        while i <= size(x_u, 1)
            xi = x_u[i, j]; xui = xu_u[i, j]; zi = zu_u[i, j]
            a_cu = max(a_cu, abs(xui - xi) * zi); a_zxu += zi * xui; a_zu = max(a_zu, abs(zi))
            a_xu = max(a_xu, abs(xui)); a_ru = max(a_ru, max(o, xi))
            i += nt
        end
        # primal variables (dual infeasibility certificate)
        a_x = o; a_fx = o; a_f = o
        i = tid
        while i <= size(xp, 1)
            xi = xp[i, j]; fi = fp[i, j]
            a_x = max(a_x, abs(xi)); a_fx += fi * xi; a_f = max(a_f, abs(fi))
            i += nt
        end
        a_pr = _bred(max, a_pr, o); a_yr = _bred(+, a_yr, o); a_y = _bred(max, a_y, o)
        a_r = _bred(max, a_r, o); a_cr = _bred(max, a_cr, o)
        a_du = _bred(max, a_du, o); a_pcr = _bred(max, a_pcr, o)
        a_cl = _bred(max, a_cl, o); a_zxl = _bred(+, a_zxl, o); a_zl = _bred(max, a_zl, o)
        a_xl = _bred(max, a_xl, o); a_rl = _bred(max, a_rl, o)
        a_cu = _bred(max, a_cu, o); a_zxu = _bred(+, a_zxu, o); a_zu = _bred(max, a_zu, o)
        a_xu = _bred(max, a_xu, o); a_ru = _bred(max, a_ru, o)
        a_x = _bred(max, a_x, o); a_fx = _bred(+, a_fx, o); a_f = _bred(max, a_f, o)
        if tid == 1
            l = one(T)
            nc = max(l, norm_c[1, j])
            ipr = a_pr / max(l, norm_b[1, j]); idu = a_du / nc; icp = max(a_cl, a_cu) / nc
            inf_pr[1, j] = ipr; inf_du[1, j] = idu; inf_compl[1, j] = icp
            best[1, j] = min(best[1, j], icp)
            dobj = -a_yr + a_zxl - a_zxu
            dual_obj[1, j] = dobj
            cert_norm = max(l, a_y, a_zl, a_zu)
            data_norm = max(l, a_r, a_xl, a_xu)
            pres = a_pcr / cert_norm
            pmar = dobj / (cert_norm * data_norm)
            pc_res[1, j] = pres; pc_margin[1, j] = pmar
            if islp
                xn = max(l, a_x)
                dres = a_cr / xn; dbnd = max(a_rl, a_ru) / xn; dmar = -(a_fx / xn) / max(l, a_f)
            else
                dres = typemax(T); dbnd = typemax(T); dmar = -typemax(T)
            end
            dc_res[1, j] = dres; dc_bound[1, j] = dbnd; dc_margin[1, j] = dmar
            ov = obj_val[1, j]
            term[1, j] = if ls_error[1, j] > zero(Int32)
                Int64(Int(MadNLP.INTERNAL_ERROR))
            elseif max(ipr, idu, icp) <= tol
                Int64(Int(MadNLP.SOLVE_SUCCEEDED))
            elseif cert_enabled & (pres <= pcert_tol) & (pmar > pcert_tol)
                Int64(Int(MadNLP.INFEASIBLE_PROBLEM_DETECTED))
            elseif cert_enabled & (dres <= dcert_tol) & (dbnd <= dcert_tol) & (dmar > dcert_tol)
                Int64(Int(MadNLP.DIVERGING_ITERATES))
            elseif (icp > div_tol * best[1, j]) & (dobj > max(ds * abs(ov), l))
                Int64(Int(MadNLP.INFEASIBLE_PROBLEM_DETECTED))
            elseif ov < -(div_tol * max(ds * abs(dobj), l))
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
    FUSED_TERMINATION[] ||
        return invoke(MadIPM.update_termination_criteria!, Tuple{MadIPM.AbstractBatchMPCSolver}, s)
    ws = s.workspace; opt = s.opt
    x, xl, xu, zl, zu = s.x, s.xl, s.xu, s.zl, s.zu
    args = (
        MadNLP.full(s.c), MadNLP.full(s.y), MadNLP.full(s.rhs),
        MadNLP.full(s.f), MadNLP.full(zl), MadNLP.full(zu), MadNLP.full(s.jacl),
        MadIPM.lower(x), MadIPM.lower(xl), MadIPM.lower(zl),
        MadIPM.upper(x), MadIPM.upper(xu), MadIPM.upper(zu),
        MadNLP.primal(x), MadNLP.primal(s.f),
        ws.norm_b, ws.norm_c, ws.best_complementarity, ws.inf_pr, ws.inf_du, ws.inf_compl, ws.dual_obj,
        ws.primal_cert_res, ws.primal_cert_margin, ws.dual_cert_res, ws.dual_cert_bound, ws.dual_cert_margin,
        ws._ls_error, ws.obj_val, ws._term_gpu,
        s.bcb.nlp.meta.islp, opt.certificate_termination,
        T(opt.tol), T(opt.divergence_tol), T(opt.divergence_scale),
        T(opt.primal_infeasibility_cert_tol), T(opt.dual_infeasibility_cert_tol),
    )
    _launch_per_instance(_term_fused_kernel, s, args...)
    minimum!(ws._any_nonregular_gpu, ws._term_gpu)
    return
end

# Fused tail of the prediction step: fraction to the boundary (four step
# lengths), zero_inactive_step!, the affine complementarity, the correction
# terms, the complementarity measure and Mehrotra's barrier update, in one
# kernel with one block per instance.
const FUSED_PREDICTION = Ref(true)

function _pred_tail_kernel(
    dx_l, x_l, xl_l, dz_l, z_l, dx_u, x_u, xu_u, dz_u, z_u, mask, tau,
    a_xl, a_zl, a_xu, a_zu, alpha_p, alpha_d, corr_l, corr_u,
    mu_affine, mu_curr, mu_batch, mu_min::T,
) where {T}
    j = blockIdx().x; tid = threadIdx().x; nt = blockDim().x
    sh = CuStaticSharedArray(T, 2)
    inf = T(Inf); o = zero(T); l = one(T)
    nlb = size(x_l, 1); nub = size(x_u, 1)
    @inbounds begin
        τ = l
        bxl = inf; bzl = inf
        i = tid
        while i <= nlb
            d = dx_l[i, j]; d < o && (bxl = min(bxl, (-x_l[i, j] + xl_l[i, j]) * τ / d))
            e = dz_l[i, j]; e < o && (bzl = min(bzl, -z_l[i, j] * τ / e))
            i += nt
        end
        bxu = inf; bzu = inf
        i = tid
        while i <= nub
            d = dx_u[i, j]; d > o && (bxu = min(bxu, (-x_u[i, j] + xu_u[i, j]) * τ / d))
            e = dz_u[i, j]; z = z_u[i, j]
            (e < o && z + e < o) && (bzu = min(bzu, -z * τ / e))
            i += nt
        end
        bxl = _bred(min, bxl, inf); bzl = _bred(min, bzl, inf)
        bxu = _bred(min, bxu, inf); bzu = _bred(min, bzu, inf)
        if tid == 1
            tau[1, j] = l
            nlb == 0 && (bxl = l; bzl = l)
            nub == 0 && (bxu = l; bzu = l)
            a_xl[1, j] = bxl; a_zl[1, j] = bzl; a_xu[1, j] = bxu; a_zu[1, j] = bzu
            m = mask[1, j]
            ap = min(bxl, bxu, l) * m; ad = min(bzl, bzu, l) * m
            alpha_p[1, j] = ap; alpha_d[1, j] = ad
            sh[1] = ap; sh[2] = ad
        end
        sync_threads()
        ap = sh[1]; ad = sh[2]
        s_l = o; c_l = o
        i = tid
        while i <= nlb
            x = x_l[i, j]; xl = xl_l[i, j]; z = z_l[i, j]; d = dx_l[i, j]; e = dz_l[i, j]
            s_l += (x + ap * d - xl) * (z + ad * e)
            c_l += (x - xl) * z
            corr_l[i, j] = d * e
            i += nt
        end
        s_u = o; c_u = o
        i = tid
        while i <= nub
            x = x_u[i, j]; xu = xu_u[i, j]; z = z_u[i, j]; d = dx_u[i, j]; e = dz_u[i, j]
            s_u += (xu - (x + ap * d)) * (z + ad * e)
            c_u += (xu - x) * z
            corr_u[i, j] = d * e
            i += nt
        end
        s_l = _bred(+, s_l, o); s_u = _bred(+, s_u, o); c_l = _bred(+, c_l, o); c_u = _bred(+, c_u, o)
        if tid == 1
            nb = nlb + nub
            if nb == 0
                mu_affine[1, j] = o; mu_curr[1, j] = o; mu_batch[1, j] = max(mu_min, o)
            else
                maff = (s_l + s_u) / nb; mc = (c_l + c_u) / nb
                mu_affine[1, j] = maff; mu_curr[1, j] = mc
                mb = clamp((maff / mc)^3, T(1e-6), T(10.0))
                mu_batch[1, j] = max(mu_min, mb * mc)
            end
        end
    end
    return nothing
end

function MadIPM.prediction_step!(s::MadIPM.UniformBatchMPCSolver{T, MT}) where {T, MT <: CuMatrix{T}}
    (FUSED_PREDICTION[] && s.opt.barrier_update isa MadIPM.Mehrotra) ||
        return invoke(MadIPM.prediction_step!, Tuple{MadIPM.AbstractBatchMPCSolver}, s)
    ws = s.workspace; d = s.d
    MadIPM.affine_direction!(s)
    _launch_per_instance(_pred_tail_kernel, s, 
        MadIPM.xp_lr(d), MadIPM.lower(s.x), MadIPM.lower(s.xl), MadNLP.dual_lb(d), MadIPM.lower(s.zl),
        MadIPM.xp_ur(d), MadIPM.upper(s.x), MadIPM.upper(s.xu), MadNLP.dual_ub(d), MadIPM.upper(s.zu),
        ws.active_mask, ws.tau,
        ws.alpha_xl, ws.alpha_zl, ws.alpha_xu, ws.alpha_zu, ws.alpha_p, ws.alpha_d,
        MadNLP.full(s.correction_lb), MadNLP.full(s.correction_ub),
        ws.mu_affine, ws.mu_curr, ws.mu_batch, T(s.opt.mu_min),
    )
    return
end

# Fused residual check of solve_system!: masks w and p, takes the per-instance
# max norms and updates the linear-solve error flag in one kernel.
const FUSED_RESIDUAL = Ref(true)

function _residual_kernel(w, p, mask, norm_w, norm_p, ls_error, check::Bool, tol::T) where {T}
    j = blockIdx().x; tid = threadIdx().x; nt = blockDim().x
    o = zero(T)
    @inbounds begin
        m = mask[1, j]
        aw = o; ap = o
        i = tid
        while i <= size(w, 1)
            wi = w[i, j] * m; pi = p[i, j] * m
            w[i, j] = abs(pi); p[i, j] = pi   # as the generic path, which reuses w
            aw = max(aw, abs(wi)); ap = max(ap, abs(pi))
            i += nt
        end
        aw = _bred(max, aw, o); ap = _bred(max, ap, o)
        if tid == 1
            r = aw / max(one(T), ap)
            norm_w[1, j] = r; norm_p[1, j] = ap
            ls_error[1, j] |= isnan(r) | (check & (r > tol))
        end
    end
    return nothing
end

function MadIPM.solve_system!(
    d::MadIPM.BatchUnreducedKKTVector{T},
    s::MadIPM.UniformBatchMPCSolver{T, MT},
    p::MadIPM.BatchUnreducedKKTVector{T},
) where {T, MT <: CuMatrix{T}}
    FUSED_RESIDUAL[] || return invoke(MadIPM.solve_system!,
        Tuple{MadIPM.BatchUnreducedKKTVector{T}, MadIPM.AbstractBatchMPCSolver{T}, MadIPM.BatchUnreducedKKTVector{T}}, d, s, p)
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(s.kkt, s)
    w = s._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, s.kkt, d, -one(T), one(T))
    ws = s.workspace
    _launch_per_instance(_residual_kernel, s, 
        MadNLP.full(w), MadNLP.full(p), ws.active_mask, ws._norm_gpu_w, ws._norm_gpu_p, ws._ls_error,
        s.opt.check_residual, T(s.opt.tol_linear_solve))
    return d
end

# Fused factorization setup for FixedRegularization: the regularization update
# and set_aug_diagonal_reg! (masked) in one kernel, one block per instance.
# The bound terms are subtracted from the primal diagonal after it is reset,
# so the two phases are separated by a block barrier.
const FUSED_FACTORIZE_SETUP = Ref(true)

function _factorize_setup_kernel(
    nzVals, reg, l_diag, u_diag, l_lower, u_lower, ind_lb, ind_ub,
    xl_r, x_lr, zl_r, xu_r, x_ur, zu_r, mask, del_w, del_c,
    delta_p::T, delta_d::T, n_tot::Int, du_off::Int, m_rows::Int,
) where {T}
    j = blockIdx().x; tid = threadIdx().x; nt = blockDim().x
    @inbounds if mask[1, j] == one(T)
        dw = delta_p; dc = delta_d
        if tid == 1
            del_w[1, j] = dw; del_c[1, j] = dc
        end
        i = tid
        while i <= n_tot
            reg[i, j] = dw; nzVals[i, j] = dw
            i += nt
        end
        i = tid
        while i <= m_rows
            nzVals[du_off + i, j] = dc
            i += nt
        end
        i = tid
        while i <= size(l_diag, 1)
            l_diag[i, j] = xl_r[i, j] - x_lr[i, j]; l_lower[i, j] = zl_r[i, j]
            i += nt
        end
        i = tid
        while i <= size(u_diag, 1)
            u_diag[i, j] = x_ur[i, j] - xu_r[i, j]; u_lower[i, j] = zu_r[i, j]
            i += nt
        end
        sync_threads()
        # indices are unique within each bound set, but a variable can have both
        # bounds: the two sets are applied one after the other
        k = tid
        while k <= size(l_diag, 1)
            r = ind_lb[k]; nzVals[r, j] -= l_lower[k, j] / l_diag[k, j]
            k += nt
        end
        sync_threads()
        k = tid
        while k <= size(u_diag, 1)
            r = ind_ub[k]; nzVals[r, j] -= u_lower[k, j] / u_diag[k, j]
            k += nt
        end
    end
    return nothing
end

function MadIPM.factorize_system!(s::MadIPM.UniformBatchMPCSolver{T, MT}) where {T, MT <: CuMatrix{T}}
    reg = s.opt.regularization
    (FUSED_FACTORIZE_SETUP[] && reg isa MadIPM.FixedRegularization && s.kkt.batch_solver isa CUDSSSolver) ||
        return invoke(MadIPM.factorize_system!, Tuple{MadIPM.AbstractBatchMPCSolver}, s)
    kkt = s.kkt
    _launch_per_instance(_factorize_setup_kernel, s, 
        kkt.nzVals, kkt.reg, kkt.l_diag, kkt.u_diag, kkt.l_lower, kkt.u_lower,
        MadIPM._get_ind_lb(s), MadIPM._get_ind_ub(s),
        MadIPM.lower(s.xl), MadIPM.lower(s.x), MadIPM.lower(s.zl),
        MadIPM.upper(s.xu), MadIPM.upper(s.x), MadIPM.upper(s.zu),
        s.workspace.active_mask, s.del_w, s.del_c, T(reg.delta_p), T(reg.delta_d),
        kkt.n_tot, size(kkt.nzVals, 1) - kkt.m, kkt.m)
    # cuDSS always reports success (is_factorized), so there is no retry
    MadNLP.factorize_wrapper!(s)
    return
end

# Fused right-hand sides of the predictor and the corrector: the four blocks
# of p in one kernel (every entry of p is written, so no fill is needed).
const FUSED_RHS = Ref(true)

function _rhs_kernel(px, py, pzl, pzu, f, zl, zu, jacl, c, xl_r, x_lr, zl_r, xu_r, x_ur, zu_r,
                     mu, corr_l, corr_u, correction::Bool)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = blockIdx().y
    @inbounds begin
        i <= size(px, 1) && (px[i, j] = -f[i, j] + zl[i, j] - zu[i, j] - jacl[i, j])
        i <= size(py, 1) && (py[i, j] = -c[i, j])
        if i <= size(pzl, 1)
            v = (xl_r[i, j] - x_lr[i, j]) * zl_r[i, j]
            correction && (v = v + mu[1, j] - corr_l[i, j])
            pzl[i, j] = v
        end
        if i <= size(pzu, 1)
            v = (xu_r[i, j] - x_ur[i, j]) * zu_r[i, j]
            correction && (v = v - mu[1, j] - corr_u[i, j])
            pzu[i, j] = v
        end
    end
    return nothing
end

function _fused_rhs!(s, correction::Bool)
    p = s.p
    px, py, pzl, pzu = MadNLP.primal(p), MadNLP.dual(p), MadNLP.dual_lb(p), MadNLP.dual_ub(p)
    nrow = max(size(px, 1), size(py, 1), size(pzl, 1), size(pzu, 1))
    @cuda threads = 256 blocks = (cld(nrow, 256), s.batch_size) _rhs_kernel(
        px, py, pzl, pzu, MadNLP.primal(s.f), MadNLP.full(s.zl), MadNLP.full(s.zu), MadNLP.full(s.jacl),
        MadNLP.full(s.c), MadIPM.lower(s.xl), MadIPM.lower(s.x), MadIPM.lower(s.zl),
        MadIPM.upper(s.xu), MadIPM.upper(s.x), MadIPM.upper(s.zu),
        s.workspace.mu_batch, MadNLP.full(s.correction_lb), MadNLP.full(s.correction_ub), correction)
    return
end

function MadIPM.set_predictive_rhs!(s::MadIPM.UniformBatchMPCSolver{T, MT}, kkt::MadIPM.AbstractBatchKKTSystem) where {T, MT <: CuMatrix{T}}
    FUSED_RHS[] || return invoke(MadIPM.set_predictive_rhs!, Tuple{MadIPM.AbstractBatchMPCSolver, MadIPM.AbstractBatchKKTSystem}, s, kkt)
    _fused_rhs!(s, false)
end

function MadIPM.set_correction_rhs!(s::MadIPM.UniformBatchMPCSolver{T, MT}, kkt::MadIPM.AbstractBatchKKTSystem,
                                    mu, correction_lb, correction_ub, ind_lb, ind_ub) where {T, MT <: CuMatrix{T}}
    (FUSED_RHS[] && mu === s.workspace.mu_batch && correction_lb === MadNLP.full(s.correction_lb) &&
     correction_ub === MadNLP.full(s.correction_ub)) ||
        return invoke(MadIPM.set_correction_rhs!, Tuple{MadIPM.AbstractBatchMPCSolver, MadIPM.AbstractBatchKKTSystem, Any, Any, Any, Any, Any},
                      s, kkt, mu, correction_lb, correction_ub, ind_lb, ind_ub)
    _fused_rhs!(s, true)
end

# Fused apply_step!: the primal, dual and bound-multiplier updates in one
# kernel, then the bound adjustment (which reads the updated x) in another.
const FUSED_APPLY = Ref(true)

function _apply_kernel(x, dx, y, dy, zl_r, dzl, zu_r, dzu, alpha_p, alpha_d)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = blockIdx().y
    @inbounds begin
        ap = alpha_p[1, j]; ad = alpha_d[1, j]
        i <= size(x, 1) && (x[i, j] += ap * dx[i, j])
        i <= size(y, 1) && (y[i, j] += ad * dy[i, j])
        i <= size(zl_r, 1) && (zl_r[i, j] += ad * dzl[i, j])
        i <= size(zu_r, 1) && (zu_r[i, j] += ad * dzu[i, j])
    end
    return nothing
end

function _adjust_kernel(x_lr, xl_r, x_ur, xu_r, mu, mask)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = blockIdx().y
    T = eltype(xl_r)
    c2 = eps(T)^(T(3) / T(4)); c1 = eps(T)
    @inbounds if mask[1, j] != 0
        m = mu[1, j]
        if i <= size(x_lr, 1)
            x = x_lr[i, j]; xl = xl_r[i, j]
            x - xl < c1 * m && (xl_r[i, j] = xl - c2 * max(one(T), abs(x)))
        end
        if i <= size(x_ur, 1)
            x = x_ur[i, j]; xu = xu_r[i, j]
            xu - x < c1 * m && (xu_r[i, j] = xu + c2 * max(one(T), abs(x)))
        end
    end
    return nothing
end

function MadIPM.apply_step!(s::MadIPM.UniformBatchMPCSolver{T, MT}) where {T, MT <: CuMatrix{T}}
    FUSED_APPLY[] || return invoke(MadIPM.apply_step!, Tuple{MadIPM.AbstractBatchMPCSolver}, s)
    ws = s.workspace; d = s.d
    x = MadNLP.full(s.x); y = MadNLP.full(s.y)
    zl_r = MadIPM.lower(s.zl); zu_r = MadIPM.upper(s.zu)
    n1 = max(size(x, 1), size(y, 1), size(zl_r, 1), size(zu_r, 1))
    @cuda threads = 256 blocks = (cld(n1, 256), s.batch_size) _apply_kernel(
        x, MadNLP.primal(d), y, MadNLP.dual(d), zl_r, MadNLP.dual_lb(d), zu_r, MadNLP.dual_ub(d),
        ws.alpha_p, ws.alpha_d)
    x_lr = MadIPM.lower(s.x); x_ur = MadIPM.upper(s.x)
    n2 = max(size(x_lr, 1), size(x_ur, 1), 1)
    @cuda threads = 256 blocks = (cld(n2, 256), s.batch_size) _adjust_kernel(
        x_lr, MadIPM.lower(s.xl), x_ur, MadIPM.upper(s.xu), ws.mu_batch, ws.active_mask)
    MadIPM.increment_k!(s)
    return
end

# Fused first part of update_step!(::MehrotraAdaptiveStep): the fraction to
# the boundary (tau = 1) and the affine complementarity at those steps, scaled
# by 1/gamma_a into mu_curr, in one kernel; Mehrotra's step kernel follows.
function _step_pre_kernel(
    dx_l, x_l, xl_l, dz_l, z_l, dx_u, x_u, xu_u, dz_u, z_u, tau,
    a_xl, a_zl, a_xu, a_zu, alpha_p, alpha_d, sum_l, sum_u, mu_affine, mu_full, gamma_a::T,
) where {T}
    j = blockIdx().x; tid = threadIdx().x; nt = blockDim().x
    sh = CuStaticSharedArray(T, 2)
    inf = T(Inf); o = zero(T); l = one(T)
    nlb = size(x_l, 1); nub = size(x_u, 1)
    @inbounds begin
        bxl = inf; bzl = inf
        i = tid
        while i <= nlb
            d = dx_l[i, j]; d < o && (bxl = min(bxl, (-x_l[i, j] + xl_l[i, j]) / d))
            e = dz_l[i, j]; e < o && (bzl = min(bzl, -z_l[i, j] / e))
            i += nt
        end
        bxu = inf; bzu = inf
        i = tid
        while i <= nub
            d = dx_u[i, j]; d > o && (bxu = min(bxu, (-x_u[i, j] + xu_u[i, j]) / d))
            e = dz_u[i, j]; z = z_u[i, j]
            (e < o && z + e < o) && (bzu = min(bzu, -z / e))
            i += nt
        end
        bxl = _bred(min, bxl, inf); bzl = _bred(min, bzl, inf)
        bxu = _bred(min, bxu, inf); bzu = _bred(min, bzu, inf)
        if tid == 1
            tau[1, j] = l
            nlb == 0 && (bxl = l; bzl = l)
            nub == 0 && (bxu = l; bzu = l)
            a_xl[1, j] = bxl; a_zl[1, j] = bzl; a_xu[1, j] = bxu; a_zu[1, j] = bzu
            ap = min(bxl, bxu, l); ad = min(bzl, bzu, l)
            alpha_p[1, j] = ap; alpha_d[1, j] = ad
            sh[1] = ap; sh[2] = ad
        end
        sync_threads()
        ap = sh[1]; ad = sh[2]
        s_l = o
        i = tid
        while i <= nlb
            s_l += (x_l[i, j] + ap * dx_l[i, j] - xl_l[i, j]) * (z_l[i, j] + ad * dz_l[i, j])
            i += nt
        end
        s_u = o
        i = tid
        while i <= nub
            s_u += (xu_u[i, j] - (x_u[i, j] + ap * dx_u[i, j])) * (z_u[i, j] + ad * dz_u[i, j])
            i += nt
        end
        s_l = _bred(+, s_l, o); s_u = _bred(+, s_u, o)
        if tid == 1
            nb = nlb + nub
            sum_l[1, j] = s_l; sum_u[1, j] = s_u
            maff = nb == 0 ? o : (s_l + s_u) / nb
            mu_affine[1, j] = maff
            mu_full[1, j] = maff / gamma_a
        end
    end
    return nothing
end

const FUSED_STEP = Ref(true)

function MadIPM.update_step!(rule::MadIPM.MehrotraAdaptiveStep, s::MadIPM.UniformBatchMPCSolver{T, MT}) where {T, MT <: CuMatrix{T}}
    FUSED_STEP[] || return invoke(MadIPM.update_step!, Tuple{MadIPM.MehrotraAdaptiveStep, MadIPM.AbstractBatchMPCSolver}, rule, s)
    ws = s.workspace; d = s.d
    gamma_f = T(rule.gamma_f); gamma_a = one(T) / (one(T) - gamma_f)
    x_l, xl_l, z_l = MadIPM.lower(s.x), MadIPM.lower(s.xl), MadIPM.lower(s.zl)
    x_u, xu_u, z_u = MadIPM.upper(s.x), MadIPM.upper(s.xu), MadIPM.upper(s.zu)
    dx_l, dz_l, dx_u, dz_u = MadIPM.xp_lr(d), MadNLP.dual_lb(d), MadIPM.xp_ur(d), MadNLP.dual_ub(d)
    _launch_per_instance(_step_pre_kernel, s, 
        dx_l, x_l, xl_l, dz_l, z_l, dx_u, x_u, xu_u, dz_u, z_u, ws.tau,
        ws.alpha_xl, ws.alpha_zl, ws.alpha_xu, ws.alpha_zu, ws.alpha_p, ws.alpha_d,
        ws.sum_lb, ws.sum_ub, ws.mu_affine, ws.mu_curr, gamma_a)
    MadIPM._mehrotra_step!(
        ws.alpha_p, ws.alpha_d, ws.mu_curr, gamma_f,
        dx_l, x_l, xl_l, d.nlb, dz_l, z_l,
        dx_u, x_u, xu_u, d.nub, dz_u, z_u,
        d.values, d.ind_lb, d.ind_ub, d.n + d.m, d.n + d.m + d.nlb)
    return
end

# Fused reduction of the bound rows into the primal rows of the right-hand
# side (lower, then upper bounds: a variable can have both), and fused
# recovery of the bound-multiplier steps (independent rows), one launch each.
const FUSED_AUG = Ref(true)

function _reduce_rhs_fused_kernel(values, ind_lb, lb_off::Int, l_diag, ind_ub, ub_off::Int, u_diag)
    j = blockIdx().x; tid = threadIdx().x; nt = blockDim().x
    @inbounds begin
        k = tid
        while k <= length(ind_lb)
            values[ind_lb[k], j] -= values[lb_off + k, j] / l_diag[k, j]
            k += nt
        end
        sync_threads()
        k = tid
        while k <= length(ind_ub)
            values[ind_ub[k], j] -= values[ub_off + k, j] / u_diag[k, j]
            k += nt
        end
    end
    return nothing
end

function _finish_aug_fused_kernel(values, ind_lb, lb_off::Int, l_lower, l_diag, ind_ub, ub_off::Int, u_lower, u_diag)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = blockIdx().y
    @inbounds begin
        if i <= length(ind_lb)
            values[lb_off + i, j] = (-values[lb_off + i, j] + l_lower[i, j] * values[ind_lb[i], j]) / l_diag[i, j]
        end
        if i <= length(ind_ub)
            values[ub_off + i, j] = (values[ub_off + i, j] - u_lower[i, j] * values[ind_ub[i], j]) / u_diag[i, j]
        end
    end
    return nothing
end

function _fused_reduce_rhs!(values, ind_lb, lb_off, l_diag, ind_ub, ub_off, u_diag)
    length(ind_lb) + length(ind_ub) == 0 && return
    k = @cuda launch = false _reduce_rhs_fused_kernel(values, ind_lb, Int(lb_off), l_diag, ind_ub, Int(ub_off), u_diag)
    threads = min(CUDA.maxthreads(k), max(32, 32 * cld(max(length(ind_lb), length(ind_ub)), 32)), 1024)
    k(values, ind_lb, Int(lb_off), l_diag, ind_ub, Int(ub_off), u_diag; threads, blocks = size(values, 2))
    return
end

function _fused_finish_aug!(values, ind_lb, lb_off, l_lower, l_diag, ind_ub, ub_off, u_lower, u_diag)
    n = max(length(ind_lb), length(ind_ub))
    n == 0 && return
    @cuda threads = 256 blocks = (cld(n, 256), size(values, 2)) _finish_aug_fused_kernel(
        values, ind_lb, Int(lb_off), l_lower, l_diag, ind_ub, Int(ub_off), u_lower, u_diag)
    return
end
