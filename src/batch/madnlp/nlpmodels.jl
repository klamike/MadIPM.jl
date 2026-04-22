# ============================================================================
# MadNLP callback plumbing for `UniformBatchCallback`.
#
# Two dispatches per hook: the default (`NoFixedVariables`-style) path passes
# the variable block through unchanged; the `MakeParameter` path scatters the
# solver's free-slot values into the full-variable buffer and pins the fixed
# slots to `lvar`.
# ============================================================================

# Shorthand for the `MakeParameter` fixed-variable handler dispatch.
const _UBC_MP = UniformBatchCallback{T, VT, MT, VI, BM, FH, EH} where
    {T, VT, MT, VI, BM, FH <: MadNLP.MakeParameter, EH}

# ---------- x / variable reshape ----------

MadNLP._update_x!(::UniformBatchCallback, x) = x

function MadNLP._update_x!(bcb::_UBC_MP, x)
    fh = bcb.fixed_handler
    BX = reshape(fh.x_full, bcb.nlp.meta.nvar, bcb.batch_size)
    view(BX, fh.fixed, :) .= view(bcb.nlp.meta.lvar, fh.fixed, :)
    view(BX, fh.free,  :) .= x
    return BX
end

MadNLP.unpack_x!(X_full::AbstractMatrix, ::UniformBatchCallback, x::BatchPrimalVector) =
    (X_full .= MadNLP.variable(x); X_full)

function MadNLP.unpack_x!(X_full::AbstractMatrix, bcb::_UBC_MP, x::BatchPrimalVector)
    fh = bcb.fixed_handler
    X_full[fh.free,  :] .= MadNLP.variable(x)
    X_full[fh.fixed, :] .= view(bcb.nlp.meta.lvar, fh.fixed, :)
    return X_full
end

# ---------- dual (bound multipliers) ----------

MadNLP.unpack_z!(Z_full::AbstractMatrix, bcb::UniformBatchCallback, z_free) =
    (Z_full .= z_free ./ bcb.obj_scale; Z_full)

function MadNLP.unpack_z!(Z_full::AbstractMatrix{T}, bcb::_UBC_MP, z_free) where {T}
    fill!(Z_full, zero(T))
    Z_full[bcb.fixed_handler.free, :] .= z_free ./ bcb.obj_scale
    return Z_full
end

# ---------- accessors ----------

MadNLP.get_x0(bcb::_UBC_MP) = view(bcb.nlp.meta.x0, bcb.fixed_handler.free, :)
MadNLP.get_y0(bcb::UniformBatchCallback) = bcb.nlp.meta.y0

MadNLP.get_lvar(bcb::UniformBatchCallback) = bcb.nlp.meta.lvar
MadNLP.get_lvar(bcb::_UBC_MP)              = view(bcb.nlp.meta.lvar, bcb.fixed_handler.free, :)
MadNLP.get_uvar(bcb::UniformBatchCallback) = bcb.nlp.meta.uvar
MadNLP.get_uvar(bcb::_UBC_MP)              = view(bcb.nlp.meta.uvar, bcb.fixed_handler.free, :)
MadNLP.get_lcon(bcb::UniformBatchCallback) = bcb.nlp.meta.lcon
MadNLP.get_ucon(bcb::UniformBatchCallback) = bcb.nlp.meta.ucon

MadNLP.unpack_y!(y_full, bcb::UniformBatchCallback, y) =
    @. y_full = y * bcb.con_scale * bcb.obj_sign / bcb.obj_scale

function unpack_obj!(dst, bcb::UniformBatchCallback, obj_val)
    dst_mat = reshape(dst, 1, length(dst))
    @. dst_mat = bcb.obj_sign * obj_val / bcb.obj_scale
    return dst
end

function MadNLP.unpack_cons!(c_full, bcb::UniformBatchCallback, c, rhs, ind_ineq, slack)
    c_full .= c ./ bcb.con_scale .+ rhs
    isempty(ind_ineq) || (view(c_full, ind_ineq, :) .+= slack)
    return c_full
end

# ---------- low-level eval wrappers ----------

function MadNLP._eval_f_wrapper(bcb::UniformBatchCallback, bx::AbstractMatrix,
                                 bf::AbstractVector)
    NLPModels.obj!(bcb.nlp, bx, bf)
    bf .*= vec(bcb.obj_scale)
    return bf
end

function MadNLP._eval_cons_wrapper!(bcb::UniformBatchCallback, bx::AbstractMatrix,
                                     bc::AbstractMatrix)
    NLPModels.cons!(bcb.nlp, bx, bc)
    bc .*= bcb.con_scale
    return bc
end

function MadNLP._eval_grad_f_wrapper!(bcb::UniformBatchCallback, bx::AbstractMatrix,
                                       bg::AbstractMatrix)
    NLPModels.grad!(bcb.nlp, bx, bg)
    bg .*= bcb.obj_scale
    return bg
end

function MadNLP._eval_grad_f_wrapper!(bcb::_UBC_MP, bx::AbstractMatrix,
                                       bg::AbstractMatrix)
    fh = bcb.fixed_handler
    GF = reshape(fh.g_full, bcb.nlp.meta.nvar, bcb.batch_size)
    NLPModels.grad!(bcb.nlp, bx, GF)
    view(bg, 1:bcb.nvar, :) .= view(GF, fh.free, :) .* bcb.obj_scale
    return bg
end

function MadNLP._eval_jac_wrapper!(bcb::UniformBatchCallback, bx::AbstractMatrix,
                                    jac::AbstractMatrix)
    NLPModels.jac_coord!(bcb.nlp, bx, jac)
    jac .*= bcb.jac_scale
    return jac
end

function MadNLP._eval_jac_wrapper!(bcb::_UBC_MP, bx::AbstractMatrix,
                                    jac::AbstractMatrix)
    NLPModels.jac_coord!(bcb.nlp, bx, jac)
    jac_free = view(jac, bcb.fixed_handler.ind_jac_free, :)
    jac_free .*= bcb.jac_scale
    return jac_free
end

function MadNLP._eval_lag_hess_wrapper!(
    bcb::UniformBatchCallback, bx::AbstractMatrix, y_mat::AbstractMatrix,
    bv::AbstractMatrix, hess::AbstractMatrix;
    obj_weight::AbstractVector = vec(bcb.obj_scale),
)
    bv .= y_mat .* bcb.con_scale
    NLPModels.hess_coord!(bcb.nlp, bx, bv, obj_weight, hess)
    return hess
end

function MadNLP._eval_lag_hess_wrapper!(
    bcb::_UBC_MP, bx::AbstractMatrix, y_mat::AbstractMatrix,
    bv::AbstractMatrix, hess::AbstractMatrix;
    obj_weight::AbstractVector = vec(bcb.obj_scale),
)
    bv .= y_mat .* bcb.con_scale
    NLPModels.hess_coord!(bcb.nlp, bx, bv, obj_weight, bcb.hess_buffer)
    hess .= view(bcb.hess_buffer, bcb.fixed_handler.ind_hess_free, :)
    return hess
end

# ---------- solver-level eval wrappers ----------

function MadNLP.eval_f_wrapper(solver::UniformBatchMPCSolver, bx::AbstractMatrix)
    state = solver.state
    ws    = state.workspace
    bcb   = solver.problem.bcb

    state.cnt.eval_function_time += @elapsed begin
        MadNLP._eval_f_wrapper(bcb, bx, ws.bf)
        ws.bf         .*= vec(bcb.obj_sign)
        vec(ws.obj_val) .= ws.bf
    end
    state.cnt.obj_cnt += 1
    return nothing
end

function MadNLP.eval_cons_wrapper!(solver::UniformBatchMPCSolver, bx::AbstractMatrix)
    state = solver.state
    bcb   = solver.problem.bcb
    c     = MadNLP.full(state.c)

    state.cnt.eval_function_time += @elapsed begin
        MadNLP._eval_cons_wrapper!(bcb, bx, c)
        isempty(bcb.ind_ineq) ||
            (view(c, bcb.ind_ineq, :) .-= MadNLP.slack(state.x))
        c .-= MadNLP.full(state.rhs)
    end
    state.cnt.con_cnt += 1
    return nothing
end

function MadNLP.eval_grad_f_wrapper!(solver::UniformBatchMPCSolver, bx::AbstractMatrix)
    state = solver.state
    ws    = state.workspace
    bcb   = solver.problem.bcb

    state.cnt.eval_function_time += @elapsed begin
        MadNLP._eval_grad_f_wrapper!(bcb, bx, ws.bg)
        BG = view(ws.bg, 1:bcb.nvar, :)
        BG .*= bcb.obj_sign
        copyto!(MadNLP.variable(state.f), BG)
    end
    state.cnt.obj_grad_cnt += 1
    return nothing
end
