# Std-form batch initializer: only lower bounds and equality constraints, so
# no uvar / xu storage is needed.

function MadNLP.set_initial_bounds!(xl::AbstractMatrix{T}, tol) where {T}
    if tol > zero(T)
        xl .= xl .- max.(one(T), abs.(xl)) .* tol
    end
end

function MadNLP.initialize!(
    bcb::UniformBatchCallback{T},
    x,
    xl,
    y,
    rhs,
    ind_ineq,
    bx_buffer;
    tol = 1e-8,
    bound_push = 1e-2,
    bound_fac = 1e-2,
) where {T}
    x0   = MadNLP.variable(x)
    lvar = MadNLP.variable(xl)

    x0   .= MadNLP.get_x0(bcb)
    lvar .= MadNLP.get_lvar(bcb)
    y    .= MadNLP.get_y0(bcb)
    lcon = copy(MadNLP.get_lcon(bcb))

    # In std form all constraints are equalities and there are no fixed vars.
    MadNLP.set_initial_bounds!(lvar, tol)
    # Push x0 strictly above lvar.
    x0 .= ifelse.(x0 .< lvar .+ bound_push, lvar .+ bound_push, x0)

    x_full = MadNLP._update_x!(bcb, x0)
    copyto!(bx_buffer, x_full)
    MadNLP._eval_cons_wrapper!(bcb, bx_buffer, bcb.con_buffer)

    rhs .= lcon  # equalities: lcon == ucon
    return
end

function MadNLP.set_con_scale_sparse!(
    con_scale::MT,
    jac_I,
    jac_buffer,
    max_gradient,
) where {T, MT <: AbstractMatrix{T}}
    fill!(con_scale, one(T))
    MadNLP._set_con_scale_sparse!(con_scale, jac_I, jac_buffer)
    con_scale .= min.(one(T), max_gradient ./ con_scale)
    return con_scale
end

function MadNLP._set_con_scale_sparse!(con_scale::MT, jac_I, jac_buffer) where {T, MT <: AbstractMatrix{T}}
    nnzj = length(jac_I)
    bs = size(jac_buffer, 2)
    @inbounds for k in 1:nnzj
        row = jac_I[k]
        for j in 1:bs
            con_scale[row, j] = max(con_scale[row, j], abs(jac_buffer[k, j]))
        end
    end
    return con_scale
end

function MadNLP.set_jac_scale_sparse!(jac_scale::MT, con_scale, jac_I) where {T, MT <: AbstractMatrix{T}}
    return copyto!(jac_scale, @view(con_scale[jac_I, :]))
end

function MadNLP.set_obj_scale!(obj_scale, F::MT, max_gradient) where {T, MT <: AbstractMatrix{T}}
    return obj_scale .= min.(one(T), max_gradient ./ maximum(abs, F; dims = 1))
end

function MadNLP.set_scaling!(
    cb::UniformBatchCallback,
    x, xl, y, rhs, ind_ineq, nlp_scaling_max_gradient,
    bx_buffer,
)
    x0 = MadNLP.variable(x)
    x_full = MadNLP._update_x!(cb, x0)
    copyto!(bx_buffer, x_full)

    jac_free = MadNLP._eval_jac_wrapper!(cb, bx_buffer, cb.jac_buffer)
    MadNLP.set_con_scale_sparse!(cb.con_scale, cb.jac_I, jac_free, nlp_scaling_max_gradient)
    MadNLP.set_jac_scale_sparse!(cb.jac_scale, cb.con_scale, cb.jac_I)

    MadNLP._eval_grad_f_wrapper!(cb, bx_buffer, cb.grad_buffer)
    MadNLP.set_obj_scale!(cb.obj_scale, cb.grad_buffer, nlp_scaling_max_gradient)

    y ./= cb.con_scale
    rhs .*= cb.con_scale
    return
end
