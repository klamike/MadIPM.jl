
function _build_batch_kkt_element(
    cb::MadNLP.SparseCallback,
    ind_cons,
    I::AbstractVector{Int32},
    J::AbstractVector{Int32},
    V::VT,
    structure,
    aug_com_template::CUSPARSE.CuSparseMatrixCSC,
    aug_csc_map,
    csc_nzVal::VT
) where {VT}

    views = MadNLP._build_sparsekkt_views(VT, I, J, V, structure)

    aug_com = CUSPARSE.CuSparseMatrixCSC(
        aug_com_template.colPtr, aug_com_template.rowVal,
        csc_nzVal, aug_com_template.dims
    )
    jac_com, jac_csc_map = MadNLP.coo_to_csc(views.jac_raw)
    hess_com, hess_csc_map = MadNLP.coo_to_csc(views.hess_raw)

    quasi_newton = MadNLP.create_quasi_newton(MadNLP.ExactHessian, cb, structure.n)
    return MadNLP.SparseKKTSystem(
        views.hess, views.jac_callback, views.jac, quasi_newton,
        views.reg, views.pr_diag, views.du_diag,
        views.l_diag, views.u_diag, views.l_lower, views.u_lower,
        views.aug_raw, aug_com, aug_csc_map,
        views.hess_raw, hess_com, hess_csc_map,
        views.jac_raw, jac_com, jac_csc_map,
        nothing,
        structure.ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
    )
end

function MadIPM.SparseSameStructureBatchKKTSystem(
    batch_cb::MadIPM.SparseBatchCallback{T,VT},
    ind_cons,
    linear_solver::Type{<:MadNLPGPU.CUDSSSolver};
    opt_linear_solver=MadNLP.default_options(linear_solver),
) where {T,VT}
    cb1 = batch_cb.callbacks[1]

    structure = MadNLP._build_sparsekkt_structure(cb1, ind_cons, MadNLP.ExactHessian)
    (; aug_mat_length) = structure

    aug_I = MadNLP.create_array(cb1, Int32, aug_mat_length)
    aug_J = MadNLP.create_array(cb1, Int32, aug_mat_length)
    MadNLP.build_aug_indices!(aug_I, aug_J, structure)

    nzVals = VT(undef, aug_mat_length * batch_size)
    fill!(nzVals, zero(T))

    # FIXME some extra work here
    V_1 = MadNLP._madnlp_unsafe_wrap(nzVals, aug_mat_length, 1)
    views = _build_sparsekkt_views(VT, aug_I, aug_J, V_1, structure)
    aug_com, aug_csc_map = coo_to_csc(views.aug_raw)
    nnz_csc = length(aug_com.nzVal)

    csc_nzVals = VT(undef, nnz_csc * batch_size)
    fill!(csc_nzVals, zero(T))

    csc_nzVal_1 = MadNLP._madnlp_unsafe_wrap(csc_nzVals, nnz_csc, 1)
    kkt_1 = _build_batch_kkt_element(
        cb1, ind_cons, aug_I, aug_J, V_1, structure,
        aug_com, aug_csc_map, csc_nzVal_1;
    )

    kkts = Vector{typeof(kkt_1)}(undef, batch_size)
    kkts[1] = kkt_1

    for i in 2:batch_size
        cb_i = batch_cb.callbacks[i]
        val_offset = (i - 1) * aug_mat_length
        csc_offset = (i - 1) * nnz_csc

        V_i = MadNLP._madnlp_unsafe_wrap(nzVals, aug_mat_length, val_offset + 1)
        csc_nzVal_i = MadNLP._madnlp_unsafe_wrap(csc_nzVals, nnz_csc, csc_offset + 1)

        kkts[i] = _build_batch_kkt_element(
            cb_i, ind_cons, aug_I, aug_J, V_i, structure,
            aug_com, aug_csc_map, csc_nzVal_i;
        )
    end

    batch_solver = CUDSSUniformBatchSolver(
        aug_com, csc_nzVals,
        batch_size;
        opt=opt_linear_solver,
    )

    return MadIPM.SparseSameStructureBatchKKTSystem(
        nzVals, aug_I, aug_J,
        kkts, batch_solver,
        aug_mat_length, batch_size,
    )
end
