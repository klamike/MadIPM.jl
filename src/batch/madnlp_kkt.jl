abstract type AbstractBatchKKTSystem{T, KKT<:MadNLP.AbstractKKTSystem} end

struct SparseSameStructureBatchKKTSystem{
    T,
    VT <: AbstractVector{T},
    VI32 <: AbstractVector{Int32},
    KKT <: MadNLP.SparseKKTSystem,
    LS <: MadNLP.AbstractLinearSolver{T},
} <: AbstractBatchKKTSystem{T, KKT}
    nzVals::VT

    aug_I::VI32
    aug_J::VI32

    kkts::Vector{KKT}

    batch_solver::LS

    nzVal_length::Int
    batch_size::Int

    function SparseSameStructureBatchKKTSystem(
        nzVals::VT,
        aug_I::VI32,
        aug_J::VI32,
        kkts::Vector{KKT},
        batch_solver::LS,
        nzVal_length::Int,
        batch_size::Int,
    ) where {T, VT<:AbstractVector{T}, VI32<:AbstractVector{Int32}, KKT<:MadNLP.SparseKKTSystem, LS<:MadNLP.AbstractLinearSolver{T}}
        new{T, VT, VI32, KKT, LS}(nzVals, aug_I, aug_J, kkts, batch_solver, nzVal_length, batch_size)
    end
end

# Default constructor that uses individual linear solvers (no batch solver)
function SparseSameStructureBatchKKTSystem(
    ::Type{KKT},
    batch_cb::SparseBatchCallback,
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
) where {KKT <: MadNLP.SparseKKTSystem}

    batch_size = batch_cb.batch_size
    batch_size == 0 && error("SparseSameStructureBatchKKTSystem requires at least one callback")

    cb1 = batch_cb.callbacks[1]
    T = eltype(cb1.con_buffer)
    VT = typeof(cb1.con_buffer)

    n_slack = length(ind_cons.ind_ineq)
    n = cb1.nvar
    m = cb1.ncon

    jac_sparsity_I = MadNLP.create_array(cb1, Int32, cb1.nnzj)
    jac_sparsity_J = MadNLP.create_array(cb1, Int32, cb1.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb1, jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I, hess_sparsity_J = MadNLP.build_hessian_structure(cb1, MadNLP.ExactHessian)
    MadNLP.force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)
    ind_ineq = ind_cons.ind_ineq

    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    aug_vec_length = n_tot + m
    aug_mat_length = n_tot + m + n_hess + n_jac + n_slack

    aug_I = MadNLP.create_array(cb1, Int32, aug_mat_length)
    aug_J = MadNLP.create_array(cb1, Int32, aug_mat_length)

    offset = n_tot + n_jac + n_slack + n_hess + m

    aug_I[1:n_tot] .= 1:n_tot
    aug_I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    aug_I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I .+ n_tot)
    aug_I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    aug_I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    aug_J[1:n_tot] .= 1:n_tot
    aug_J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    aug_J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    aug_J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    aug_J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    jac_I = Int32[jac_sparsity_I; ind_ineq]
    jac_J = Int32[jac_sparsity_J; n+1:n+n_slack]

    nzVal_length = aug_mat_length
    nzVals = VT(undef, nzVal_length * batch_size)
    fill!(nzVals, zero(T))

    cb_1 = batch_cb.callbacks[1]
    V_1 = MadNLP._madnlp_unsafe_wrap(nzVals, nzVal_length, 1)

    pr_diag_1 = MadNLP._madnlp_unsafe_wrap(V_1, n_tot)
    du_diag_1 = MadNLP._madnlp_unsafe_wrap(V_1, m, n_jac + n_slack + n_hess + n_tot + 1)

    reg_1 = VT(undef, n_tot)
    l_diag_1 = VT(undef, nlb)
    u_diag_1 = VT(undef, nub)
    l_lower_1 = VT(undef, nlb)
    u_lower_1 = VT(undef, nub)

    hess_1 = MadNLP._madnlp_unsafe_wrap(V_1, n_hess, n_tot + 1)
    jac_1 = MadNLP._madnlp_unsafe_wrap(V_1, n_jac + n_slack, n_hess + n_tot + 1)
    jac_callback_1 = MadNLP._madnlp_unsafe_wrap(V_1, n_jac, n_hess + n_tot + 1)

    aug_raw_1 = MadNLP.SparseMatrixCOO(aug_vec_length, aug_vec_length, aug_I, aug_J, V_1)
    jac_raw_1 = MadNLP.SparseMatrixCOO(m, n_tot, jac_I, jac_J, jac_1)
    hess_raw_1 = MadNLP.SparseMatrixCOO(n_tot, n_tot, hess_sparsity_I, hess_sparsity_J, hess_1)

    aug_com_1, aug_csc_map_1 = MadNLP.coo_to_csc(aug_raw_1)
    jac_com_1, jac_csc_map_1 = MadNLP.coo_to_csc(jac_raw_1)
    hess_com_1, hess_csc_map_1 = MadNLP.coo_to_csc(hess_raw_1)

    quasi_newton_1 = MadNLP.create_quasi_newton(MadNLP.ExactHessian, cb_1, n)
    linear_solver_1 = linear_solver(aug_com_1; opt=opt_linear_solver)

    kkt_1 = MadNLP.SparseKKTSystem(
        hess_1, jac_callback_1, jac_1, quasi_newton_1, reg_1, pr_diag_1, du_diag_1,
        l_diag_1, u_diag_1, l_lower_1, u_lower_1,
        aug_raw_1, aug_com_1, aug_csc_map_1,
        hess_raw_1, hess_com_1, hess_csc_map_1,
        jac_raw_1, jac_com_1, jac_csc_map_1,
        linear_solver_1,
        ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
    )

    KKTType = typeof(kkt_1)
    kkts = Vector{KKTType}(undef, batch_size)
    kkts[1] = kkt_1

    for i in 2:batch_size
        cb_i = batch_cb.callbacks[i]
        val_offset = (i - 1) * nzVal_length

        V_i = MadNLP._madnlp_unsafe_wrap(nzVals, nzVal_length, val_offset + 1)

        pr_diag_i = MadNLP._madnlp_unsafe_wrap(V_i, n_tot)
        du_diag_i = MadNLP._madnlp_unsafe_wrap(V_i, m, n_jac + n_slack + n_hess + n_tot + 1)

        reg_i = VT(undef, n_tot)
        l_diag_i = VT(undef, nlb)
        u_diag_i = VT(undef, nub)
        l_lower_i = VT(undef, nlb)
        u_lower_i = VT(undef, nub)

        hess_i = MadNLP._madnlp_unsafe_wrap(V_i, n_hess, n_tot + 1)
        jac_i = MadNLP._madnlp_unsafe_wrap(V_i, n_jac + n_slack, n_hess + n_tot + 1)
        jac_callback_i = MadNLP._madnlp_unsafe_wrap(V_i, n_jac, n_hess + n_tot + 1)

        aug_raw_i = MadNLP.SparseMatrixCOO(aug_vec_length, aug_vec_length, aug_I, aug_J, V_i)
        jac_raw_i = MadNLP.SparseMatrixCOO(m, n_tot, jac_I, jac_J, jac_i)
        hess_raw_i = MadNLP.SparseMatrixCOO(n_tot, n_tot, hess_sparsity_I, hess_sparsity_J, hess_i)

        aug_com_i, aug_csc_map_i = MadNLP.coo_to_csc(aug_raw_i)
        jac_com_i, jac_csc_map_i = MadNLP.coo_to_csc(jac_raw_i)
        hess_com_i, hess_csc_map_i = MadNLP.coo_to_csc(hess_raw_i)

        quasi_newton_i = MadNLP.create_quasi_newton(MadNLP.ExactHessian, cb_i, n)

        linear_solver_i = linear_solver(aug_com_i; opt=opt_linear_solver)

        kkts[i] = MadNLP.SparseKKTSystem(
            hess_i, jac_callback_i, jac_i, quasi_newton_i, reg_i, pr_diag_i, du_diag_i,
            l_diag_i, u_diag_i, l_lower_i, u_lower_i,
            aug_raw_i, aug_com_i, aug_csc_map_i,
            hess_raw_i, hess_com_i, hess_csc_map_i,
            jac_raw_i, jac_com_i, jac_csc_map_i,
            linear_solver_i,
            ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
        )
    end

    batch_solver = linear_solver_1   # FIXME: make this a dummy that errors if tried to be used

    return SparseSameStructureBatchKKTSystem(
        nzVals,
        aug_I, aug_J,
        kkts,
        batch_solver,
        nzVal_length, batch_size,
    )
end

Base.length(batch_kkt::SparseSameStructureBatchKKTSystem) = batch_kkt.batch_size
Base.iterate(batch_kkt::SparseSameStructureBatchKKTSystem, i=1) = i > length(batch_kkt) ? nothing : (batch_kkt.kkts[i], i+1)
Base.getindex(batch_kkt::SparseSameStructureBatchKKTSystem, i::Int) = batch_kkt.kkts[i]
