mutable struct CUDSSUniformBatchSolver{T,V} <: MadNLP.AbstractLinearSolver{T}
    inner::CUDSS.CudssSolver{T,Cint}
    rowPtr::CuVector{Cint}
    colVal::CuVector{Cint}
    nzVal::V
    coo_nzVal::V
    csr_map::CuVector{Int}
    x_gpu::CUDSS.CudssMatrix{T,Cint}
    b_gpu::CUDSS.CudssMatrix{T,Cint}
    buffer::V

    n::Int
    nnz::Int
    nbatch::Int

    opt::MadNLPGPU.CudssSolverOptions
    logger::MadNLP.MadNLPLogger
end

function CUDSSUniformBatchSolver(
    coo_I::CuVector{Int32},
    coo_J::CuVector{Int32},
    coo_nzVal::CuVector{T},
    n::Int,
    nbatch::Int;
    opt=MadNLPGPU.CudssSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
) where T
    nnz_per_batch = length(coo_I)
    @assert length(coo_nzVal) == nnz_per_batch * nbatch

    view = 'L'
    structure = "G"
    (opt.cudss_algorithm == MadNLP.LU) && error(logger, "The sparse LU of cuDSS is not supported for uniform batch.")
    (opt.cudss_algorithm == MadNLP.CHOLESKY) && (structure = "SPD")
    (opt.cudss_algorithm == MadNLP.LDL) && (structure = "S")

    # Convert COO indices to CSR format
    rowPtr, colVal, csr_map = _coo_to_csr_structure(coo_I, coo_J, n)

    # Build the strided nzVal for all batches using the CSR mapping
    nzVal = CuVector{T}(undef, nnz_per_batch * nbatch)
    _apply_csr_map!(nzVal, coo_nzVal, csr_map, nnz_per_batch, nbatch)

    # Create CUDSS solver with uniform batch
    solver = CUDSS.CudssSolver(rowPtr, colVal, nzVal, structure, view)
    CUDSS.cudss_set(solver, "ubatch_size", nbatch)
    MadNLPGPU.set_cudss_options!(solver, opt)

    if opt.cudss_ordering != MadNLPGPU.DEFAULT_ORDERING
        error(logger, "Custom ordering is not supported for uniform batch solver.")
    end

    # Create solution and RHS matrices for batch
    x_gpu = CUDSS.CudssMatrix(T, n; nbatch)
    b_gpu = CUDSS.CudssMatrix(T, n; nbatch)

    # Perform analysis phase
    CUDSS.cudss("analysis", solver, x_gpu, b_gpu; asynchronous=true)

    # Allocate buffer for iterative refinement
    buffer = CuVector{T}(undef, n * nbatch)

    return CUDSSUniformBatchSolver(
        solver,
        rowPtr, colVal, nzVal, coo_nzVal, csr_map,
        x_gpu, b_gpu, buffer,
        n, nnz_per_batch, nbatch,
        opt, logger,
    )
end

function _coo_to_csr_structure(I::CuVector{Int32}, J::CuVector{Int32}, n::Int)
    # Sort by (row, col) to get CSR order, keeping track of original indices
    nnz = length(I)
    zvals = CuVector{Int}(1:nnz)
    coord = map((i, j, k) -> ((i, j), k), I, J, zvals)
    if nnz > 0
        sort!(coord, lt = (((i, j), k), ((m, n), l)) -> (i, j) < (m, n))
    end

    # Build rowPtr using getptr to find row boundaries
    mapptr = MadNLP.getptr(coord; by = ((x1, x2), (y1, y2)) -> x1[1] != y1[1])

    rowPtr = CuVector{Cint}(undef, n + 1)
    fill!(rowPtr, Cint(nnz + 1))
    rowPtr[1] = Cint(1)

    if nnz > 0
        # Set row pointers from the sorted coordinates
        coord_first = coord[@view(mapptr[1:end-1])]
        for idx in 1:length(coord_first)
            row = coord_first[idx][1][1]
            rowPtr[row] = Cint(mapptr[idx])
        end
        # Forward fill rowPtr for empty rows
        for row in 2:n+1
            if rowPtr[row] > nnz
                rowPtr[row] = rowPtr[row-1]
            end
        end
        rowPtr[n+1] = Cint(nnz + 1)
    end

    # Extract column indices in CSR order
    colVal = CuVector{Cint}(map(x -> Cint(x[1][2]), coord))

    # Build the mapping from COO to CSR (csr_map[csr_idx] = coo_idx)
    csr_map = CuVector{Int}(map(x -> x[2], coord))

    return rowPtr, colVal, csr_map
end

function _apply_csr_map!(dest::CuVector{T}, src::CuVector{T}, csr_map::CuVector{Int}, nnz::Int, nbatch::Int) where T
    for i in 1:nbatch
        offset = (i - 1) * nnz
        src_view = @view src[offset+1:offset+nnz]
        dst_view = @view dest[offset+1:offset+nnz]
        copyto!(dst_view, src_view[csr_map])
    end
    return dest
end

function MadNLP.factorize!(M::CUDSSUniformBatchSolver)
    # Update nzVal from COO format using stored mapping
    _apply_csr_map!(M.nzVal, M.coo_nzVal, M.csr_map, M.nnz, M.nbatch)
    CUDSS.cudss_update(M.inner, M.rowPtr, M.colVal, M.nzVal)

    if M.inner.fresh_factorization
        CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu; asynchronous=true)
    else
        CUDSS.cudss("refactorization", M.inner, M.x_gpu, M.b_gpu; asynchronous=true)
    end
    if !M.opt.cudss_hybrid_memory && !M.opt.cudss_hybrid_execute
        CUDA.synchronize()
    end
    return M
end

function MadNLP.solve!(M::CUDSSUniformBatchSolver{T,V}, xb::V) where {T,V}
    if M.opt.cudss_ir > 0
        copyto!(M.buffer, xb)
        CUDSS.cudss_update(M.b_gpu, M.buffer)
    else
        CUDSS.cudss_update(M.b_gpu, xb)
    end
    CUDSS.cudss_update(M.x_gpu, xb)
    CUDSS.cudss("solve", M.inner, M.x_gpu, M.b_gpu; asynchronous=true)
    if !M.opt.cudss_hybrid_memory && !M.opt.cudss_hybrid_execute
        CUDA.synchronize()
    end
    return xb
end

MadNLP.input_type(::Type{CUDSSUniformBatchSolver}) = :coo
MadNLP.default_options(::Type{CUDSSUniformBatchSolver}) = MadNLPGPU.CudssSolverOptions()
MadNLP.is_inertia(M::CUDSSUniformBatchSolver) = true

function MadNLP.inertia(M::CUDSSUniformBatchSolver)
    n = M.n
    info = CUDSS.cudss_get(M.inner, "info")

    if M.opt.cudss_algorithm == MadNLP.CHOLESKY
        if info == 0
            return (n, 0, 0)
        else
            return (n-2, 1, 1)
        end
    elseif M.opt.cudss_algorithm == MadNLP.LDL
        if info == 0
            (k, l) = CUDSS.cudss_get(M.inner, "inertia")
            @assert 0 ≤ k + l ≤ n
            return (k, n - k - l, l)
        else
            return (0, 1, n)
        end
    else
        error(M.logger, "Unsupported cudss_algorithm")
    end
end

MadNLP.improve!(M::CUDSSUniformBatchSolver) = false
MadNLP.is_supported(::Type{CUDSSUniformBatchSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{CUDSSUniformBatchSolver}, ::Type{Float64}) = true
MadNLP.introduce(M::CUDSSUniformBatchSolver) = "cuDSS Uniform Batch v$(CUDSS.version()) (nbatch=$(M.nbatch))"

function MadIPM.SparseSameStructureBatchKKTSystem(
    ::Type{KKT},
    batch_cb::MadIPM.SparseBatchCallback,
    ind_cons,
    linear_solver::Type{<:MadNLPGPU.CUDSSSolver};
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

    # Create the batch solver for uniform CUDSS solving
    batch_solver = CUDSSUniformBatchSolver(
        aug_I, aug_J, nzVals,
        aug_vec_length, batch_size;
        opt=opt_linear_solver,
    )

    return MadIPM.SparseSameStructureBatchKKTSystem(
        nzVals,
        aug_I, aug_J,
        kkts,
        batch_solver,
        nzVal_length, batch_size,
    )
end
