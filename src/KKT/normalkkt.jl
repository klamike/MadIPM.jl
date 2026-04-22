# ============================================================================
# Normal-equations KKT system `A Σ⁻¹ Aᵀ Δy = r`.
#
# Only valid for LPs (asserted: `cb.nnzh == 0`). `(Δx, Δz)` are recovered by
# back-substitution after `Δy` is solved. Compared to `SparseKKTSystem`, much
# faster on tall-thin LPs (small `m × m` normal matrix); unusable on QPs and
# loses sparsity when `A` has dense rows.
# ============================================================================

"""
    NormalKKTSystem

- `aug_com`       — normal matrix `A Σ⁻¹ Aᵀ` (lower triangle, CSC).
- `A`             — Jacobian extended with slack columns `-I` on inequality
                     rows (COO).
- `AT`            — `A` transposed in CSC.
- `A_csr_map`     — permutation `A.V → AT.nzval` so value updates on `A`
                     propagate to `AT` without rebuilding the CSC structure.
"""
struct NormalKKTSystem{T, VT, MT, VI, VI32, LS} <:
        MadNLP.AbstractKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    aug_com::MT
    A::MadNLP.SparseMatrixCOO{T, Int32, VT, VI32}
    AT::MT
    A_csr_map::Union{Nothing, VI}
    jac::VT

    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    buffer_n::VT
    buffer_m::VT

    linear_solver::LS

    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    n::Int
    m::Int
end

function MadNLP.create_kkt_system(
    ::Type{NormalKKTSystem}, cb::MadNLP.SparseCallback{T, VT},
    linear_solver::Type;
    opt_linear_solver = MadNLP.default_options(linear_solver),
    hessian_approximation = MadNLP.ExactHessian,
    qn_options = MadNLP.QuasiNewtonOptions(),
) where {T, VT}
    cb.nnzh == 0 || error(
        "NormalKKTSystem supports linear programs only; the problem has " *
        "$(cb.nnzh) nonzeros in its Hessian.")

    n        = cb.nvar
    m        = cb.ncon
    ind_ineq = cb.ind_ineq
    ns       = length(ind_ineq)
    nlb, nub = length(cb.ind_lb), length(cb.ind_ub)
    ntot     = n + ns

    # ---------- diagonals and scratch ----------
    reg      = VT(undef, ntot)
    pr_diag  = VT(undef, ntot)
    du_diag  = VT(undef, m)
    l_diag   = VT(undef, nlb);  u_diag  = VT(undef, nub)
    l_lower  = VT(undef, nlb);  u_lower = VT(undef, nub)
    buffer_n = VT(undef, ntot); buffer_m = VT(undef, m)

    # ---------- Jacobian with inequality-slack columns ----------
    jac_I = MadNLP.create_array(cb, Int32, cb.nnzj)
    jac_J = MadNLP.create_array(cb, Int32, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb, jac_I, jac_J)
    nnzj = length(jac_I)

    I = MadNLP.create_array(cb, Int32, nnzj + ns)
    J = MadNLP.create_array(cb, Int32, nnzj + ns)
    V = VT(undef, nnzj + ns)
    I[1:nnzj]         .= jac_I
    J[1:nnzj]         .= jac_J
    I[nnzj+1:end]     .= ind_ineq
    J[nnzj+1:end]     .= (n+1):(n+ns)
    A_coo = MadNLP.SparseMatrixCOO(m, ntot, I, J, V)
    jac   = MadNLP._madnlp_unsafe_wrap(V, nnzj, 1)

    # tag values by their original index to recover the COO→CSC permutation
    A_coo.V .= 1:(nnzj + ns)
    Ap, Aj, Ax = coo_to_csr(A_coo)
    A_csr_map  = convert.(Int, Ax)

    # ---------- AT (CSC) and normal system (CSC, lower tri) ----------
    CSC = sparse_csc_format(VT)
    AT  = CSC <: SparseArrays.SparseMatrixCSC ?
        CSC(ntot, m, Ap, Aj, Ax) : CSC(Ap, Aj, Ax, (ntot, m))

    AAp, AAj = if CSC <: SparseArrays.SparseMatrixCSC
        build_normal_system(m, ntot, Ap, Aj)
    else
        AAp_h, AAj_h = build_normal_system(m, ntot, Vector(Ap), Vector(Aj))
        VII = typeof(Ap)
        VII(AAp_h), VII(AAj_h)
    end
    AAx = VT(undef, length(AAj))

    aug_com = CSC <: SparseArrays.SparseMatrixCSC ?
        CSC(m, m, AAp, AAj, AAx) : CSC(AAp, AAj, AAx, (m, m))

    ls = linear_solver(aug_com; opt = opt_linear_solver)
    fill!(jac, zero(T))

    return NormalKKTSystem(
        aug_com, A_coo, AT, A_csr_map, jac,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        buffer_n, buffer_m, ls,
        ind_ineq, cb.ind_lb, cb.ind_ub, ntot, m,
    )
end

# ---------- MadNLP interface ----------

MadNLP.num_variables(kkt::NormalKKTSystem)  = length(kkt.pr_diag)
MadNLP.get_jacobian(kkt::NormalKKTSystem)   = kkt.jac
MadNLP.get_hessian(::NormalKKTSystem{T, VT}) where {T, VT} = VT(undef, 0)

MadNLP.is_inertia_correct(kkt::NormalKKTSystem, npos, nzero, _nneg) =
    nzero == 0 && npos == kkt.m

function MadNLP.initialize!(kkt::NormalKKTSystem{T}) where {T}
    fill!(kkt.reg,      one(T))
    fill!(kkt.pr_diag,  one(T))
    fill!(kkt.du_diag,  zero(T))
    fill!(kkt.l_lower,  zero(T))
    fill!(kkt.u_lower,  zero(T))
    fill!(kkt.l_diag,   one(T))
    fill!(kkt.u_diag,   one(T))
    fill!(kkt.buffer_m, zero(T))
    fill!(kkt.buffer_n, zero(T))
    return nothing
end

function MadNLP.compress_jacobian!(kkt::NormalKKTSystem)
    ns = length(kkt.ind_ineq)
    kkt.A.V[end-ns+1:end] .= -1.0
    @inbounds for i in eachindex(kkt.A_csr_map)
        kkt.AT.nzval[i] = kkt.A.V[kkt.A_csr_map[i]]
    end
    return nothing
end

MadNLP.compress_hessian!(::NormalKKTSystem) = nothing

MadNLP.jtprod!(y::AbstractVector, kkt::NormalKKTSystem, x::AbstractVector) =
    mul!(y, kkt.AT, x)

function MadNLP.build_kkt!(kkt::NormalKKTSystem{T}) where {T}
    D  = kkt.buffer_n
    Cp = _colptr(kkt.aug_com); Cj = _rowval(kkt.aug_com); Cx = _nzval(kkt.aug_com)
    Ap = _colptr(kkt.AT);      Aj = _rowval(kkt.AT);      Ax = _nzval(kkt.AT)

    D .= one(T) ./ kkt.pr_diag
    assemble_normal_system!(kkt.m, kkt.n, Ap, Aj, Ax, Cp, Cj, Cx, D)
    return nothing
end

function MadNLP.solve_kkt!(kkt::NormalKKTSystem{T}, w::MadNLP.AbstractKKTVector) where {T}
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag,
                       w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)

    r1 = kkt.buffer_n
    r2 = kkt.buffer_m
    Σ  = kkt.pr_diag
    wx, wy = MadNLP.primal(w), MadNLP.dual(w)

    r1 .= wx ./ Σ                                # Σ⁻¹ r₁
    r2 .= wy
    mul!(r2, kkt.AT', r1, one(T), -one(T))       # A Σ⁻¹ r₁ − r₂
    MadNLP.solve_linear_system!(kkt.linear_solver, r2)   # Δy
    wy .= r2

    r1 .= wx
    mul!(r1, kkt.AT, wy, -one(T), one(T))        # r₁ − Aᵀ Δy
    wx .= r1 ./ Σ

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

function MadNLP.mul!(
    w::MadNLP.AbstractKKTVector{T}, kkt::NormalKKTSystem,
    v::MadNLP.AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T}
    mul!(MadNLP.primal(w), kkt.AT,  MadNLP.dual(v),   alpha, beta)
    mul!(MadNLP.dual(w),   kkt.AT', MadNLP.primal(v), alpha, beta)
    MadNLP._kktmul!(w, v, kkt.reg, kkt.du_diag,
                    kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag,
                    alpha, beta)
    return w
end
