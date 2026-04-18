# Predictor (affine) and Mehrotra correction direction kernels.
# Unified across scalar `MPCSolver` and batched `AbstractBatchMPCSolver`
# via dispatch on `AnyMPCSolver`.

function affine_direction!(s::AnyMPCSolver)
    set_predictive_rhs!(s, _kkt(s))
    solve_system!(_d(s), s, _p(s))
    return
end

function mehrotra_correction_direction!(s::AnyMPCSolver)
    set_correction_rhs!(s, _kkt(s), _mu(s), _correction_lb(s))
    solve_system!(_d(s), s, _p(s))
    return
end
