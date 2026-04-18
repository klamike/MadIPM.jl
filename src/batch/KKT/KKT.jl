abstract type AbstractBatchKKTSystem{T} end

# LB-only KKT multiplication for std-form batch solver.
@inbounds function _kktmul!(
    w::BatchUnreducedKKTVector,
    x::BatchUnreducedKKTVector,
    reg, du_diag_val, l_lower, l_diag,
    alpha, beta,
)
    MadNLP.primal(w) .+= alpha .* reg .* MadNLP.primal(x)
    MadNLP.dual(w)   .+= alpha .* du_diag_val .* MadNLP.dual(x)
    xp_lr(w)         .-= alpha .* MadNLP.dual_lb(x)
    MadNLP.dual_lb(w) .= beta .* MadNLP.dual_lb(w) .+ alpha .* (xp_lr(x) .* l_lower .- MadNLP.dual_lb(x) .* l_diag)
    return
end

include("Sparse/augmented.jl")
