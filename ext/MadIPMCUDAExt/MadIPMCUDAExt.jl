module MadIPMCUDAExt

# ============================================================================
# MadIPM × CUDA.
#
# Activated with CUDA + KernelAbstractions + MadNLPGPU loaded. Overrides:
#
#   * COO → CSC transfer path used by `compress_hessian!` / `compress_jacobian!`
#   * CPU sparse builders (`coo_to_csr`, `build_normal_system`,
#     `assemble_normal_system!`) with CUSPARSE / KA equivalents
#   * Batch-KKT `NormalUniformBatchKKTSystem.build_kkt!` (per-instance
#     normal-matrix assembly)
#   * Scalar-indexing kernels in `kernels/step.jl` and
#     `kernels/complementarity.jl` with KernelAbstractions variants
#
# See `cuda_wrapper.jl` for the method definitions.
# ============================================================================

using Adapt
using BatchQuadraticModels
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
using SparseArrays

using Atomix: Atomix

import MadIPM
import MadNLP
import NLPModels

include("cuda_wrapper.jl")

end # module
