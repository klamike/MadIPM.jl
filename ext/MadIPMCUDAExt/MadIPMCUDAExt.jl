module MadIPMCUDAExt

using LinearAlgebra
using SparseArrays
using NLPModels
using BatchQuadraticModels
using CUDA
using CUDA.CUSPARSE
using CUDSS
using KernelAbstractions
import Atomix
import LinearAlgebra: BlasFloat
import MadIPM
import MadNLP

include("cuda_wrapper.jl")
include("cuda_batch_kernels.jl")

function MadIPM._csc_with_nzval(A::CUSPARSE.CuSparseMatrixCSC, nzval, n)
    return CUSPARSE.CuSparseMatrixCSC(A.colPtr, A.rowVal, nzval, (n, n))
end

end
