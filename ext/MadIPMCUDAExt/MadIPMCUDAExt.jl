module MadIPMCUDAExt

using Adapt
using Atomix: Atomix
using BatchQuadraticModels
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
using SparseArrays
import NLPModels
import MadIPM
import MadNLP

include("cuda_wrapper.jl")

end
