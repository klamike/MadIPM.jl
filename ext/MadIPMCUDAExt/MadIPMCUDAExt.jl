module MadIPMCUDAExt

using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
import MadIPM

include("cuda_wrapper.jl")

end
