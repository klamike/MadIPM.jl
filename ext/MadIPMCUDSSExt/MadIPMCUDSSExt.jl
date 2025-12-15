module MadIPMCUDSSExt

using CUDA
using CUDA.CUSPARSE
using CUDSS

import MadIPM
import MadNLP
import MadNLPGPU

include("cudss_uniform.jl")

end  # module