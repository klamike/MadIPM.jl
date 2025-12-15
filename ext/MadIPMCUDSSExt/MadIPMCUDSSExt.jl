module MadIPMCUDSSExt

using CUDA
using CUDA.CUSPARSE
using CUDSS

import MadIPM
import MadNLP
import MadNLPGPU

include("solver.jl")
include("kkt.jl")

end  # module