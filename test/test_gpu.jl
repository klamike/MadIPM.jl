using Adapt
using KernelAbstractions
using MadNLPGPU
using CUDA.CUSPARSE

@testset "MadIPMCUDA" begin
    qp = simple_lp()
    qp_gpu = adapt(CuArray, qp)

    for (kkt, algo) in ((MadNLP.ScaledSparseKKTSystem, MadNLP.LDL     ),
                        (MadNLP.SparseKKTSystem      , MadNLP.LDL     ),
                        (MadIPM.NormalKKTSystem      , MadNLP.CHOLESKY))
        solver = MadIPM.MPCSolver(
            qp_gpu;
            kkt_system=kkt,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=algo,
            print_level=MadNLP.ERROR,
        )
        results = MadIPM.solve!(solver)
        @test results.status == MadNLP.SOLVE_SUCCEEDED
    end
end
