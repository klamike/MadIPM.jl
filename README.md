# MadIPM.jl

`MadIPM.jl` is a Mehrotra predictor-corrector interior-point solver for LPs and QPs, with CPU and NVIDIA GPU linear-solver paths.

It accepts:

- `BatchQuadraticModels.LinearModel` and `BatchQuadraticModels.QuadraticModel`
- MOI / JuMP models through the `MathOptInterface` extension

## Installation

```julia
julia> ]
pkg> add MadIPM
```

## Native model usage

```julia
using SparseArrays
using MadIPM

data = QPData(
    sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2),
    [1.0, 1.0],
    sparse(Int[], Int[], Float64[], 2, 2);
    lcon = [1.0],
    ucon = [1.0],
    lvar = [0.0, 0.0],
)

qp = QuadraticModel(data)
stats = madipm(qp)
```

## JuMP / MOI usage

```julia
using JuMP
using MadIPM

model = Model(MadIPM.Optimizer)
@variable(model, x[1:10] >= 0, start = 0.5)
@constraint(model, sum(x) == 1.0)
@objective(model, Min, sum(x))
optimize!(model)
```

## CUDA usage

Move a native model to CUDA in user code, then build the solver on the GPU-backed model:

```julia
using Adapt
using CUDA
using MadIPM
using MadNLPGPU

qp_gpu = adapt(CuArray, qp)
solver = MPCSolver(qp_gpu; linear_solver = MadNLPGPU.CUDSSSolver)
stats = solve!(solver)
```

## Citation

```bibtex
@article{MadIPM,
  title   = {{GPU Implementation of Second-Order Linear and Nonlinear Programming Solvers}},
  author  = {Montoison, Alexis and Pacaud, Fran{\c{c}}ois and Shin, Sungho and Anitescu, Mihai},
  journal = {arXiv preprint arXiv:2508.16094},
  year    = {2025}
}
```
