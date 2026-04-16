# MadIPM.jl

MadIPM.jl is a GPU-accelerated optimization solver for linear and quadratic programming.
The solver implements the Mehrotra predictor-corrector method in pure Julia,
and supports the solution of large-scale linear programs on the GPU using NVIDIA cuDSS.

## Installation

MadIPM can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add MadIPM
pkg> test MadIPM
```

## Basic usage

### JuMP

MadIPM supports JuMP models with an extension for `MathOptInterface.jl`.
For instance, you can solve any LP formulated with JuMP by using:

```julia
using JuMP
using MadIPM

c = rand(10)
model = Model(MadIPM.Optimizer)
@variable(model, 0 <= x[1:10], start=0.5)
@constraint(model, sum(x) == 1.0)
@objective(model, Min, c' * x)
JuMP.optimize!(model)
```

### Native quadratic models

MadIPM ships with a lightweight `QuadraticModel` constructor for LPs and QPs.

```julia
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
results = madipm(qp)
```

### Custom usage

MadIPM takes as input any linear program (LP) or quadratic program (QP) represented as an `AbstractNLPModel`,
following the specification in [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/).

For any `qp <: AbstractNLPModel`, you can pass it to MadIPM either directly with `madipm(qp)`, or in two steps as follows:

```julia
solver = MPCSolver(qp)
results = MadIPM.solve!(solver)
```

## Solving a LP with CUDA

MadIPM supports GPU acceleration using NVIDIA cuDSS.
It requires specifying your problem in a `QuadraticProblem` first.

The data are moved to the GPU using:
```julia
using CUDA, KernelAbstractions, MadNLPGPU
using CUDA.CUSPARSE
using Adapt
using MadIPM

qp_gpu = adapt(CuArray, qp)
```
Then, you can pass the problem `qp_gpu` to MadIPM by switching
the linear solver to NVIDIA cuDSS:
```julia
solver = MPCSolver(qp_gpu; linear_solver=MadNLPGPU.CUDSSSolver)
results = MadIPM.solve!(solver)
```
As a result, all the solution happens on the GPU, with minimum data transfer
between the host and the device.

If you have a JuMP model, build a `QuadraticModel` in user code and move its data to CUDA before calling `MPCSolver`:
```julia
using MadIPM
using CUDA, KernelAbstractions, MadNLPGPU
using CUDA.CUSPARSE
```

## Citing MadIPM.jl

If you use MadIPM.jl in your research, we would greatly appreciate your citing it.

```bibtex
@article{MadIPM,
  title   = {{GPU Implementation of Second-Order Linear and Nonlinear Programming Solvers}},
  author  = {Montoison, Alexis and Pacaud, Fran{\c{c}}ois and Shin, Sungho and Anitescu, Mihai},
  journal = {arXiv preprint arXiv:2508.16094},
  year    = {2025}
}
```
