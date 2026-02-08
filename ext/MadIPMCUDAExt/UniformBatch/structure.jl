abstract type AbstractBatchSolver end

struct UniformBatchSolver{VS,BK,BQ,SD} <: AbstractBatchSolver
    solvers::VS
    bkkt::BK
    bqp::BQ
    step::SD

    function UniformBatchSolver(models::Vector{Model}; linear_solver::Type, kwargs...) where {Model<:NLPModels.AbstractNLPModel}
        batch_size = length(models)
        
        is_rhs_batch = true
        qp1 = first(models)
        for qp in models[2:end]
            if !QuadraticModels._check_only_rhs_differs(qp1, qp)
                is_rhs_batch = false
                break
            end
        end

        bqp = if is_rhs_batch
            @info "Using RHSBatchQuadraticModel to accelerate `evaluate_model!`"
            QuadraticModels.RHSBatchQuadraticModel(models)
        else
            models
        end

        solvers = Vector{MadIPM.MPCSolver}(undef, batch_size)
        for i in 1:batch_size
            solvers[i] = MadIPM.MPCSolver(models[i]; linear_solver=NoLinearSolver, kwargs...)
        end

        solver1 = first(solvers)
        nlp1 = solver1.nlp
        kkt1 = solver1.kkt
        vec1 = solver1.d

        kkts = Vector{typeof(kkt1)}(undef, batch_size)
        vecs = Vector{typeof(vec1)}(undef, batch_size)
        for i in 1:batch_size
            solver_i = solvers[i]
            kkts[i] = solver_i.kkt
            vecs[i] = solver_i.d
        end

        options = MadIPM.load_options(nlp1; linear_solver=linear_solver, kwargs...)
        bkkt = UniformBatchKKTSystem(kkts, vecs, linear_solver, opt_linear_solver=options.linear_solver)

        # TODO: profile to see if this is worth it
        CUDA.enable_synchronization!(bkkt.batch_nzVal, false)
        CUDA.enable_synchronization!(bkkt.batch_rhs, false)
        step = BatchStepData(solver1, batch_size)
        return new{typeof(solvers),typeof(bkkt),typeof(bqp),typeof(step)}(solvers, bkkt, bqp, step)
    end
end

all_done(batch_solver::UniformBatchSolver) = all_done(batch_solver.bkkt)
is_active(batch_solver::UniformBatchSolver, i) = is_active(batch_solver.bkkt, i)
n_active(batch_solver::UniformBatchSolver) = sum(batch_solver.bkkt.is_active)
Base.length(batch_solver::UniformBatchSolver) = length(batch_solver.solvers)
Base.iterate(batch_solver::UniformBatchSolver, i=1) = iterate(batch_solver.solvers, i)
Base.getindex(batch_solver::UniformBatchSolver, i) = batch_solver.solvers[i]

NVTX.@annotate function update_batch!(batch_solver::UniformBatchSolver)
    needs_update = false
    for (i, solver) in enumerate(batch_solver)
        if is_active(batch_solver, i) && MadIPM.is_done(solver)
            needs_update = true
            batch_solver.bkkt.is_active[i] = false
        end
    end
    needs_update && update_batch!(batch_solver.bkkt)
    return
end


Base.eltype(batch_solver::UniformBatchSolver) = eltype(batch_solver.step.x_lr)