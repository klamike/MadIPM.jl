using BatchQuadraticModels
import BatchQuadraticModels: LPData, QPData, LinearModel, QuadraticModel,
    operator_sparse_matrix, ruiz_equilibration!
using QPSReader
using GZip
using CodecBzip2
using NLPModels
using SparseArrays
using SparseMatricesCOO: SparseMatrixCOO

"""
    import_mps(filename::String)

Import instance from the file whose path is specified in `filename`.

The function parses the file's extension to adapt the import. If the extension
is `.mps`, `.sif` or `.SIF`, it directly reads the file. If the extension
is `.gz` or `.bz2`, it decompresses the file using gzip or bzip2, respectively.
"""
function import_mps(filename)
    ext = match(r"(.*)\.(.*)", filename).captures[2]
    data = if ext ∈ ("mps", "sif", "SIF")
        readqps(filename)
    elseif ext == "gz"
        GZip.open(filename, "r") do gz
            readqps(gz)
        end
    elseif ext == "bz2"
        open(filename, "r") do io
            stream = Bzip2DecompressorStream(io)
            readqps(stream)
        end
    end
    return data
end

"""
    qm_from_qpsdata(qps::QPSData)

Build a `BatchQuadraticModels.QuadraticModel` from a parsed `QPSData`.
"""
function qm_from_qpsdata(qps::QPSData)
    nvar = length(qps.lvar); ncon = length(qps.lcon)
    A = SparseMatrixCOO(ncon, nvar, qps.arows, qps.acols, qps.avals)
    H = SparseMatrixCOO(nvar, nvar, qps.qrows, qps.qcols, qps.qvals)
    data = QPData(A, qps.c, H;
        lvar = qps.lvar, uvar = qps.uvar,
        lcon = qps.lcon, ucon = qps.ucon,
        c0 = qps.c0)
    # QPSReader returns `:notset` when the MPS file omits an OBJSENSE section;
    # treat that as minimization (the LP convention) — only flip on explicit `:max`.
    return QuadraticModel(data; minimize = (qps.objsense != :max), name = qps.name)
end

"""
    scale_qp(qp; eps = 1e-3)

Ruiz-equilibrate the constraint matrix of `qp` using BQM's `ruiz_equilibration`,
then apply the resulting `(Dr, Dc)` to the model via `scale_model`.
"""
function scale_qp(qp::QuadraticModel; eps = 1e-3)
    _, scaling = ruiz_equilibration(operator_sparse_matrix(qp.data.A); eps)
    return scale_model(qp, scaling.row, scaling.col)
end
