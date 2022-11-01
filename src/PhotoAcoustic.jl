module PhotoAcoustic


using LinearAlgebra, Reexport
@reexport using JUDI

# path for data loading 
PhotoAcoustic_path = dirname(pathof(PhotoAcoustic))
PhotoAcoustic_data = joinpath(PhotoAcoustic_path, "../data")

using JUDI.DSP, JUDI.PyCall

import Base: getindex, *, copy!, copyto!, similar
import JUDI: judiMultiSourceVector, judiComposedPropagator, judiPropagator, judiNoopOperator, jAdjoint, Projection
import JUDI: RangeOrVec, make_input, propagate, zero, process_input_data, wrapcall_data
import LinearAlgebra: adjoint

const impl = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"),PhotoAcoustic_path)
    copy!(impl, pyimport("implementation"))
end

# Sources
include("judiInitialState.jl")
# Operators
include("judiPhoto.jl")
# Utilities
include("utils/utilities.jl")

end # module

