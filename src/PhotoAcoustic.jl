module PhotoAcoustic


using JUDI, LinearAlgebra
using JUDI.DSP, JUDI.PyCall

import Base: getindex, *, copy!, copyto!, similar
import JUDI: judiMultiSourceVector, judiComposedPropagator, judiPropagator, make_input, propagate, zero
import LinearAlgebra: adjoint

const impl = PyNULL()
# Useful types
const RangeOrVec = Union{AbstractRange, Vector}

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"),dirname(pathof(PhotoAcoustic)))
    copy!(impl, pyimport("implementation"))
end

# Sources
include("judiInitialState.jl")
# Operators
include("judiPhoto.jl")

end # module

