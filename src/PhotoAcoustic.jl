module PhotoAcoustic


using LinearAlgebra, Reexport
@reexport using JUDI

PhotoAcoustic_path = dirname(pathof(PhotoAcoustic))

using JUDI.DSP, JUDI.PyCall, JUDI.FFTW, JUDI.JOLI
using FourierTools

import Base: getindex, *, copy!, copyto!, similar, getproperty, display
import JUDI: judiMultiSourceVector, judiComposedPropagator, judiPropagator, judiNoopOperator
import JUDI: judiDataModeling, judiModeling, jAdjoint, Projection, judiVector, Geometry
import JUDI: RangeOrVec, make_input, propagate, zero, process_input_data, setup_grid
import JUDI: wrapcall_data, wrapcall_function, compute_illum, wrapcall_weights
import JUDI: time_resample, make_src, get_nsrc, filter_none
import LinearAlgebra: adjoint

const impl = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"),PhotoAcoustic_path)
    copy!(impl, pyimport("implementation"))
end

# utility for data loading 

PhotoAcoustic_data = joinpath(PhotoAcoustic_path, "../data")

# Utilities
include("utils.jl")
# Sources
include("judiInitialState.jl")
# Operators
include("judiPhoto.jl")
# Transducer
include("transducer.jl")

end # module

