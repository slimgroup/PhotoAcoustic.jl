module PhotoAcoustic

	# Useful types
	const RangeOrVec = Union{AbstractRange, Vector}

    using JUDI
    using JUDI.DSP, JUDI.PyCall

    import Base: getindex, *
    import Base.copy!, Base.copyto!, Base.similar, JUDI.zero
    import JUDI: judiMultiSourceVector, judiComposedPropagator, judiPropagator, make_input, propagate
    import JUDI.LinearAlgebra: adjoint

    const impl = PyNULL()

    function __init__()
        pushfirst!(PyVector(pyimport("sys")."path"),dirname(pathof(PhotoAcoustic)))
        copy!(impl, pyimport("implementation"))
    end

	# Sources
	include("judiPhotoSource.jl")

	# Operators
	include("judiPhoto.jl")

end # module

