export judiPhoto, judiInitialStateProjection

struct judiInitialStateProjection{D} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
end

"""
    judiInitialStateProjection(model)

Construct the projection operator that sets the initial state into the wavefield for propagation.
This operator is a No-op operation that will propagate a [`judiInitialState`](@ref) if combined with a JUDI
propagator.
"""
judiInitialStateProjection(model) = judiInitialStateProjection{eltype(model.m)}(space(model.n), time_space(model.n))

struct judiPhoto{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    F::judiModeling
    rInterpolation::Projection{D}
    Init::jAdjoint{<:judiInitialStateProjection{D}}
end

"""
    judiPhoto(F::judiPropagator, geometry::Geometry;)

Constructs a photoacoustic linear operator solving the wave equation associated with F.model. The parametrizations
currently supported through JUDI are isotropic acoustic (with or without density), acoustic anisotropic (TTI/VTI) and 
visco-acoustic.

Arguments
============
`F`: The base JUDI propagator (judiModeling)
`geometry`: the receiver interpolation (judiProjection) for data measurment
"""
function judiPhoto(F::judiPropagator{D, O}, geometry::Geometry;) where {D, O}
    initState = adjoint(judiInitialStateProjection{D}(space(F.model.n), time_space(F.model.n)))
    return judiPhoto{D, :forward}(rec_space(geometry), space(F.model.n), F, judiProjection(geometry), initState)
end

judiPhoto(model::Model, geometry::Geometry; options=Options()) = judiPhoto(judiModeling(model; options=options), geometry)
*(F::judiDataModeling{D, O}, I::jAdjoint{<:judiInitialStateProjection{D}}) where {D, O} = judiPhoto{D, :forward}(F.m, space(F.model.n), F.F, F.rInterpolation, I)

adjoint(J::judiPhoto{D, O}) where {D, O} = judiPhoto{D, adjoint(O)}(J.n, J.m, J.F, J.rInterpolation, J.Init)
getindex(J::judiPhoto{D, O}, i) where {D, O} = judiPhoto{D, O}(J.m[i], J.n[i], J.F, J.rInterpolation[i], J.Init)

make_input(J::judiPhoto{D, O}, q) where {D<:Number, O} = Geometry(J.rInterpolation.geometry), make_input(q, J.model, J.options)
make_input(J::judiPhoto{D, O}, ::Nothing) where {D<:Number, O} = Geometry(J.rInterpolation.geometry), nothing

*(J::judiPhoto{T, :forward}, q::Array{T, 3}) where {T} = J*vec(q)
*(J::judiPhoto{T, :forward}, q::Array{T, 4}) where {T} = J*vec(q)

process_input_data(::judiPhoto{D, :forward}, q::judiInitialState{D}) where {D<:Number} = q

############################################################

function _forward_prop(J::judiPhoto{T, O}, q::AbstractArray{T}, op::PyObject; dm=nothing) where {T, O}

    # Get necessary inputs 
    recGeometry, init_dist = make_input(J, q)

    # Set up Python model structure
    modelPy = devito_model(J.F.model, J.F.options, dm)
    dtComp  = convert(Float32, modelPy."critical_dt")
    nt = length(0:dtComp:recGeometry.t[1])

    # Set up coordinates
    rec_coords = setup_grid(recGeometry, J.F.model.n)

    # Devito interface
    dsim = wrapcall_data(op, modelPy, rec_coords, init_dist, nt, space_order=J.F.options.space_order)
    dsim = time_resample(dsim, dtComp, recGeometry)

    # Output shot record as judiVector
    return judiVector{Float32, Matrix{Float32}}(1, recGeometry, [dsim])
end

function _reverse_propagate(J::judiPhoto{T, O}, q::AbstractArray{T}, op::PyObject; init=nothing) where {T, O}
    
    # Get input data from source and operator
    srcData = q.data[1]
    recGeometry, init_dist = make_input(J, init)

    # Set up Python model structure
    modelPy = devito_model(J.F.model, J.F.options)
    dtComp  = convert(Float32, modelPy."critical_dt")

    # Set up coordinates
    rec_coords = setup_grid(recGeometry, J.F.model.n)

    # Extrapolate input data to computational grid     
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    args = isnothing(init) ? (modelPy, qIn, rec_coords) : (modelPy, qIn, rec_coords, init_dist)
    g = pycall(op, PyArray, args..., space_order=J.F.options.space_order, freq_list=nothing)

    g = remove_padding(g, modelPy.padsizes; true_adjoint=(J.options.sum_padding && ~isnothing(init)))
    return g
end

# JUDI interface for single source wave operator
propagate(J::judiPhoto{T, :forward}, q::AbstractArray{T}) where {T} = _forward_prop(J, q, impl."forwardis_data")
propagate(J::judiJacobian{D, :born, FT}, q::AbstractArray{T}) where {T, D, FT<:judiPhoto} = _forward_prop(J.F, J.q, impl."bornis_data"; dm=q)
propagate(J::judiPhoto{T, :adjoint}, q::AbstractArray{T}) where {T} = judiInitialState(_reverse_propagate(J, q, impl."adjointis"))
propagate(J::judiJacobian{D, :adjoint_born, FT}, q::AbstractArray{T}) where {T, D, FT<:judiPhoto} = PhysicalParameter(_reverse_propagate(J.F, q, impl."adjointbornis"; init=J.q), J.model.d, J.model.o)
