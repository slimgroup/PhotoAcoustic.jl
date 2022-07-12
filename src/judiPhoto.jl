export judiPhoto, judiInitialStateProjection

struct judiInitialStateProjection{D} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
end

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

function make_input(J::judiPhoto{D, :forward}, q) where {D<:Number}
    rec_geom = Geometry(J.rInterpolation.geometry)
    init_dist = pad_array(q.data[1], pad_sizes(J.F.model, J.F.options; so=0); mode=:zeros)
    return rec_geom, init_dist
end 

*(J::judiPhoto{T, :forward}, q::Array{T, 3}) where {T} = J*vec(q)
*(J::judiPhoto{T, :forward}, q::Array{T, 4}) where {T} = J*vec(q)

process_input_data(::judiPhoto{D, :forward}, q::judiInitialState{D}) where {D<:Number} = q

############################################################

function propagate(J::judiPhoto{T, :forward}, q::AbstractArray{T}) where {T}
    
    # Get necessary inputs 
    recGeometry, init_dist = make_input(J, q)

    # Set up Python model structure
    modelPy = devito_model(J.F.model, J.F.options)
    dtComp  = convert(Float32, modelPy."critical_dt")
    nt = floor(Int, recGeometry.t[1] / dtComp) + 1

    # Set up coordinates
    rec_coords = setup_grid(recGeometry, J.F.model.n)    # shifts rec coordinates by origin

    # Devito interface
    dsim = wrapcall_data(impl."forward_photo", modelPy, rec_coords, init_dist, nt, space_order=J.F.options.space_order)

    dsim = time_resample(dsim, dtComp, recGeometry)

    # Output shot record as judiVector
    return judiVector{Float32, Matrix{Float32}}(1, recGeometry, [dsim])
end

function propagate(J::judiPhoto{T, :adjoint}, q::AbstractArray{T}) where {T, O}
    
    # Get input data from source and operator
    srcData = q.data[1]
    recGeometry = Geometry(J.rInterpolation.geometry)

    # Set up Python model structure
    modelPy = devito_model(J.F.model, J.F.options)
    dtComp  = convert(Float32, modelPy."critical_dt")

    # Set up coordinates
    rec_coords = setup_grid(recGeometry, J.F.model.n)    # shifts rec coordinates by origin

    # Extrapolate input data to computational grid     
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    g = pycall(impl."adjoint_photo", PyArray, modelPy, qIn, rec_coords,  space_order=J.F.options.space_order)
    #println(size(g))
    g = remove_padding(g, modelPy.padsizes; true_adjoint=J.options.sum_padding)
   
    return judiInitialState(g)
end
