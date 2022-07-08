export judiPhoto

struct judiPhoto{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    F::judiModeling
    rInterpolation::Geometry
end

function judiPhoto(F::judiPropagator{D, O}, rInterpolation::Geometry;) where {D, O}
    return judiPhoto{D, :forward}( rec_space(rInterpolation),space(F.model.n), F, rInterpolation)
end

adjoint(J::judiPhoto{D, O}) where {D, O} = judiPhoto{D, adjoint(O)}(J.n, J.m, J.F, J.rInterpolation)
getindex(J::judiPhoto{D, O}, i) where {D, O} = judiPhoto{D, O}(J.m[i], J.n[i], J.F, J.rInterpolation[i])


function make_input(J::judiPhoto{D, :forward}, q) where {D<:Number}
    recGeometry = Geometry(J.rInterpolation)
    init_dist = pad_array(q.data[1], pad_sizes(J.F.model, J.F.options; so=0); mode=:zeros)
    nt = J.rInterpolation.nt[1]
    return recGeometry, init_dist, nt
end 

*(J::judiPhoto{T, :forward}, q::Array{T, 3}) where {T} = J*vec(q)
*(J::judiPhoto{T, :forward}, q::Array{T, 4}) where {T} = J*vec(q)

JUDI.process_input_data(::judiPhoto{D, :forward}, q::judiPhotoSource{D}) where {D<:Number, N} = q

############################################################

function propagate(J::judiPhoto{T, :forward}, q::AbstractArray{T}) where {T, O}
    
    # Get necessary inputs 
    recGeometry, init_dist, nt = make_input(J,q)

    # Set up Python model structure
    modelPy = devito_model(J.F.model, J.F.options)
    dtComp  = convert(Float32, modelPy."critical_dt")

    # Set up coordinates
    rec_coords = setup_grid(recGeometry, J.F.model.n)    # shifts rec coordinates by origin

    # Devito interface
    dsim = JUDI.wrapcall_data(impl."forward_photo", modelPy, rec_coords, init_dist, nt, space_order=J.F.options.space_order)

    dsim = time_resample(dsim, dtComp, recGeometry)

    # Output shot record as judiVector
    return judiVector{Float32, Matrix{Float32}}(1, recGeometry, [dsim])
end

function propagate(J::judiPhoto{T, :adjoint}, q::AbstractArray{T}) where {T, O}
    
    # Get input data from source and operator
    srcData = q.data[1]
    recGeometry = Geometry(J.rInterpolation)

    # Set up Python model structure
    modelPy = devito_model(J.F.model, J.F.options)
    dtComp  = convert(Float32, modelPy."critical_dt")

    # Set up coordinates
    rec_coords = setup_grid(recGeometry, J.F.model.n)    # shifts rec coordinates by origin

    # Extrapolate input data to computational grid     
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    g = pycall(impl."adjoint_photo", PyArray, modelPy, qIn, rec_coords,  space_order=J.F.options.space_order)
    #println(size(g))
    g = remove_padding(g, modelPy.padsizes; true_adjoint=false)
   
    return judiPhotoSource(g);
end
