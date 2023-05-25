export judiTransducerProjection

"""
struct TransducerGeometry
    G::Geometry
    r::Vector{Vector{T}}
    θ::Vector{Vector{T}}
    ϕ::Vector{Vector{T}}
    ψ::Vector{Vector{T}}


A geometry for transducer. Unlike generic point geometries, transducer have an angle and
radius used to define a dirctional transducer for each point in the geometry. 
    
This is an internal type used for the propagation and shouldn't be used directly
"""
mutable struct TransducerGeometry{T, D} <: Geometry{T}
    geometry::Geometry{T}
    r::Vector{Vector{T}}
    θ::Vector{Vector{T}}
    ϕ
    ψ
end

get_nsrc(G::TransducerGeometry) = get_nsrc(G.geometry)
n_samples(G::TransducerGeometry, nsrc::Integer) = n_samples(G.geometry, nsrc)

function getindex(G::TransducerGeometry{T, D}, I) where {T, D}
    iϕ = isnothing(G.ϕ) ? nothing : G.ϕ[I]
    iψ = isnothing(G.ψ) ? nothing : G.ψ[I]
    return TransducerGeometry{T, D}(G.geometry[I], G.r[I], G.θ[I], iϕ, iψ)
end

getindex(G::TransducerGeometry, I::Integer) = getindex(G, I:I)
getproperty(G::TransducerGeometry, s::Symbol) = s ∈ [:geometry, :r, :θ, :ϕ, :ψ] ? getfield(G, s) : getproperty(G.geometry, s)
Geometry(G::TransducerGeometry{T, D}) where {T, D} = TransducerGeometry{T, D}(Geometry(G.geometry), G.r, G.θ, G.ϕ, G.ψ)

"""
    judiTransducerProjection

Transducer projection operator for sources/receivers to restrict to or inject data from a transducer

Examples
========
`F` is a modeling operator of type `judiModeling` and `q` is a seismic source of type `judiVector`:
Pr = judiTransducerProjection(rec_geometry, radius, theta)
Ps = judiTransducerProjection(q.geometry, radius, theta)
dobs = Pr*F*Ps'*q
qad = Ps*F'*Pr'*dobs

Parameters
========

* `geometry`: JUDI Geometry structure containing the positions and time axis parameters of the transducer
* `d`: Grid spacing (single number). Used to choose the number of points defining the transducer.
* `radius`: Radius for each transducer. 
* `theta`: Orientation for each transducer
* `phi`: Orientation for each transducer
* `psi`: Orientation for each transducer

The direction of the transducer (theta, phi, psi) is used to define the orientation of it's orthogonal plane.
The roations are performed in order ZYX (psi, phi, theta)
"""
function judiTransducerProjection(G::Geometry{T}, d::AbstractFloat, radius::Vector{Vector{T}}, theta::Vector{Vector{T}}, phi=nothing, psi=nothing) where T
    # Check that all sizes match
    size.(radius) == size.(G.xloc) || throw(ArgumentError("One radius per transducer required"))
    size.(theta) == size.(G.xloc) || throw(ArgumentError("One angle per transducer required"))
    ~isnothing(phi) && (size.(phi) == size.(G.xloc) || throw(ArgumentError("One angle per transducer required")))
    ~isnothing(psi) && (size.(psi) == size.(G.xloc) || throw(ArgumentError("One angle per transducer required")))
    if (isnothing(phi) && isnothing(psi))
        @warn "Only one angle provided, assuming 2D model"
    end
    geom = TransducerGeometry{T, T(d)}(G, radius, theta, phi, psi)
    judiProjection{Float32}(rec_space(G), time_space_src(get_nsrc(geom), G.nt, 3), geom)
end

judiTransducerProjection(G::Geometry, d::AbstractFloat, radius::T, theta) where T = judiTransducerProjection(G, d, fill!.(similar.(G.xloc), radius), convert(Vector{Vector{T}}, theta))

"""
    setup_grid(geometry::TransducerGeometry, n)

Sets up the coordinate arrays for Devito. This is the main interface that translates the position of the
transducer into a plane

Parameters:
* `geometry`: Geometry containing the coordinates
* `n`: Domain size

For visualization ,in 2D, the point source is converted into:

    Theta=0 points downward:
    
    . . . . - - - . . . . . .
    
    . . . . + + + . . . . . .
    
    . . . . . . . . . . . . .
    
    . . . . . . . . . . . . .
    
    
    Theta=pi/2 points right:
    
    . . . . - + . . . . . . .
    
    . . . . - + . . . . . . .
    
    . . . . - + . . . . . . .
    
    . . . . . . . . . . . . .
    

"""
function setup_grid(G::TransducerGeometry{T, D}, n::Tuple{Integer, Integer}) where {T, D}
    @assert get_nsrc(G) == 1 "Multiple sources are used in a single-source propagation"
    N = length(n)
    # Emprirical, use 11 point for the radius, 11x11 in 3D
    # Array of source
    nint = Int(max(1, div(G.r[1][1], D)))
    nb = nint*11
    nm = zeros(T, 1, nb)
    nm_b = zeros(T, 1, nb) .- D
    # angle for each point in the shot record
    in_coords = setup_grid(G.geometry, n)
    nrec = size(in_coords, 1)
    # Augmented position
    out_coords = zeros(T, nrec, 2 * nb^(N-1), N)
    for r=1:nrec
        width = collect(range(-G.r[1][r], G.r[1][r], length=nb)')
        R = _rMatrix(G.θ, G.ϕ, G.ψ, r, Val(N))
        # +1 coords
        front = rotate(R, width, nm)
        # -1 coords
        back = rotate(R, width, nm_b)
        # Create transducer "plane"
        for d=1:N
            out_coords[r, :, d] = in_coords[r, d] .+ vec(vcat(front[d, :], back[d, :]))
        end
    end
    # Reshape and return
    return reshape(out_coords, :, N)
end

# Rotation matrices with z pointing down convention (- pi/2)
function _rMatrix(θ::Vector{Vector{T}}, ::Any, ::Any, r::Integer, ::Val{2}) where T 
    sz, cz = sincos(θ[1][r] - pi/2)
    return T[cz -sz; sz cz]
end

function _rMatrix(θ::Vector{Vector{T}}, ϕ::Vector{Vector{T}}, ψ::Vector{Vector{T}}, r, ::Integer, ::Val{3}) where T
    sx, cx = sincos(θ[1][r] - pi/2)
    sy, cy = sincos(ϕ[1][r] - pi/2)
    sz, cz = sincos(ψ[1][r] - pi/2)
    # We use the order ZYX for the rotation orders
    return T[cy*cz cz*sx*sy-cx*sz cx*cz*sy+sx*sz;
             cy*sz cx*cz+sx*sy*sz -cz*sx+cx*sy*sz;
             -sy cy*sx cx*cy]
end

# Apply matrix
rotate(R::Array{T, 2}, width::Matrix{T}, nm::Matrix{T}) where T = R * [width; nm]
rotate(R::Array{T, 3}, width::Matrix{T}, nm::Matrix{T}) where T = R * [width; width; nm]

# Rebuild "summed" judiVector from transducer as a receiver
function adjoint_transducer(geom::TransducerGeometry{T, D}, data::AbstractMatrix{T}) where {T, D}
    nt, nrec = geom.geometry.nt[1], geom.geometry.nrec[1]
    # Radius in number of points
    n = (isnothing(geom.ψ) && isnothing(geom.ϕ)) ? 2 : 3
    nint = Int(max(1, div(geom.r[1][1], D)))
    nr = 2 * (nint*11)^(n-1)
    # Reshape data
    data = reshape(data, nt, nrec, :)
    # Check dimension is correct
    @assert size(data, 3) == nr "Incorrect number of traces, expected $(nr) but got $(size(data, 3))"
    # Weights are currently just uniform
    w = ones(T, 1, 1, size(data, 3)) ./ size(data, 3)
    w[:, :, div(size(data, 3), 2)+1:end] .*= -1
    data = sum(data, dims=3)[:, :, 1]
    return judiVector{Float32, Matrix{Float32}}(1, geom.geometry, [data])
end

time_resample(q::Matrix{T}, G::TransducerGeometry{T}, dtComp::T) where T<:AbstractFloat = time_resample(_augment_data(q, G), G.geometry, dtComp)

function _augment_data(q::Matrix{T},  G::TransducerGeometry{T, D}) where {T, D}
    n = (isnothing(G.ψ) && isnothing(G.ϕ)) ? 2 : 3
    nint = Int(max(1, div(G.r[1][1], D)))
    nb = (nint*11)^(n-1)
    nt, nrec = size(q)
    q_trans = zeros(T, nt, nrec, 2*nb)
    w = 1 / nb
    q_trans[:, :, 1:nb] .= q
    q_trans[:, :, nb+1:end] .= -q
    q_trans .*= w
    return reshape(q_trans, nt, :)
end

# Modeling output
post_process(v::AbstractArray, modelPy::PyObject, ::Val{:forward}, G::TransducerGeometry, options::JUDIOptions) =
    adjoint_transducer(G, time_resample(v, calculate_dt(modelPy), G.geometry))

post_process(v::AbstractArray, modelPy::PyObject, ::Val{:adjoint}, G::TransducerGeometry, options::JUDIOptions) =
    adjoint_transducer(G, time_resample(v, calculate_dt(modelPy), G.geometry))
