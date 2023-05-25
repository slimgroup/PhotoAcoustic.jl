############################################################
# judiInitialState ##############################################
############################################################

export judiInitialState

############################################################

# structure for photoacoustic source data as an abstract vector
mutable struct judiInitialState{T<:Number} <: judiMultiSourceVector{T}
    nsrc::Integer
    data::Vector{Array{T, N}} where N
end

############################################################
## outer constructors

"""
    judiInitialState(field)

Construct an the initial pressure wavefield distributed across space for each experiments.
This is a single-time intistate and assumes that the initial time derivative is zero to define
the second time step of the second order wave equation. 

Arguments
============
`source`: a (Vector of) 2D-3D array with spatial distribution of photoacoustic sources

"""
judiInitialState(source::Vector{Array{T, N}}) where {T<:Number, N} = judiInitialState{T}(length(source), source)
judiInitialState(source::Array{T, N}) where {T<:Number, N} = judiInitialState([source])
judiInitialState(source::Array{T, N}, nsrc::Integer) where {T<:Number, N} = judiInitialState([source for s=1:nsrc])
judiInitialState(x::judiInitialState) = x

############################################################
# JOLI conversion
jo_convert(::Type{T}, jw::judiInitialState{T}, ::Bool) where {T<:AbstractFloat} = jw
jo_convert(::Type{T}, jw::judiInitialState{vT}, B::Bool) where {T<:AbstractFloat, vT} = judiInitialState{T}(jo_convert.(T, jw.data, B))
zero(::Type{T}, v::judiInitialState{T}) where {T} = judiInitialState{T}(1, T(0) .* v.data)
(w::judiInitialState)(x::Vector{<:Array}) = judiInitialState(x)

function copy!(jv::judiInitialState, jv2::judiInitialState)
    copy!(jv.data, jv2.data)
    jv
end

copyto!(jv::judiInitialState, jv2::judiInitialState) = copy!(jv, jv2)
getindex(a::judiInitialState{T}, srcnum::RangeOrVec) where T = judiInitialState{T}(length(srcnum), a.data)
make_input(q::judiInitialState, model::JUDI.AbstractModel, options::JUDIOptions) = pad_array(q.data[1], pad_sizes(model, options; so=0); mode=:zeros)

# Parallel reduction
JUDI.single_reduce!(J::judiInitialState, I::judiInitialState) = push!(J, I)

# push!
function JUDI.push!(a::judiInitialState{T}, b::judiInitialState{T}) where {T}
    append!(a.data, b.data)
    a.nsrc += b.nsrc
end
