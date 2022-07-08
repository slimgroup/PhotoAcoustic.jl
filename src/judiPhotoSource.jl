############################################################
# judiPhotoSource ##############################################
############################################################

export judiPhotoSource

############################################################

# structure for photoacoustic source data as an abstract vector
mutable struct judiPhotoSource{T<:Number} <: judiMultiSourceVector{T}
    nsrc::Integer
    #data::Array{T, N} where N
    data::Vector{Array{T, N}} where N
end

############################################################

## outer constructors

"""
    judiPhotoSource
        nsrc=1
        data
Abstract vector for photoacoustic source. Represents an the initial pressure wavefield distributed across space. There is no 
active source so nsrc always 1. Source structure is kept and src is set to 1 to inherit functions such as dot, norm etc
Constructors
============
`source` is 2D-3D array with spatial distribution of photoacoustic source
    judiPhotoSource(source)
"""

function judiPhotoSource(source::Array{T, N}) where {T<:Number, N}
    source = convert(Array{Float32}, deepcopy(source))
    sourceCell = [source]
    return judiPhotoSource{Float32}(1, sourceCell)
end

############################################################
# JOLI conversion
jo_convert(::Type{T}, jw::judiPhotoSource{T}, ::Bool) where {T<:AbstractFloat} = jw
jo_convert(::Type{T}, jw::judiPhotoSource{vT}, B::Bool) where {T<:AbstractFloat, vT} = judiPhotoSource{T}(jo_convert.(T, jw.data, B))
zero(::Type{T}, v::judiPhotoSource{T}) where {T} = judiPhotoSource{T}(1, T(0) .* v.data)
(w::judiPhotoSource)(x::Vector{<:Array}) = judiPhotoSource(x)

function copy!(jv::judiPhotoSource, jv2::judiPhotoSource)
    jv.data .= jv2.data
    jv
end
copyto!(jv::judiPhotoSource, jv2::judiPhotoSource) = copy!(jv, jv2)

#make_input(w::judiPhotoSource) = w.data

getindex(a::judiPhotoSource{avDT}, srcnum::RangeOrVec) where avDT = judiPhotoSource{avDT}(length(srcnum), a.data)