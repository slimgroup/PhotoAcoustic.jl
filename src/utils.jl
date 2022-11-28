export circle_geometry, blackman_upscale

"""
    xs, ys, theta = circle_geometry(center, rad, numpoints)

Creates a set of coordinates in circle geometry. Used for making circular receiver geometries in 
photoacoustic simulations or circular source geometries in ultrasound simulations. 

Parameters:
* `center`: coordinates of the circle center, tuple or vector
* `rad`: r is the radius
* `numpoints`: number of points on the circle to return

Output:
* `coords`: Coordinates of the points on the circle/sphere
* `angles`: angles that locate each point on the circle as an array

"""
function circle_geometry(center::NTuple{2, T}, r, numpoints::Integer) where T
    cx, cy = center
    theta = LinRange(0f0, 2f0*pi, numpoints+1)[1:end-1]
    sct = sincos.(theta)
    st, ct = first.(sct), last.(sct)
    Float32.(cx .- r*ct), Float32.(cy .- r*st), Vector{Float32}(theta)
end

function circle_geometry(center::NTuple{3, T}, r, numpoints::Integer) where T
    cx, cy, cz = center
    theta = LinRange(0f0, 2f0*pi, numpoints+1)[1:end-1]
    phi = LinRange(0f0, pi, numpoints+1)[1:end-1]

    sct = sincos.(theta)
    st, ct = first.(sct), last.(sct)
    scp = sincos.(phi)
    sp, cp = first.(scp), last.(scp)
    Float32.(cx .- r*vec(ct*sp')), Float32.(cy .- r*vec(st*sp')), Float32.(cz .- r*vec(cp')), (theta, phi)
end

"""
    p0_up_smooth, dx_up = blackman_upscale(p0, dx_orig, upsample_fact=1.25; pad_a=16)

Upsamples and smooths a spatial distribution. Uses FFT for upsampling and then a blackman window filter for 
smoothing.  

Parameters:
* `p0`: spatial distribution as array. 
* `dx_orig`: original discretization of spatial distribution
* `upsample_fact`: factor to upsample by
* `pad_a`: padding on array to avoid edge artifacts when doing FFT upsampling

Output:
* `p0_up_smooth`: smoothed and upsampled array 
* `dx_up`: discretization of new array

"""
function blackman_upscale(p0::Array{T, N}, d::NTuple{N, T}, upsample_fact=1.25; pad_a=16) where {T, N}
    # Some safety checks
    try
        Int(upsample_fact * pad_a)
    catch InexactError
        throw(ArgumentError("Padding must be integer after upsampling, pad*up = $(upsample_fact * pad_a)"))
    end
    # Pad input
    pad = ntuple(_->(pad_a, pad_a), N)
    p0_zeropad = pad_zeros(p0, pad)
    N_orig = size(p0_zeropad)
    x       = d .* (N_orig .- 1)
    d_up   = Float32.(d ./ upsample_fact)
    N_up   = ceil.(Int, x ./ d_up)

    ###################Smooth Iniital pressure distribution by upsampling and also blackman smooth
    #upsample p0 in FFT space
    p0_up = FFTResampling.resample(p0_zeropad, N_up, true; boundary_handling=false)

    #smooth p0 using blackman filter
    window = Float32.(blackman(N_orig; padding=0));

    pad = (N_up .- N_orig) .÷ 2
    pad = ntuple(i-> (pad[i], pad[i]), N)

    window_padded = pad_zeros(window, pad)

    p0_up_smooth = real.(ifft(fft(p0_up) .* ifftshift(window_padded)))
    p0_up_smooth = p0_up_smooth / norm(p0_up_smooth, Inf)
    
    zero_pad = Int(upsample_fact * pad_a)
    zpad = ntuple(_ -> (zero_pad, zero_pad), N)
    p0_up_smooth = unpad(p0_up_smooth, zpad)

    return p0_up_smooth, d_up
end


function pad_zeros(m::Array{T, N}, nb::NTuple{N, NTuple{2, Int64}}) where {T, N}
    n = size(m)
    Ei = []
    new_size = n .+ map(sum, nb)
    for ((left, right), nn) in zip(nb, n)
        push!(Ei, joExtend(nn, :zeros; pad_upper=right, pad_lower=left, RDT=T, DDT=T))
    end
    padded = joKron(Ei...) * m[:]
    return reshape(padded, new_size)
end


function unpad(m::Array{T, N}, nb::NTuple{N, NTuple{2, Int64}}) where {T, N}
    n = size(m)
    Ei = []
    new_size = n .- map(sum, nb)
    for ((left, right), nn) in zip(nb, new_size)
        push!(Ei, joExtend(nn, :zeros; pad_upper=right, pad_lower=left, RDT=T, DDT=T))
    end
    padded = joKron(Ei...)' * m[:]
    return reshape(padded, new_size)
end