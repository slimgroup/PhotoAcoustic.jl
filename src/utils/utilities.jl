export circle_geometry, blackman_upscale

using JUDI.FFTW
using FFTResampling
using Images 

"""
    xs, ys, theta = circle_geometry(center_x, center_y, rad, numpoints)

Creates a set of coordinates in circle geometry. Used for making circular receiver geometries in 
photoacoustic simulations or circular source geometries in ultrasound simulations. 

Parameters:
* `center_x`: x coordinates of the circle center 
* `center_y`: y coordinates of the circle center 
* `rad`: r is the radius
* `numpoints`: number of points on the circle to return


Output:
* `xs`: x coordinates of the calculated points as an array
* `ys`: y coordinates of the calculated points as an array
* `theta`: angles that locate each point on the circle as an array

"""
function circle_geometry(center_x, center_y, r, numpoints)
    #center_
    #
    theta = LinRange(0f0, 2f0*pi, numpoints+1)[1:end-1]
    Float32.(center_x .- r*cos.(theta)), Float32.(center_y .+ r*sin.(theta)), theta
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
function blackman_upscale(p0, dx_orig, upsample_fact=1.25; pad_a=16)
    p0_zeropad = collect(padarray(p0, Fill(0,(pad_a,pad_a),(pad_a,pad_a))));
    Nx_orig = size(p0_zeropad)[1]
    x       = dx_orig*(Nx_orig - 1)
    dx_up   = Float32(dx_orig/upsample_fact)
    Nx_up   = ceil(Int, x/dx_up)

    ###################Smooth Iniital pressure distribution by upsampling and also blackman smooth
    #upsample p0 in FFT space
    p0_up = FFTResampling.resample(p0_zeropad, (Nx_up, Nx_up), true; boundary_handling=false)

    #smooth p0 using blackman filter
    window = Float32.(blackman(Nx_orig; padding=0));
    window2d = window*window';

    pad = (Nx_up-Nx_orig)÷2

    window_padded = collect(padarray(window2d, Fill(0,(pad,pad),(pad,pad))));

    p0_up_smooth = real.(ifft(fft(p0_up) .* ifftshift(window_padded)));
    p0_up_smooth = p0_up_smooth / maximum(p0_up_smooth);
    
    zero_pad = Int(upsample_fact*pad_a)
    p0_up_smooth = p0_up_smooth[zero_pad+1:end-zero_pad,zero_pad+1:end-zero_pad]

    return p0_up_smooth, dx_up
end