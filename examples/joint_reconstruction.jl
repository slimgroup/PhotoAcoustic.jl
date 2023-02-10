using DrWatson

using JLD2
using PhotoAcoustic
using Statistics, LinearAlgebra, PyPlot
using Images 
using Distributions 
using Random 
using Printf
using SlimPlotting
using FFTResampling
using DSP
using FFTW

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=300)
plot_path = "plots/dm"

@load "data/breast_2d.jld2" v rho p0
d = (0.08f0, 0.08f0)

p0, d = blackman_upscale(p0, d)
v = imresize(v, size(p0))
rho = imresize(rho, size(rho))

mask = v .== v[1,1]
# Set up model structure
n = size(v) # (x,y,z) or (x,z)
o = (0., 0.)

#Make background model with original v but without calcifications
v0 = 1 ./ imfilter(1 ./ v, Kernel.gaussian(8f0));
m0 = (1f0 ./ v0).^2;

# v = make_calcifications(v;x=100,z=100,spread=80, n=20)
m = (1f0 ./ v).^2;

#m0 = 0.4444f0*ones(Float32,n) #with water background 
dm = m0 - m

model = Model(n, d, o, m;)
model0 = Model(n, d, o, m0;)

# Set up receiver geometry
nxrec = 256
domain_x = (n[1] - 1)*d[1]
domain_z = (n[2] - 1)*d[2]
rad = .95*domain_x / 2
xrec, zrec, theta = circle_geometry((domain_x / 2, domain_z / 2), rad, nxrec)
yrec = 0f0 #2d so always 0

# receiver sampling and recording time
time_rec = 40.2333 #[microsec] 

# receiver sampling interval [microsec] 
# On the order of 10nanoseconds which 
# is similar to hauptmans 16.6ns
dt = 0.01f0
f0 = 1/dt

# Set up receiver structure
nsrc = 1	# number of sources
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time_rec, nsrc=nsrc)

############################# Create forward operators ############################################
opt = Options(IC="fwi")

F = judiModeling(model; options=opt)

# Setup operators
A = judiPhoto(F, recGeometry;)

# Photoacoustic source distribution
p = judiInitialState(p0);


# Make observed data
dsim = A*p
p_adj = A'*dsim

A0 = judiPhoto(model0, recGeometry;)
J = judiJacobian(A0, p)

d_lin = J*vec(dm)
dm_lin = J'*d_lin

dsim_back = A0*p

dm_back = J'*(dsim_back - dsim)
dm_nonlin = J'*dsim

############################################# Plotting results

##### Plot gradients wrt m 

# Linear gradient
fig = figure(figsize=(6,6))
plot_simage(dm_lin'; name="dm_lin = J(m0,p)'J(m0,p)dm", cbar=true, units=(:mm, :mm), new_fig=false, perc=98)

scatter(xrec, zrec; s=1)
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_dm_linear.png"), fig);

# nonlinear gradient
fig = figure(figsize=(6,6))
plot_simage(dm_nonlin'; name="dm_nonlin = -J(m0,p)'A(m)p with illum", cbar=true, units=(:mm, :mm), new_fig=false, perc=98)
scatter(xrec, zrec; s=1)
fig_name = @strdict time 
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"dm_nonlinear.png"), fig); 


fig = figure(figsize=(6,6))
plot_simage(dm_back'; name="dm_back = J(m0,p)'(A(m0)p - A(m)p)", cbar=true, units=(:mm, :mm), new_fig=false, perc=98)
scatter(xrec, zrec; s=1)
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_dm_back.png"), fig); 

##### Plot models

fig = figure(figsize=(6,6))
plot_velocity(m', d;cmap="gray", name="slowness m ground truth", new_fig=false, units=(:mm, :mm))
fig_name = @strdict  
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"m_ground_truth.png"), fig); 

fig = figure(figsize=(6,6))
plot_velocity(m0', d;cmap="gray", name="Background m0", new_fig=false, units=(:mm, :mm))
fig_name = @strdict  
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"m0.png"), fig); 


fig = figure(figsize=(6,6))
plot_velocity(dm', d;cmap="gray", name="dm", new_fig=false, units=(:mm, :mm))
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_dm.png"), fig); 



##### Plot photoacoustic source 

fig = figure(figsize=(6,6))
plot_simage(p.data[1]', d; name="photo source p ground truth", cbar=true, units=(:mm, :mm), new_fig=false, perc=98)
fig_name = @strdict 
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"p_ground_truth.png"), fig); 


fig = figure(figsize=(6,6))
plot_simage(p_adj.data[1]', d; name="adjoint source A'dsim", cbar=true, units=(:mm, :mm), new_fig=false, perc=98)
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"p_adjoint.png"), fig); 


##### Plot data 
fig = figure(figsize=(8,10))
plot_sdata(dsim_back[1];name="simulated background data A(m0)p",cmap="seismic", units=(:μs, :mm), new_fig=false)
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"d_back.png"), fig); 

data_to_plot = dsim_back - dsim
fig = figure(figsize=(8,10))
plot_sdata(data_to_plot[1];name="residual background data A(m0)p - A(m)p",cmap="seismic", units=(:μs, :mm), new_fig=false)
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"d_residaul.png"), fig); 


fig = figure(figsize=(8,10))
plot_sdata(d_lin[1];name="simulated linear data J(m0,p)dm",cmap="seismic", units=(:μs, :mm), new_fig=false)
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"d_lin.png"), fig); 

fig = figure(figsize=(8,10))
plot_sdata(dsim[1];name="simulated data dsim=A(m)p",cmap="seismic", units=(:μs, :mm), new_fig=false)
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"d_sim.png"), fig); 