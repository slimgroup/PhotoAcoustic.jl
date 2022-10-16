using DrWatson

using JLD2
using PhotoAcoustic
using Statistics, LinearAlgebra, PyPlot
using Images 
using Distributions 
using Random 
using Printf

function circle_geom(h, k, r, numpoints)
    #h and k are the center coords
    #r is the radius
    theta = LinRange(0f0, 2*pi, numpoints+1)[1:end-1]
    Float32.(h .- r*cos.(theta)), Float32.(k .+ r*sin.(theta)), theta
end

function make_calcifications(v;x=100,z=100,spread=80, n=20)
    #make n calcifications centered around x,y with a certain spread
    normal_two_dim = MvNormal(vec([x z]), Diagonal(spread*ones(2)))
    norm_coords =convert(Array{Int64,2},round.(rand(normal_two_dim, n)))
    norm_coords_x = norm_coords[1,:]
    norm_coords_z = norm_coords[2,:]
    for i in 1:n
        #use random size and location to place "microcalcification" of high velocity
        box_size_x = rand(0:2)
        box_size_z = rand(0:2)
        v[x+norm_coords_x[i]:x+norm_coords_x[i]+box_size_x, z+norm_coords_z[i]:z+norm_coords_z[i]+box_size_z] .= 3.43203 
    end
    v
end

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=300)
plot_path = "plots/dm"

@load "data/breast_2d.jld2" v rho p0

# Set up model structure
n = size(v) # (x,y,z) or (x,z)
d = (0.08f0, 0.08f0)
o = (0., 0.)

#Make background model with original v but without calcifications
v0 = 1 ./ imfilter(1 ./ v, Kernel.gaussian(8f0));
m0 = (1f0 ./ v0).^2;

v = make_calcifications(v;x=100,z=100,spread=80, n=20)
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
xrec, zrec, theta = circle_geom(domain_x / 2, domain_z / 2, rad, nxrec)
yrec = 0f0 #2d so always 0

# receiver sampling and recording time
time_rec = 40.2333 #[microsec] 

# receiver sampling interval [microsec] 
# On the order of 10nanoseconds which 
# is similar to hauptmans 16.6ns
dt = calculate_dt(model) / 2    

# Set up receiver structure
nsrc = 1	# number of sources
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time_rec, nsrc=nsrc)

############################# Create forward operators ############################################
opt = Options(dt_comp=dt)

F = judiModeling(model; options=opt)

# Setup operators
A = judiPhoto(F, recGeometry;)

# Photoacoustic source distribution

#make single vessel 
#p0_single = 0f0 .* p0
#p0_single[425:444,117:123] .= p0[425:444,117:123]
#p = judiInitialState(p0_single);

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
data_extent = (0, nxrec, time_rec, 0)
model_extent = (0,(n[1]-1)*d[1],(n[2]-1)*d[2],0)


##### Plot gradients wrt m 

# Linear gradient
dm_lin_to_plot = dm_lin.data'
a = quantile(abs.(vec(dm_lin_to_plot)), 98/100)
fig = figure(figsize=(6,6))
title("dm_lin = J(m0,p)'J(m0,p)dm")
imshow(dm_lin_to_plot;vmin=-a,vmax=a,extent=model_extent,cmap="gray")
cb = colorbar()
cb.ax.set_yticklabels([@sprintf("%.2e",i) for i in cb.get_ticks()])
PyPlot.scatter(xrec, zrec;s=1)
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )

tight_layout()
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_dm_linear.png"), fig); 


# nonlinear gradient
dm_nonlin_to_plot = dm_nonlin.data'
a = quantile(abs.(vec(dm_nonlin_to_plot)), 98/100)
fig = figure(figsize=(6,6))
title("dm_nonlin = -J(m0,p)'A(m)p with illum")
imshow(dm_nonlin_to_plot;vmin=-a,vmax=a,extent=model_extent,cmap="gist_ncar")
cb = colorbar(fraction=0.046, pad=0.04);
cb.ax.set_yticklabels([@sprintf("%.2e",i) for i in cb.get_ticks()])
PyPlot.scatter(xrec, zrec;s=1)
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )

tight_layout()
fig_name = @strdict time 
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"dm_nonlinear.png"), fig); 


dm_back_to_plot = dm_back.data'
a = quantile(abs.(vec(dm_back_to_plot)), 98/100)
fig = figure(figsize=(6,6))
title("dm_back = J(m0,p)'(A(m0)p - A(m)p)")
imshow(dm_back_to_plot;vmin=-a,vmax=a,extent=model_extent,cmap="gray")
cb = colorbar(fraction=0.046, pad=0.04);
cb.ax.set_yticklabels([@sprintf("%.2e",i) for i in cb.get_ticks()])
PyPlot.scatter(xrec, zrec; s=1)
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )

tight_layout()
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_dm_back.png"), fig); 

##### Plot models

fig = figure(figsize=(6,6))
title("slowness m ground truth")
imshow(m';extent=model_extent,cmap="gray")
cb = colorbar()
cb.ax.set_yticklabels([@sprintf("%.2e",i) for i in cb.get_ticks()])
PyPlot.scatter(xrec, zrec;s=1)
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )

tight_layout()
fig_name = @strdict  
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"m_ground_truth.png"), fig); 

fig = figure(figsize=(6,6))
title("background m0")
imshow(m0';extent=model_extent,cmap="gray")
cb = colorbar()
cb.ax.set_yticklabels([@sprintf("%.2e",i) for i in cb.get_ticks()])
PyPlot.scatter(xrec, zrec;s=1)
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )

tight_layout()
fig_name = @strdict  
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"m0.png"), fig); 


fig = figure(figsize=(6,6))
title("dm")
imshow(dm';extent=model_extent,cmap="gray")
cb = colorbar()
cb.ax.set_yticklabels([@sprintf("%.2e",i) for i in cb.get_ticks()])
PyPlot.scatter(xrec, zrec;s=1)
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )

tight_layout()
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_dm.png"), fig); 



##### Plot photoacoustic source 

fig = figure(figsize=(6,6))
title("photo source p ground truth")
imshow(p.data[1]';extent=model_extent,cmap="gray")
colorbar()
PyPlot.scatter(xrec, zrec;s=1)
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )

tight_layout()
fig_name = @strdict 
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"p_ground_truth.png"), fig); 


fig = figure(figsize=(6,6))
title("adjoint source A'dsim")
imshow(p_adj.data[1]';extent=model_extent,cmap="gray")
colorbar()
PyPlot.scatter(xrec, zrec;s=1)
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )

tight_layout()
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"p_adjoint.png"), fig); 



##### Plot data 
vmax_data = quantile(abs.(vec(dsim_back.data[1])), 98/100)
fig = figure(figsize=(8,10))
; title("simulated background data A(m0)p")
imshow(dsim_back.data[1];extent=data_extent,cmap="seismic", aspect="auto",vmin=-vmax_data, vmax=vmax_data)
xlabel("Receiver index"); ylabel("Time [microseconds]");
colorbar()
tight_layout()
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"d_back.png"), fig); 

data_to_plot = dsim_back - dsim
vmax_data = quantile(abs.(vec(data_to_plot.data[1])), 98/100)
fig = figure(figsize=(8,10))
; title("residual background data A(m0)p - A(m)p")
imshow(data_to_plot.data[1];extent=data_extent,cmap="seismic", aspect="auto",vmin=-vmax_data, vmax=vmax_data)
xlabel("Receiver index"); ylabel("Time [microseconds]");
colorbar()
tight_layout()
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"d_residaul.png"), fig); 

vmax_data = quantile(abs.(vec(d_lin.data[1])), 98/100)
fig = figure(figsize=(8,10))
; title("simulated linear data J(m0,p)dm")
imshow(d_lin.data[1];extent=data_extent,cmap="seismic", aspect="auto",vmin=-vmax_data, vmax=vmax_data)
xlabel("Receiver index"); ylabel("Time [microseconds]");
colorbar()
tight_layout()
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"d_lin.png"), fig); 

vmax_data = quantile(abs.(vec(dsim.data[1])), 98/100)
fig = figure(figsize=(8,10))
; title("simulated data dsim=A(m)p")
imshow(dsim.data[1];extent=data_extent,cmap="seismic", aspect="auto",vmin=-vmax_data, vmax=vmax_data)
xlabel("Receiver index"); ylabel("Time [microseconds]");
colorbar()
tight_layout()
fig_name = @strdict time
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"d_sim.png"), fig); 






