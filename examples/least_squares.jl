using PhotoAcoustic
using JUDI, LinearAlgebra, PyPlot
using Statistics
using DrWatson
using IterativeSolvers

# Set up model structure
n = (80, 80)   # (x,z)
d = (0.08f0, 0.08f0)
o = (0., 0.)

# Constant water velocity [mm/microsec]
v = 1.5*ones(Float32,n) 
m = (1f0 ./ v).^2

# Setup model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m;)

# Set up receiver geometry
nxrec = 64
xrec = range(0, stop=d[1]*(n[1]-1), length=nxrec)
yrec = [0f0]
zrec = range(0, stop=0, length=nxrec)

# receiver recording time in [microsec]  
time = 5.2333 

# receiver sampling interval [microsec] 
# On the order of 10nanoseconds which 
# is similar to hauptmans 16.6ns
dt = calculate_dt(model) / 2    

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

############################# Build Photoacoustic operator##############################################

# Need to fix this
opt = Options(dt_comp=dt)

# Setup operators
F = judiModeling(model; options=opt)
A = judiPhoto(F, recGeometry;)

############################# Build Photoacoustic source ##############################################

# Random
init_dist = randn(Float32, n)
p = judiPhotoSource(init_dist);

############################# Simulate forward data    ##############################################

# Make observed data from ground truth p
d_obs = A*p

# Make adjoint solution from observed data 
p_adj = A'*d_obs

# Use iterative solver for least squares solution
maxiters = 5
p_lsqr = 0f0 .* p
_, history = lsqr!(p_lsqr, A, d_obs; maxiter=maxiters, verbose=true, log=true)
d_pred = A*p_lsqr

# Plotting results
data_extent = (0, nxrec, time,0)
model_extent = (0,(n[1]-1)*d[1],(n[2]-1)*d[2],0)
vmax_data = quantile(abs.(vec(d_obs.data[1])), 98/100)

fig = figure(figsize=(10,7))
suptitle("LSQR results with $(maxiters) iterations")
subplot(2,3,1); title("Ground truth") 
imshow(p.data[1]';extent=model_extent,cmap="gray")
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; ); colorbar()

subplot(2,3,2);  title("Adjoint solution")
imshow(p_adj.data[1]';extent=model_extent,cmap="gray")
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; ); colorbar()

subplot(2,3,3); title("LSQR solution")
imshow(p_lsqr.data[1]';extent=model_extent,cmap="gray")
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; ); colorbar()

subplot(2,3,4); title("Observed data")
imshow(d_obs.data[1];extent=data_extent,cmap="seismic", aspect=15, vmin=-vmax_data, vmax=vmax_data)
xlabel("Receiver index"); ylabel("Time [microseconds]");; colorbar()

subplot(2,3,5); title("Predicted data")
imshow(d_pred.data[1];extent=data_extent,cmap="seismic", aspect=15, vmin=-vmax_data, vmax=vmax_data)
xlabel("Receiver index"); ylabel("Time [microseconds]");; colorbar()

tight_layout()
fig_name = @strdict maxiters 
safesave(savename(fig_name; digits=6)*"_lsqr.png", fig); #close(fig)

