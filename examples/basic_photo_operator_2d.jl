using PhotoAcoustic
using JUDI
using Statistics, LinearAlgebra, PyPlot

# Set up model structure
n = (80, 80)   # (x,y,z) or (x,z)
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

# receiver sampling and recording time
time = 5.2333 #[microsec] 

# receiver sampling interval [microsec] 
# On the order of 10nanoseconds which 
# is similar to hauptmans 16.6ns
dt = calculate_dt(model) / 2    

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

###################################################################################################

# Need to fix this
opt = Options(dt_comp=dt)

F = judiModeling(model; options=opt)

# Setup operators
A = judiPhoto(F, recGeometry;)

# Photoacoustic source distribution
init_dist = zeros(Float32, n)
init_dist[div(n[1],2), div(n[2],2)] = 1
p = judiPhotoSource(init_dist);

dsim = A*p
p_adj = A'*dsim

# Plotting results
data_extent = (0, nxrec, time, 0)
model_extent = (0,(n[1]-1)*d[1],(n[2]-1)*d[2],0)

vmax_data = quantile(abs.(vec(dsim.data[1])), 98/100)

fig = figure(figsize=(10,5))
subplot(1,3,1)
imshow(p.data[1];extent=model_extent,cmap="gray")
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )
colorbar()

subplot(1,3,2)
imshow(dsim.data[1];extent=data_extent,cmap="seismic", aspect=15, vmin=-vmax_data, vmax=vmax_data)
xlabel("Receiver index"); ylabel("Time [microseconds]");
colorbar()

subplot(1,3,3)
imshow(p_adj.data[1]';extent=model_extent,cmap="gray")
ylabel("Z Position [mm]";); xlabel("X Position [mm]"; )
colorbar()

tight_layout()
savefig("data.png")





