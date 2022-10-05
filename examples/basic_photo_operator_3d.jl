using PhotoAcoustic
using JUDI
using Statistics, LinearAlgebra, PyPlot

function plot_3d_mip(p_array, title_plt;dx=0.0678f0, vmin_po=0, vmax_po=nothing)

	nx, ny, nz = size(p_array)
	extentx = nx*dx  #in mm
	extentz = nz*dx  #in mm
	extenty = ny*dx  #in mm

	fig=figure(1,figsize=(6, 6)); axis("off")
	widths = [1, 1]; heights = [1, 1]
	spec5 = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
	                          height_ratios=heights)
	spec5 = reshape(collect(spec5),2,2)
	title(title_plt);

	fig.add_subplot(spec5[1]); plt.xticks([])
	imshow(maximum(p_array,dims=3)[:,:,1], extent=(0, extentz, extentx,0), vmin=vmin_po, vmax=vmax_po )
	ylabel("Depth Z [mm]");

	fig.add_subplot(spec5[3])
	imshow(maximum(p_array,dims=1)[1,:,:]', extent=(0,extentz,extenty,0) , vmin=vmin_po, vmax=vmax_po)
	xlabel("Lateral Position Y [mm]"); ylabel("Vertical Position X [mm] ");

	fig.add_subplot(spec5[4]); plt.yticks([])
	imt = imshow(maximum(p_array,dims=2)[:,1,:]', extent=(0,extentx,extentz,0), vmin=vmin_po, vmax=vmax_po)
	xlabel("Depth Z [mm]");
	fig.colorbar(imt, ax=fig.axes,fraction=0.049, pad=0.04)

	return fig 
end

# Set up model structure
n = (80, 80, 80)   # (x,y,z) or (x,z)
d = (0.08f0, 0.08f0, 0.08f0)
o = (0., 0., 0.)

# Constant water velocity [mm/microsec]
v = 1.5*ones(Float32,n) 

# Model is parameterized in slowness squared
m = (1f0 ./ v).^2

# Setup model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m;)

# Set up receiver geometry
nyrec = 120
nzrec = 120
# side 1 top
yrec_1 = range(d[1], stop=d[1]*(240-1f0), length=nyrec);
zrec_1 = range(d[1], stop=d[1]*(240-1f0), length=nzrec);
xrec_1 = 0f0;

# Construct 3D grid from basis vectors
(yrec, zrec, xrec) = setup_3D_grid(yrec_1, zrec_1, xrec_1)

# receiver sampling and recording time
time = 5.2333f0 #[microsec] 

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

# Photoacoustic point source distribution
init_dist = zeros(Float32, n)
init_dist[div(n[1],2), div(n[2],2), div(n[3],2)] = 1
p = judiInitialState(init_dist);

dsim = A*p
p_adj = A'*dsim

#################################### plotting results #########################################

data_extent = (0, nxrec, time, 0)
model_extent = (0,(n[1]-1)*d[1],(n[2]-1)*d[2],0)

vmax_data = quantile(abs.(vec(dsim.data[1])), 98/100)

fig = plot_3d_mip(p.data[1], "Ground Truth";)
savefig("gt_3d.png"); close(fig);

fig = plot_3d_mip(p_adj.data[1], "Adjoint solution";)
savefig("adj_3d.png"); close(fig);




