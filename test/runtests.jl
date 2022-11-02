using PhotoAcoustic

using LinearAlgebra, Test, Printf

using TimerOutputs: TimerOutputs, @timeit

# Collect timing and allocations information to show in a clear way.
const TIMEROUTPUT = TimerOutputs.TimerOutput()
timeit_include(path::AbstractString) = @timeit TIMEROUTPUT path include(path)

include("utils.jl")


##### Setup model, ... for test


# Set up model structure
n = (80, 80)   # (x,y,z) or (x,z)
d = (0.08f0, 0.08f0)
o = (0., 0.)

# Constant water velocity [mm/microsec]
v = 1.5f0*ones(Float32,n)
v[:, 41:end] .= 2f0
v0 = 1f0 .* v
v0[:, 41:end] .= 1.85f0
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2

# Setup model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m;)
model0 = Model(n, d, o, m0;)
dm = model.m - model0.m

# Set up receiver geometry
nxrec = 64
xrec = range(0.08f0, stop=d[1]*(n[1]-2), length=nxrec)
yrec = [0f0]
zrec = range(0.32f0, stop=0.32f0, length=nxrec)

# receiver sampling and recording time
time = 5f0 #[microsec] 

# receiver sampling interval [microsec] 
# On the order of 10nanoseconds which 
# is similar to hauptmans 16.6ns
dt = 0.01f0
f0 = 1/0.08

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

w = zeros(Float32, model.n...)
w[35:45, 35:45] .= 1f0.+ rand(Float32, 11, 11)
w = judiInitialState(w)


#####  Run tests

include("test_all_options.jl")
include("test_sensitivities.jl")

# Testing memory and runtime summary
show(TIMEROUTPUT; compact=true, sortby=:firstexec)
