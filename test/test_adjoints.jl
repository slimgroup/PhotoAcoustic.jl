using PhotoAcoustic

using LinearAlgebra, Test, Printf

include("./runtests.jl")

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
dm = m .- m0

# Setup model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m;)
model0 = Model(n, d, o, m0;)

# Set up receiver geometry
nxrec = 64
xrec = range(0, stop=d[1]*(n[1]-1), length=nxrec)
yrec = [0f0]
zrec = range(0, stop=0, length=nxrec)

# receiver sampling and recording time
time = 5f0 #[microsec] 

# receiver sampling interval [microsec] 
# On the order of 10nanoseconds which 
# is similar to hauptmans 16.6ns
dt = 0.01f0

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# testing parameters and utils
tol = 5f-4
maxtry = 3

# Options
opt = Options(sum_padding=true, dt_comp=dt)

@testset "Jacobian test" begin
    w = zeros(Float32, model.n...)
    w[35:45, 35:45] .= randn(Float32, 11, 11)
    w = judiInitialState(w)
    # Setup operators
    F = judiPhoto(model, recGeometry; options=opt)
    F0 = judiPhoto(model0, recGeometry; options=opt)
    J = judiJacobian(F0, w)

    # Linear modeling
    dobs = F*w
    dD = J*vec(dm)

    # Gradient test
    grad_test(x-> F(;m=x)*w, m0, dm, dD; data=true)
end

@testset "FWI gradient test" begin
    w = zeros(Float32, model.n...)
    w[35:45, 35:45] .= randn(Float32, 11, 11)
    w = judiInitialState(w)
    # Setup operators
    F = judiPhoto(model, recGeometry; options=opt)
    dobs = F*w

    # Background operators
    F0 = judiPhoto(model0, recGeometry; options=opt)
    J = judiJacobian(F0, w)

	# Check get same misfit as l2 misifit on forward data
	grad = J'*(F0*w - dobs)

	grad_test(x-> .5f0*norm(F(;m=x)*w - dobs)^2, model0.m, dm, grad)
end

@testset "Photoacoustic  adjoint test with constant background" begin
    F = judiModeling(model; options=opt)
    Pr = judiProjection(recGeometry)
    I = judiInitialStateProjection(model)

	# Setup operators
	A = judiPhoto(F, recGeometry)
    A2 = Pr*F*I'
    A3 = judiPhoto(model, recGeometry; options=opt)

    w = zeros(Float32, model.n...)
    w[35:45, 35:45] .= randn(Float32, 11, 11)
    w = judiInitialState(w)

    # Nonlinear modeling
    y = A*w
    y2 = A2*w
    y3 = A3*w

    @test y ≈ y2
    @test y ≈ y3

    # Run test until succeeds in case of bad case
    adj_F, adj_J = false, false
    ntry = 0
    while (!adj_F || !adj_J) && ntry < maxtry
        q = rand(Float32, model.n...)
        q_rand = judiInitialState(q)

        adj_F, adj_J = run_adjoint(A, q_rand, y, vec(dm); test_F=!adj_F, test_J=!adj_J)
        ntry +=1
        test_adjoint(adj_F, adj_J, ntry==maxtry)
    end
    println("Adjoint test after $(ntry) tries")
end
