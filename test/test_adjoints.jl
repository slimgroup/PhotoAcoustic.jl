using PhotoAcoustic, LinearAlgebra, Test, Printf

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

# testing parameters and utils
tol = 5f-4
maxtry = 3

function run_adjoint(F, q, y; test_F=true )
    # Forward-adjoint
    d_hat = F*q
    q_hat = F'*y

    # Result F
    a = dot(y, d_hat)
    b = dot(q, q_hat)

    @printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e, ratio: %2.5e \n", a, b, (a - b)/(a + b), b/a)
    isapprox(a/(a+b), b/(a+b), atol=tol, rtol=0)
end


test_adjoint(f::Bool, last::Bool) = test_adjoint(f, last)
test_adjoint(adj::Bool, last::Bool) = (adj || last) ? (@test adj) : (@test_skip adj)

# Photoacoustic operator
@testset "Photoacoustic  adjoint test with constant background" begin
  
    opt = Options()
    F = judiModeling(model; options=opt)
    Pr = judiProjection(recGeometry)
    I = judiInitialStateProjection(model)

	# Setup operators
	A = judiPhoto(F, recGeometry)
    A2 = Pr*F*I'
    A3 = judiPhoto(model, recGeometry; options=opt)

    w = rand(Float32, model.n...)
    w = judiInitialState(w)

    # Nonlinear modeling
    y = A*w
    y2 = A2*w
    y3 = A3*w

    @test y ≈ y2
    @test y ≈ y3

    # Run test until succeeds in case of bad case
    adj_F = false
    ntry = 0
    while (!adj_F) && ntry < maxtry
        q = rand(Float32, model.n...)
        q_rand = judiInitialState(q)

        adj_F = run_adjoint(A, q_rand, y; test_F=!adj_F)
        ntry +=1
        test_adjoint(adj_F, ntry==maxtry)
    end
    println("Adjoint test after $(ntry) tries")
end
