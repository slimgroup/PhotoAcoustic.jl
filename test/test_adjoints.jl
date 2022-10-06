using PhotoAcoustic

using LinearAlgebra, Test, Printf
# Set up model structure
n = (80, 80)   # (x,y,z) or (x,z)
d = (0.08f0, 0.08f0)
o = (0., 0.)

# Constant water velocity [mm/microsec]
v = 1.5f0*ones(Float32,n)
v[:, 41:end] .= 2f0
m = (1f0 ./ v).^2
dm = zeros(Float32, n...)
dm[:, 40] .= .01f0

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
dt = 0.01

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# testing parameters and utils
tol = 5f-4
maxtry = 3

######## Copy paste from JUDI, should reorganize JUDI so can just be imported

function run_adjoint(F, q, y, dm; test_F=true, test_J=true)
    adj_F, adj_J = !test_F, !test_J
    if test_F
        # Forward-adjoint
        d_hat = F*q
        q_hat = F'*y

        # Result F
        a = dot(y, d_hat)
        b = dot(q, q_hat)
        @printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e ratio : %2.5e \n", a, b, (a - b)/(a + b), a/b)
        adj_F = isapprox(a/(a+b), b/(a+b), atol=tol, rtol=0)
    end

    if test_J
        # Linearized modeling
        J = judiJacobian(F, q)
        ld_hat = J*dm
        dm_hat = J'*y

        c = dot(ld_hat, y)
        d = dot(dm_hat, dm)
        @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e ratio : %2.5e \n", c, d, (c - d)/(c + d), c/d)
        adj_J = isapprox(c/(c+d), d/(c+d), atol=tol, rtol=0)
    end
    return adj_F, adj_J
end

test_adjoint(f::Bool, j::Bool, last::Bool) = (test_adjoint(f, last), test_adjoint(j, last))
test_adjoint(adj::Bool, last::Bool) = (adj || last) ? (@test adj) : (@test_skip adj)

# Photoacoustic operator
@testset "Photoacoustic  adjoint test with constant background" begin
  
    opt = Options(sum_padding=true)
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
