# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: October 2022

# testing parameters and utils
tol = 5f-4
maxtry = 3

# Options
opt = Options(sum_padding=true)

F = judiPhoto(model, recGeometry; options=opt)
dobs = F*w

@testset "Jacobian test" begin
    @timeit TIMEROUTPUT "Jacobian test" begin
        # Setup operators
        F0 = judiPhoto(model0, recGeometry; options=opt)
        J = judiJacobian(F0, w)

        # Linear modeling
        dD = J*vec(dm)

        # Gradient test
        grad_test(x-> F(;m=x)*w, m0, dm, dD; data=true)
    end
end

@testset "FWI gradient test" begin
    @timeit TIMEROUTPUT "FWI gradient test" begin
        # Background operators
        F0 = judiPhoto(model0, recGeometry; options=opt)
        J = judiJacobian(F0, w)

        # Check get same misfit as l2 misifit on forward data
        grad = J'*(F0*w - dobs)

        grad_test(x-> .5f0*norm(F(;m=x)*w - dobs)^2, model0.m, dm, grad)
    end
end

@testset "Photoacoustic  adjoint test with constant background" begin
    @timeit TIMEROUTPUT "Adjoint test" begin
        Ainv = judiModeling(model; options=opt)
        Pr = judiProjection(recGeometry)
        I = judiInitialStateProjection(model)

        # Setup operators
        A = judiPhoto(Ainv, recGeometry)
        A2 = Pr*Ainv*I'
        A3 = judiPhoto(model, recGeometry; options=opt)

        # Nonlinear modeling
        y = A*w
        y2 = A2*w
        y3 = A3*w

        @test y ≈ y2
        @test y ≈ y3

        # Zero input
        @test norm(A*(0 .* w)) == 0

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
end


