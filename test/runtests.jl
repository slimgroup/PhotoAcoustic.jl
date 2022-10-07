using PhotoAcoustic

using LinearAlgebra, Test, Printf


######## Copy paste from JUDI, should reorganize JUDI so can just be imported
test_adjoint(f::Bool, j::Bool, last::Bool) = (test_adjoint(f, last), test_adjoint(j, last))
test_adjoint(adj::Bool, last::Bool) = (adj || last) ? (@test adj) : (@test_skip adj)

mean(x) = sum(x)/length(x)

function run_adjoint(F, q, y, dm; test_F=true, test_J=true)
    adj_F, adj_J = !test_F, !test_J
    if test_F
        # Forward-adjoint
        d_hat = F*q
        q_hat = F'*y

        # Result F
        a = dot(y, d_hat)
        b = dot(q, q_hat)
        @printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e ratio : %2.5e \n", a, b, (a - b)/(a + b), b/a)
        adj_F = isapprox(a/(a+b), b/(a+b), atol=tol, rtol=0)
    end

    if test_J
        # Linearized modeling
        J = judiJacobian(F, q)
        ld_hat = J*dm
        dm_hat = J'*y

        c = dot(ld_hat, y)
        d = dot(dm_hat, dm)
        @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e ratio : %2.5e \n", c, d, (c - d)/(c + d), d/c)
        adj_J = isapprox(c/(c+d), d/(c+d), atol=tol, rtol=0)
    end
    return adj_F, adj_J
end

function grad_test(misfit, x0, dx, g; maxiter=6, h0=5f-2, data=false, stol=1f-1)
    # init
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)
    
    gdx = data ? g : dot(g, dx)
    f0 = misfit(x0)
    h = h0

    @printf("%11.5s, %11.5s, %11.5s, %11.5s, %11.5s, %11.5s \n", "h", "h * gdx", "e1", "e2", "rate1", "rate2")
    for j=1:maxiter
        f = misfit(x0 + h*dx)
        err1[j] = norm(f - f0, 1)
        err2[j] = norm(f - f0 - h*gdx, 1)
        j == 1 ? prev = 1 : prev = j - 1
        @printf("%5.5e, %5.5e, %5.5e, %5.5e, %5.5e, %5.5e \n", h, h*norm(gdx, 1), err1[j], err2[j], err1[prev]/err1[j], err2[prev]/err2[j])
        h = h * .8f0
    end

    rate1 = err1[1:end-1]./err1[2:end]
    rate2 = err2[1:end-1]./err2[2:end]
    # @test isapprox(mean(rate1), 1.25f0; atol=stol)
    # @test isapprox(mean(rate2), 1.5625f0; atol=stol)
end


# include("test_adjoints.jl")