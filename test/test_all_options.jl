# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: October 2022

@testset "Gradient options test" begin
        ##################################ISIC########################################################
        printstyled("Testing isic \n"; color = :red)
        @timeit TIMEROUTPUT "ISIC" begin
                fs = rand([true, false])
                opt = Options(sum_padding=true, free_surface=fs, isic=true, f0=f0)
                F = judiPhoto(model0, recGeometry; options=opt)

                # Linearized modeling
                J = judiJacobian(F, w)
                @test norm(J*(0f0.*dm)) == 0

                y0 = F*w
                y_hat = J*dm
                x_hat1 = adjoint(J)*y0

                c = dot(y0, y_hat)
                d = dot(dm, x_hat1)
                @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
                @test isapprox(c, d, rtol=5f-2)
                @test !isnan(norm(y0))
                @test !isnan(norm(y_hat))
                @test !isnan(norm(x_hat1))
        end

        ##################################checkpointing###############################################
        printstyled("Testing checkpointing \n"; color = :red)
        @timeit TIMEROUTPUT "Checkpointing" begin
                fs = rand([true, false])
                opt = Options(sum_padding=true, free_surface=fs, optimal_checkpointing=true, f0=f0)
                F = judiPhoto(model0, recGeometry; options=opt)

                # Linearized modeling
                J = judiJacobian(F, w)

                y_hat = J*dm
                x_hat2 = adjoint(J)*y0

                c = dot(y0, y_hat)
                d = dot(dm, x_hat2)
                @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
                @test isapprox(c, d, rtol=1f-2)

                @test !isnan(norm(y_hat))
                @test !isnan(norm(x_hat2))
        end

        ##################################DFT#########################################################
        printstyled("Testing DFT \n"; color = :red)
        @timeit TIMEROUTPUT "DFT" begin
                fs = false
                opt = Options(sum_padding=true, free_surface=fs, frequencies=[2.5, 4.5], f0=f0)
                F = judiPhoto(model0, recGeometry; options=opt)

                # Linearized modeling
                J = judiJacobian(F, w)
                @test norm(J*(0f0.*dm)) == 0

                y_hat = J*dm
                x_hat3 = adjoint(J)*y0

                c = dot(y0, y_hat)
                d = dot(dm, x_hat3)
                @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
                @test !isnan(norm(y_hat))
                @test !isnan(norm(x_hat3))
        end

        ################################## DFT time subsampled#########################################
        printstyled("Testing subsampled in time DFT \n"; color = :red)
        @timeit TIMEROUTPUT "Subsampled DFT" begin
                fs = false
                opt = Options(sum_padding=true, free_surface=fs, frequencies=[2.5, 4.5], dft_subsampling_factor=4, f0=f0)
                F = judiPhoto(model0, recGeometry; options=opt)

                # Linearized modeling
                J = judiJacobian(F, w)
                @test norm(J*(0f0.*dm)) == 0

                y_hat = J*dm
                x_hat3 = adjoint(J)*y0

                c = dot(y0, y_hat)
                d = dot(dm, x_hat3)
                @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
                @test !isnan(norm(y_hat))
                @test !isnan(norm(x_hat3))
        end

        ##################################subsampling#################################################
        printstyled("Testing subsampling \n"; color = :red)
        @timeit TIMEROUTPUT "Subsampling" begin
                fs = rand([true, false])
                opt = Options(sum_padding=true, free_surface=fs, subsampling_factor=4, f0=f0)
                F = judiPhoto(model0, recGeometry; options=opt)

                # Linearized modeling
                J = judiJacobian(F, w)

                y_hat = J*dm
                x_hat4 = adjoint(J)*y0

                c = dot(y0, y_hat)
                d = dot(dm, x_hat4)
                @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
                @test !isnan(norm(y_hat))
                @test !isnan(norm(x_hat3))
        end
        ##################################ISIC + DFT #########################################################
        printstyled("Testing isic+dft \n"; color = :red)
        @timeit TIMEROUTPUT "ISIC+DFT" begin
                fs = false
                opt = Options(sum_padding=true, free_surface=fs, isic=true, frequencies=[2.5, 4.5], f0=f0)
                F = judiPhoto(model0, recGeometry; options=opt)

                # Linearized modeling
                J = judiJacobian(F, w)
                @test norm(J*(0f0.*dm)) == 0

                y_hat = J*dm
                x_hat5 = adjoint(J)*y0

                c = dot(y0, y_hat)
                d = dot(dm, x_hat5)
                @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
                @test !isnan(norm(y_hat))
                @test !isnan(norm(x_hat5))
        end
end
