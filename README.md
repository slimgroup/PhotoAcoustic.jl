
| **Documentation**     | **Build Status**    |       | 
|:---------------------:|:-------------------:|:-------------------:|
| [![][docs-stable-img]][docs-stable-status] [![][docs-dev-img]][docs-dev-status] | [![CI-PhotoAcoustic](https://github.com/slimgroup/PhotoAcoustic.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/slimgroup/PhotoAcoustic.jl/actions/workflows/CI.yml) |  [![DOI](https://zenodo.org/badge/509555475.svg)](https://zenodo.org/badge/latestdoi/509555475) |

# PhotoAcoustic.jl

PhotoAcoustic operators (forward and adjoint) with a high level linear algebra interface. This package is an extension of [JUDI] for photoacoustic simulations and introduces the use of initial value problems.

## Basic usage
This package is based on [JUDI] so please read that documentation to understand how to setup basic simulation parameters such as model size and receiver locations. Once a simulation has been designed in pure JUDI, the photoacoustic simulation is defined by using the ``judiPhoto`` operator and a ``judiInitialState`` as a source. Given ``F``, a ``judiModeling`` operator describing the acoustic simulation and ``recGeometry`` a ``Geometry`` object describing the receiver setup we can setup and run a photoacoustic simulation with a point source as follows:

```Julia
# Forward Photoacoustic operator
A = judiPhoto(F, recGeometry;)

# Photoacoustic source (n contains spatial dimensions)
init_dist = zeros(Float32, n)
init_dist[div(n[1],2), div(n[2],2)] = 1
p = judiInitialState(init_dist);

# Forward simulation
d_sim = A*p
```

A complete runnable script with the above example is [here](https://github.com/slimgroup/PhotoAcoustic.jl/blob/main/examples/basic_photo_operator_2d.jl)

In order to solve photoacoustic inverse problems in a variational framework, we also need the adjoint photoacoustic operator ``A^{\top}``. In this package, we [derive](https://slimgroup.github.io/PhotoAcoustic.jl/dev/derivations/ and implement the adjoint). The operator can be accessed with simple linear algebra notation: 
```Julia
# Forward simulation
d_sim = A*p

# Adjoint simulation (adjoint(A) also works)
p_adj = A'*d_sim 
```

The adjoint operator can be used to find sensitivities of data with respect to the initial source. This allows performing gradient descent and also [iterative methods](https://github.com/slimgroup/PhotoAcoustic.jl/blob/main/examples/notebooks/Least_Squares_Iterative_Solvers_2D.ipynb) to solve a least squares variational framework. Furthermore, we also derive the adjoint simulation that defines the sensitivity with respect to the speed of sound model used in the underlying acoustic simulation. This enables independent or joint optimization of the initial value ``p`` and also acoustic parameters ``m``. The sensitivity with respect to the acoustic model is related to the linearization of wave equation around a point ``model0`` so we can implement this in the familiar JUDI way and apply the adjoint Jacobian on the data residual ``dD``:

```Julia
# Forward simulation
F0 = judiPhoto(model0, recGeometry;)
J = judiJacobian(F0, p)
dm = J'*dD
```


## Integration with machine learning
A promising new topic is scientific machine learning that marries together physics based operators with learned operators. We exemplify this with the [deep image prior](https://github.com/slimgroup/PhotoAcoustic.jl/blob/main/examples/notebooks/Flux_AD_Integration_Deep_Prior.ipynb) a technique that requires chaining together gradients of the forward operator (in our case related to the solution of a partial differential equation) and gradients of learned layers. 

```Julia
loss, grad = Flux.withgradient(Flux.params(unet)) do
    norm(A*unet(z) - y).^2
end
Flux.Optimise.update!(opt, ps, grad)
```

## Authors
This package was written by [Mathias Louboutin](https://mloubout.github.io/) and [Rafael Orozco](https://github.com/rafaelorozco) from the Seismic Laboratory for Imaging and Modeling (SLIM) at the Georgia Institute of Technology.


[docs-stable-img]:https://img.shields.io/badge/docs-stable-blue.svg?style=plastic
[docs-stable-status]:https://slimgroup.github.io/PhotoAcoustic.jl/stable

[docs-dev-img]:https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-status]:https://slimgroup.github.io/PhotoAcoustic.jl/dev


[JUDI]:https://github.com/slimgroup/JUDI.jl

