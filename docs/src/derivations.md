# PhotoAcoustic.jl Adjoint and Jacobian derivations. 

To the best of our knowledge, the literature does not contain a derivation for the adjoint of the forward photoacoustic operator in time and its Jacobian with respect to the speed of sound. 
## Forward photoacoustic modelling
The forward model of a photoacoustic experiment can be described by the second order wave equation with active source:
$$
\frac{1}{c(x,y)^2}\frac{\partial^2}{\partial t^2}u(x,y,t) - \nabla^2 u(x,y,t) = 0
$$
where
$$ 
\begin{align}
u(x,0) &= p_0 \\
\dot u(x,0) &= 0
\end{align}
$$

## Forward model adjoint derivation
The derivation is based on procedure and notation from [PDE-constrained optimization and the adjoint method][https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf]

We start by defining our optimization problem:
$minimize F(s,p_) where F(s,p_0) = $



## Jacobian derivation


