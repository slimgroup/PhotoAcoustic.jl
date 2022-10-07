
# PhotoAcoustic.jl Adjoint and Jacobian derivations. 
To the best of our knowledge, the literature does not contain a derivation for the adjoint of the forward photoacoustic operator using the second order wave equation and its Jacobian with respect to the speed of sound. 

## Forward model
The forward model of a photoacoustic experiment can be described by the second order wave equation with active source for space $x$ and time $t$:

$$\frac{1}{c(x)^2}\frac{\partial^2}{\partial t^2}u(x,t) - \nabla^2 u(x,t) = 0$$

where the source is instead defined in the initial state:
```math
\begin{aligned}
u(x,0) &= p_0 \\
\dot u(x,0) &= 0.
\end{aligned}
```

The spatial distribution $p_0$ is the initial acoustic source caused by the photonic impulse and is the parameter of interest when performing inversion. 

## Adjoint model derivation
The derivation is based on procedure and notation from [PDE-constrained optimization and the adjoint method](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf). We start by defining our optimization problem:

$$\underset{p_0}{\operatorname{min}} \, \, F(u,p_0) \, \,  \mathrm{where} \, \, F(u,p_0) = \int_{0}^{T}f(u,t)dt = \frac{1}{2} \sum_{r=1}^{R} \int_{0}^{T}\left|| u_{r}(t) - d_{r}(t)\right||_{2}^{2}dt$$

```math
\begin{aligned}
\text{subject to} \, \, m \ \ddot u &= \nabla^2 u\\
u(0) &= p_0 \\
\dot u(0) &= 0. 
\end{aligned}
```

Where we have set $m=\frac{1}{c(x)^2}$ to be the slowness squared. For ease of derivation of writing the Lagrangian, we will assign functions to each of these three constraints:

```math
\begin{aligned}
 h(\ddot u,p_0,t) &= m \ddot u - \nabla^2 u\\
g(u(0),p_0) &= u(0) - p_0 = 0 \\
k(\dot u(0)) &= \dot u(0) = 0
\end{aligned}
```

In order to solve the minimization problem, our goal is to obtain the sensitivity of our functional $F$ with respect to the variable $p_0$:

$$d_{p_0}F.$$

We start by writing out the full Lagrangian containing our functional and constraints: [...can add more details here...]. By selecting the correct multipliers [...can add more details here...] we see that the total derivative is:

$$d_{p_0}F = d_{p_0}L = \int_{0}^{T}[\partial_{p_0}f - \lambda^{\top}\partial_{p_0}h]dt - m \dot \lambda(0)^{\top}[\partial_{u(0)}g]^{-1}\partial_{p_0}g + m\lambda(0)^{\top}[\partial _{\dot u(0)}k]^{-1} \partial_{p_0}k$$

Where the notation $\partial_{p_0}f$ means the Jacobian of $f$ with respect to $p_0$ and $[\cdot]^{-1}$ is the matrix inverse. Lets simplify the expression by noting the following equalities:

```math
\begin{aligned}
\partial_{p_0}f &= 0 \\
\partial_{p_0}h &= 0 \\
[\partial_{u(0)}g]^{-1} &= I \\
\partial_{p_0}g  &= -I \\
\partial_{p_0}k &= 0
\end{aligned}
```

giving the total derivative:

$$d_{p_0}F = - m \dot \lambda(0)^{\top}$$

where $\lambda$ is the adjoint wavefield given by the solution of:

$$\ddot \lambda = \partial_{u}h^{\top}\lambda  -\partial_{u}f^{\top}  \, \, \rightarrow \, \, 
m \ddot \lambda -\nabla^2\lambda = - \partial_{u}f^{\top} =  \sum_{r=1}^{R} \int_{0}^{T}(u_{r}(t) - d_{r}(t)dt.$$

Note that this is the same wave equation as the forward model but with a source term given by the residual $-\partial_{u}f^{\top}$and it is solved in reverse time (from $t=T$ to $t=0$) with the "initial" state:

```math
\begin{aligned}
 \lambda(T)^{\top} &= 0 \\
\dot \lambda(T)^{\top} &= 0
\end{aligned}
```

## Jacobian derivation

While conventional photoacoustic imaging focuses on the initial state $p_0$ and consider the background velocity to be fixed (and in some case constant) we are interested in the generalize problem taking into account the sensitivity with respect to the physical parameter $m$. By doing though we are capable of inverting for both the photacoustic source and for the spatial variation of the propagation speed (heterogenity of the domain).

We derive the sensitivity with repect to the square slowness in a similar fation except we consider the total derivative of the Lagrangian with respect to $m$. Following the same steps we arrive at the familiar adjoint state expression of the gradient:


```math
    d_{m}F = - \int_{0}^{T} \lambda^{\top} \ddot u
```

where $\lambda$ is the solution of the same adjoint equation derived for the sensitivity with respect to the photoacoustic source.