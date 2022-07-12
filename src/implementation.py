from devito import Operator
from devito.tools import as_tuple

import numpy as np

from sources import PointSource
from geom_utils import src_rec
from fields import wavefield
from kernels import wave_kernel
from utils import opt_op


def forward_photo(model, rcv_coords, init_dist, nt, space_order=8):
    """
    Forward photoacoustic propagator. Propagates and intitial state init_dist
    conssisting of the first two time steps (u(t=0)=f and u.dt(t=0)=g)
    """
    # Setting forward wavefield
    u = wavefield(model, space_order, nt=nt)

    # Set the first two entries of wavefield to spatial distribution init_dist
    #u.data[0] = np.array(init_dist / (model.m * model.irho).data)
    #u.data[1] = np.array(init_dist / (model.m * model.irho).data)
    u.data[0] = np.array(init_dist)
    u.data[1] = np.array(init_dist)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u)

    # Setup receiver
    _, rcv = src_rec(model, u, nt=nt, rec_coords=rcv_coords)

    rec_expr = rcv.interpolate(expr=u)

    # Create operator and run
    op = Operator(pde + rec_expr,
                  subs=model.spacing_map, name="ISforward",
                  opt=opt_op(model))
    op.cfunction

    op()

    # Output
    return rcv.data


def adjoint_photo(model, y, rcv_coords, space_order=8):
    """
    Adjoint photoacoustic propagator.
    """
    # Number of time steps
    nt = y.shape[0]

    # Setting adjoint wavefield
    v = wavefield(model, space_order, nt=nt, fw=False)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, fw=False)

    # Inject -dt of source so that we directly solve for -v.dt
    dt = model.grid.time_dim.spacing
    namef = as_tuple(v)[0].name

    src = PointSource(name="src%s" % namef, grid=model.grid, ntime=nt,
                      coordinates=rcv_coords)
    src.data[:] = y[:]

    u_n = as_tuple(v)[0].backward
    geom_expr = src.inject(field=u_n, expr=-src.dt*dt**2 )

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr,
                  subs=subs, name="ISadjoint",
                  opt=opt_op(model))
    op.cfunction

    # Run operator
    op()

    # Adjoint math says that it is -dt of the adjoint variable at T=0
    # Source injection of -src.dt makes direct output of v the correct adjoint
    return v.data[0]
