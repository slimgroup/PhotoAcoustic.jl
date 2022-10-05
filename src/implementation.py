from devito import Operator, Function, Eq
from devito.tools import as_tuple

import numpy as np

from propagators import forward, adjoint, gradient, born

def forward_photo(model, rcv_coords, init_dist, nt, **kwargs):
    """
    Forward photoacoustic propagator. Propagates and intitial state init_dist
    conssisting of the first two time steps (u(t=0)=f and u.dt(t=0)=g)
    """
    kwargs['return_op'] = True
    op, u, rcv, kw = forward(model, None, rcv_coords, np.zeros((nt,)), **kwargs)

    # Set the first two entries of wavefield to spatial distribution init_dist
    u.data[0] = np.array(init_dist)
    u.data[1] = np.array(init_dist)

    op(**kw)

    return rcv.data


def adjoint_photo(model, y, rcv_coords, **kwargs):
    """
    Adjoint photoacoustic propagator.
    """
    rcv, v, summary = adjoint(model, y, None, rcv_coords, **kwargs)

    # Extract time derivative at 0. "Safer" like thhis than directly working
    # on data with numpy
    init = Function(name="ini", grid=model.grid, space_order=0)
    # Correct for default scaling in injection
    mrm = model.m * model.irho
    op = Operator(Eq(init, -mrm * v.dt.subs({v.indices[0]: 0})))
    op(dt=model.critical_dt)

    return init.data