from devito import Operator, Function, Eq
from devito.tools import as_tuple

import numpy as np

import interface
from propagators import forward, adjoint, gradient, born
from sensitivity import l2_loss


def forwardis(model, rcv_coords, init_dist, nt, **kwargs):
    """
    Forward photoacoustic propagator. Propagates and intitial state init_dist
    consisting of the first two time steps (u(t=0)=f and u.dt(t=0)=0)
    """
    return_op = kwargs.get('return_op', False)
    kwargs['return_op'] = True
    op, u, rcv, kw = forward(model, None, rcv_coords, np.zeros((nt,)), **kwargs)

    # Set the first two entries of wavefield to spatial distribution init_dist
    u.data[0] = np.array(init_dist)
    u.data[1] = np.array(init_dist)

    if return_op:
        return op, u, rcv, kw

    summary = op(**kw)

    return rcv, u, summary


def forwardis_data(*args, **kwargs):
    return forwardis(*args, **kwargs)[0].data


def bornis(model, rcv_coords, init_dist, nt, **kwargs):
    """
    Linearized forward photoacoustic propagator. Propagates and intitial state init_dist
    conssisting of the first two time steps (u(t=0)=f and u.dt(t=0)=0)
    """
    return_op = kwargs.get('return_op', False)
    kwargs['return_op'] = True
    op, u, rcv, kw = born(model, None, rcv_coords, np.zeros((nt,)), **kwargs)

    # Set the first two entries of wavefield to spatial distribution init_dist
    u.data[0] = np.array(init_dist)
    u.data[1] = np.array(init_dist)

    if return_op:
        return op, u, rcv, kw

    summary = op(**kw)

    return rcv, u, summary


def bornis_data(*args, **kwargs):
    return bornis(*args, **kwargs)[0].data


def adjointis(model, y, rcv_coords, **kwargs):  
    """
    Adjoint photoacoustic propagator.
    """
    # Make dt source
    rcv, v, summary = adjoint(model, -y, None, rcv_coords, **kwargs)

    # Extract time derivative at 0.
    init = Function(name="ini", grid=model.grid, space_order=0)
    # Correct for default scaling in injection
    mrm = model.m * model.irho
    op = Operator(Eq(init, mrm * v.dt))
    op(dt=model.critical_dt, time_m=0, time_M=0)

    return init.data


def adjointbornis(model, y, rcv_coords, init_dist, **kwargs):
    """
    Adjoint photoacoustic propagator.
    """
    nt = y.shape[0]
    born_fwd = kwargs.get('born_fwd', False)
    rec, u, _ = op_fwd_JIS[born_fwd](model, rcv_coords, init_dist, nt, **kwargs)

    kwargs['return_op'] = True
    kwargs.pop('freq_list', None)
    op, g, kwg = gradient(model, -y, rcv_coords, u, **kwargs)
    op(**kwg)

    # Need the extra v[0].dt * m
    # v = kwg['v']
    # Correct for default scaling in injection
    # mrm = model.m * model.irho
    # from IPython import embed; embed()
    # op = Operator(Eq(g, g / mrm))
    # op(dt=model.critical_dt, time_m=0, time_M=0)

    return g.data

op_fwd_JIS = {False: forwardis, True: bornis}
