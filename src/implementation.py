from devito import Operator, Function, Eq
from devito.tools import as_tuple

import numpy as np

import interface
from propagators import forward, adjoint, gradient, born
from sources import PointSource


def forwardis(model, rcv_coords, init_dist, nt, **kwargs):
    """
    Forward photoacoustic propagator. Propagates and intitial state init_dist
    consisting of the first two time steps (u(t=0)=f and u.dt(t=0)=0)
    """
    return_op = kwargs.get('return_op', False)
    kwargs['return_op'] = True
    op, u, rcv, kw = forward(model, None, rcv_coords, np.zeros((nt,)), **kwargs)

    # Set the first two entries of wavefield to spatial distribution init_dist
    mrm = model.m * model.irho
    u.data[0] = np.array(init_dist)
    u.data[1] = np.array(init_dist)
    Operator(Eq(u, u/mrm))(time_m=0, time_M=1)


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
    mrm = model.m * model.irho
    u.data[0] = np.array(init_dist)
    u.data[1] = np.array(init_dist)
    Operator([Eq(u, u/mrm), Eq(kw['ul'], u))(time_m=0, time_M=1)

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
    nt = y.shape[0]
    rcv, v, summary = adjoint(model, y, None, rcv_coords, **kwargs)

    # Extract time derivative at 0. "Safer" like thhis than directly working
    # on data with numpy
    init = Function(name="ini", grid=model.grid, space_order=0)

    # Correct for default scaling in injection
    op = Operator(Eq(init, -v.dt.subs({v.indices[0]: 0})))
    op(dt=model.critical_dt)

    return init.data


def adjointbornis(model, y, rcv_coords, init_dist, **kwargs):
    """
    Adjoint photoacoustic propagator.
    """
    func = bornis if kwargs.get('born_fwd', False) else forwardis

    nt = y.shape[0]

    fwdis = lambda *ar, **kw: func(ar[0], ar[2], init_dist, nt, **kw)
    kwargs['born_fwd'] = "fwdis"
    interface.op_fwd_J["fwdis"] = fwdis

    return interface.J_adjoint(model, None, np.zeros((nt,)), rcv_coords, y, **kwargs)
