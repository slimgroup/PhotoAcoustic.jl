from devito import Operator, Function, Eq, Inc, norm
from devito.tools import as_tuple

import numpy as np

import interface
from propagators import forward, adjoint, gradient, born


def forwardis(model, rcv_coords, init_dist, nt, **kwargs):
    """
    Forward photoacoustic propagator. Propagates and intitial state init_dist
    consisting of the first two time steps (u(t=0)=f and u.dt(t=0)=0)
    """
    return_op = kwargs.get('return_op', False)
    kwargs['return_op'] = True
    op, u, rcv, kw = forward(model, None, rcv_coords, np.zeros((nt,)), **kwargs)

    # Set the first two entries of wavefield to spatial distribution init_dist
    u.data[0, :] = init_dist[:]
    u.data[1, :] = init_dist[:]

    if return_op:
        return op, u, rcv, kw

    summary = op(**kw)

    # Illumination
    I = kw.get('Iu', None)

    # Check wich wavefield is wanted as output
    dft = kwargs.get('freq_list', None) is not None
    dft_modes = (kw['ufr%s' % u.name], kw['ufi%s' % u.name]) if dft else None
    us = kw['us_u'] if kwargs.get('t_sub', 0) > 1 else u
    # Reset initial condition in case we have a buffered `u` that needs to be reused
    us.data[0] = np.array(init_dist)
    return rcv, dft_modes or us, I, summary


def forwardis_data(*args, **kwargs):
    rcv, u, I, summary = forwardis(*args, **kwargs)
    return rcv.data, getattr(I, "data", None)


def bornis(model, rcv_coords, init_dist, nt, **kwargs):
    """
    Linearized forward photoacoustic propagator. Propagates and intitial state init_dist
    conssisting of the first two time steps (u(t=0)=f and u.dt(t=0)=0)
    """
    return_op = kwargs.get('return_op', False)
    kwargs['return_op'] = True
    op, u, rcv, kw = born(model, None, rcv_coords, np.zeros((nt,)), **kwargs)

    # Set the first two entries of wavefield to spatial distribution init_dist
    u.data[0, :] = init_dist[:]
    u.data[1, :] = init_dist[:]

    if return_op:
        return op, u, rcv, kw

    op(**kw)

    # Illumination
    I = kw.get('Iu', None)

    return rcv, u, I


def bornis_data(*args, **kwargs):
    rcv, u, I = bornis(*args, **kwargs)
    return rcv.data, getattr(I, "data", None)


def adjointis(model, y, rcv_coords, **kwargs):
    """
    Adjoint photoacoustic propagator.
    """
    kwargs.pop('checkpointing', None)
    kwargs.pop('t_sub', None)
    kwargs.pop('ic', None)
    # Make dt source
    rcv, v, I = adjoint(model, -y, None, rcv_coords, **kwargs)[:3]

    # Extract time derivative at 0.
    init = Function(name="ini", grid=model.grid, space_order=0)
    # Correct for default scaling in injection
    mrm = model.m * model.irho
    op = Operator(Eq(init, mrm * v.dt))
    op(dt=model.critical_dt, time_m=0, time_M=0)

    return init.data, getattr(I, "data", None)


def adjointbornis(model, y, rcv_coords, init_dist, checkpointing=None, freq_list=None,
                  t_sub=1, **kwargs):
    """
    Adjoint photoacoustic propagator.
    """
    nt = y.shape[0]
    born_fwd = kwargs.get('born_fwd', False)
    rec, u, Iu, _ = op_fwd_JIS[born_fwd](model, rcv_coords, init_dist, nt,
                                         save=freq_list is None, freq_list=freq_list,
                                         t_sub=t_sub, **kwargs)

    # Get operator
    kwargs['return_op'] = True
    op, g, kwg = gradient(model, y, rcv_coords, u, save=freq_list is None, freq=freq_list,
                          **kwargs)
    op(**kwg)
    Iv = kwg.get('Iv', None)
    # Need the intergation by part correction since we compute the gradient on
    # u * v.dt (see Documentation)
    if freq_list is None:
        w = model.irho if kwargs.get('ic', "as") == "as" else model.irho * model.m
        op0 = Operator(Eq(g, g -  w * kwg['v'].dt * u))
        op0(dt=model.critical_dt, time_m=0, time_M=0)
    
    return g.data, getattr(Iu, "data", None), getattr(Iv, "data", None)

op_fwd_JIS = {False: forwardis, True: bornis}
