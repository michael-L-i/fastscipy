"""scipy.integrate.trapezoid for plain float64 ndarrays, without the array-API
dispatch around the same NumPy expression. (``quad`` with a Python callable is
dominated by the callbacks; its wrapper overhead is below the cost of hooking it.)"""

import numpy as np
from scipy import integrate
from scipy.integrate import _quadrature

from . import _patch


def _trapezoid(y, x=None, dx=1.0, axis=-1):
    if (
        type(y) is not np.ndarray
        or y.dtype != np.float64
        or y.size == 0
        or type(axis) is not int
        or type(dx) not in (float, int)
        or not -y.ndim <= axis < y.ndim
    ):
        return NotImplemented
    if (
        y.ndim == 1
        and axis in (0, -1)
        and (x is None or (type(x) is np.ndarray and x.dtype == np.float64 and x.ndim == 1))
    ):
        d = dx if x is None else x[1:] - x[:-1]
        return np.add.reduce(d * (y[1:] + y[:-1]) / 2.0, axis=axis, dtype=np.float64)
    nd = y.ndim
    s1, s2 = [slice(None)] * nd, [slice(None)] * nd
    s1[axis], s2[axis] = slice(1, None), slice(None, -1)
    if x is None:
        d = dx
    else:
        if type(x) is not np.ndarray or x.dtype != np.float64 or x.ndim != 1:
            return NotImplemented
        d = x[1:] - x[:-1]
        s3 = [None] * nd
        s3[axis] = slice(None)
        d = d[tuple(s3)]
    return np.add.reduce(d * (y[tuple(s1)] + y[tuple(s2)]) / 2.0, axis=axis, dtype=np.float64)


def patch():
    _patch.require_validated("1.18.")
    old = _quadrature.trapezoid
    _patch.rebind(old, _patch.wrap(old, _trapezoid))
    assert integrate.trapezoid is not old


def smoke():
    y = np.linspace(0.0, 1.0, 20)
    integrate.trapezoid(y)
    integrate.trapezoid(y, y * y)
