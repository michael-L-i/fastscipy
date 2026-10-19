"""Sampled-data integration and Gaussian quadrature without generic NumPy dispatch.

(``quad`` with a Python callable is
dominated by the callbacks; its wrapper overhead is below the cost of hooking it.)"""

import functools
import math

import numpy as np
from scipy import integrate
from scipy._lib._array_api import array_namespace
from scipy.integrate import _quadrature

from . import _patch

_NP = array_namespace(np.empty(0))


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


def _cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    if (
        type(y) is not np.ndarray
        or y.dtype != np.float64
        or y.ndim != 1
        or not y.size
        or type(axis) is not int
        or axis not in (0, -1)
        or type(dx) not in (float, int)
        or (initial is not None and (type(initial) not in (float, int) or initial != 0))
    ):
        return NotImplemented
    if x is None:
        d = dx
    elif type(x) is np.ndarray and x.dtype == np.float64 and x.ndim == 1 and x.shape == y.shape:
        d = x[1:] - x[:-1]
    else:
        return NotImplemented
    out = np.cumsum(d * (y[1:] + y[:-1]) / 2.0, axis=axis)
    if initial is not None:
        out = np.concatenate((np.full(1, initial, dtype=out.dtype), out))
    return out


def _simpson(y, x=None, *, dx=1.0, axis=-1):
    """Uniform spacing and an odd sample count: SciPy's same composite rule."""
    if (
        type(y) is not np.ndarray
        or y.dtype != np.float64
        or x is not None
        or type(dx) not in (float, int)
        or type(axis) is not int
        or not -y.ndim <= axis < y.ndim
    ):
        return NotImplemented
    n = y.shape[axis]
    if n < 3 or n % 2 == 0:
        return NotImplemented
    s0, s1, s2 = ([slice(None)] * y.ndim for _ in range(3))
    s0[axis], s1[axis], s2[axis] = slice(0, n - 2, 2), slice(1, n - 1, 2), slice(2, n, 2)
    try:
        with np.errstate(all="raise"):
            spacing = np.asarray(dx, dtype=np.float64)
            return spacing / 3.0 * np.sum(y[tuple(s0)] + 4.0 * y[tuple(s1)] + y[tuple(s2)], axis=axis)
    except (FloatingPointError, OverflowError):
        return NotImplemented


def _fixed_quad(original):
    @functools.wraps(original)
    def fixed_quad(func, a, b, args=(), n=5):
        if (
            not _patch.enabled
            or type(a) not in (float, int)
            or type(b) not in (float, int)
            or not -(2**63) <= a < 2**63
            or not -(2**63) <= b < 2**63
            or not math.isfinite(a)
            or not math.isfinite(b)
            or type(args) is not tuple
            or type(n) is not int
            or not 1 <= n <= 1000
        ):
            return original(func, a, b, args, n)
        _patch.counts["fixed_quad", "fast"] += 1
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        x, w = _quadrature._cached_roots_legendre(n)
        x, w = np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)
        y = (b - a) * (np.real(x) + 1) / 2.0 + a
        # Call once and propagate any error: a generic fallback wrapper could
        # repeat a callback's side effects after its exception or a bad result.
        return (b - a) / 2.0 * _NP.sum(w * func(y, *args), axis=-1), None

    return fixed_quad


def patch():
    _patch.require_validated("1.18.")
    for name, fast in (
        ("trapezoid", _trapezoid),
        ("cumulative_trapezoid", _cumulative_trapezoid),
        ("simpson", _simpson),
    ):
        old = getattr(_quadrature, name)
        _patch.rebind(old, _patch.wrap(old, fast))
    old = _quadrature.fixed_quad
    _patch.rebind(old, _fixed_quad(old))


def smoke():
    y = np.linspace(0.0, 1.0, 20)
    integrate.trapezoid(y)
    integrate.trapezoid(y, y * y)
    integrate.cumulative_trapezoid(y, initial=0)
    integrate.simpson(y[:19])
    integrate.fixed_quad(np.cos, 0.0, 1.0)
