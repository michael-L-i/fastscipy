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
_SIMPSON_DTYPES = tuple(np.dtype(t) for t in ("float64", "float32", "complex128", "complex64"))


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
        or y.dtype not in (np.float64, np.float32, np.complex128)
        or not y.size
        or type(axis) is not int
        or not -y.ndim <= axis < y.ndim
        or type(dx) not in (float, int)
        or (initial is not None and (type(initial) not in (float, int) or initial != 0))
    ):
        return NotImplemented
    if x is None:
        d = dx
    elif type(x) is np.ndarray and x.dtype in (np.float64, np.float32):
        if x.ndim == 1 and x.size == y.shape[axis]:
            d = x[1:] - x[:-1]
            if y.ndim > 1:
                shape = [1] * y.ndim
                shape[axis] = -1
                d = d.reshape(shape)
        elif x.shape == y.shape:
            d = np.diff(x, axis=axis)
        else:
            return NotImplemented
    else:
        return NotImplemented
    if y.ndim == 1:
        out = np.cumsum(d * (y[1:] + y[:-1]) / 2.0, axis=axis)
    else:
        first, second = [slice(None)] * y.ndim, [slice(None)] * y.ndim
        first[axis], second[axis] = slice(1, None), slice(None, -1)
        out = np.cumsum(d * (y[tuple(first)] + y[tuple(second)]) / 2.0, axis=axis)
    if initial is not None:
        if y.ndim == 1:
            out = np.concatenate((np.full(1, initial, dtype=out.dtype), out))
        else:
            shape = list(out.shape)
            shape[axis] = 1
            out = np.concatenate((np.full(shape, initial, dtype=out.dtype), out), axis=axis)
    return out


def _simpson(y, x=None, *, dx=1.0, axis=-1):
    """SciPy's composite rule and end correction with concrete NumPy arithmetic."""
    if (
        type(y) is not np.ndarray
        or y.dtype not in _SIMPSON_DTYPES
        or type(dx) not in (float, int)
        or (type(dx) is int and not -(2**63) <= dx < 2**64)
        or type(axis) is not int
        or not -y.ndim <= axis < y.ndim
    ):
        return NotImplemented
    n = y.shape[axis]
    if n < 2:
        return NotImplemented
    s0, s1, s2 = ([slice(None)] * y.ndim for _ in range(3))
    stop = n - 2 if n % 2 else n - 3
    s0[axis], s1[axis], s2[axis] = slice(0, stop, 2), slice(1, stop + 1, 2), slice(2, stop + 2, 2)
    try:
        with np.errstate(all="raise"):
            if x is None:
                spacing = np.asarray(dx, dtype=y.dtype)
                differences = None
            else:
                if not (
                    y.size and y.dtype.kind == "f" and type(x) is np.ndarray and x.dtype in (np.float64, np.float32)
                ):
                    return NotImplemented
                dtype = np.result_type(y.dtype, x.dtype)
                y, coords = y.astype(dtype, copy=False), x.astype(dtype, copy=False)
                if coords.ndim == 1 and coords.size == n:
                    shape = [1] * y.ndim
                    shape[axis] = n
                    coords = coords.reshape(shape)
                elif coords.shape != y.shape:
                    return NotImplemented
                differences = np.diff(coords, axis=axis)
                low, high = differences.min(initial=np.inf), differences.max(initial=-np.inf)
                if not ((low >= 1e-20 and high <= 1e20) or (low >= -1e20 and high <= -1e-20)):
                    return NotImplemented
                spacing = None
            if n == 2:
                s1[axis], s2[axis] = -1, -2
                last_dx = spacing if x is None else np.squeeze(differences, axis=axis)
                return 0.0 + 0.5 * last_dx * (y[tuple(s1)] + y[tuple(s2)])
            if x is None:
                result = spacing / 3.0 * np.sum(y[tuple(s0)] + 4.0 * y[tuple(s1)] + y[tuple(s2)], axis=axis)
            else:
                h0, h1 = differences[tuple(s0)], differences[tuple(s1)]
                hsum, hprod, ratio = h0 + h1, h0 * h1, h0 / h1
                tmp = (
                    hsum
                    / 6.0
                    * (
                        y[tuple(s0)] * (2.0 - 1.0 / ratio)
                        + y[tuple(s1)] * (hsum * (hsum / hprod))
                        + y[tuple(s2)] * (2.0 - ratio)
                    )
                )
                result = np.sum(tmp, axis=axis)
            if n % 2:
                return result
            if x is None:
                h0, h1 = np.stack((spacing, spacing))
            else:
                s0[axis], s1[axis] = slice(-2, -1), slice(-1, None)
                h0, h1 = np.squeeze(differences[tuple(s0)], axis=axis), np.squeeze(differences[tuple(s1)], axis=axis)
            num, den = 2 * h1**2 + 3 * h0 * h1, 6 * (h1 + h0)
            alpha = np.asarray(num / den if x is not None or den != 0 else np.zeros((), dtype=y.dtype))
            num, den = h1**2 + 3.0 * h0 * h1, 6 * h0
            beta = np.asarray(num / den if x is not None or den != 0 else np.zeros((), dtype=y.dtype))
            num, den = 1 * h1**3, 6 * h0 * (h0 + h1)
            eta = np.asarray(num / den if x is not None or den != 0 else np.zeros((), dtype=y.dtype))
            s0[axis], s1[axis], s2[axis] = -1, -2, -3
            result += alpha * y[tuple(s0)] + beta * y[tuple(s1)] - eta * y[tuple(s2)]
            result += 0.0
            return result
    except (FloatingPointError, OverflowError):
        return NotImplemented


def _cumulative_trapezoid_wrapper(original):
    """Accepted arithmetic may report warnings before an error; never retry it."""

    @functools.wraps(original)
    def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
        if _patch.enabled:
            result = _cumulative_trapezoid(y, x, dx, axis, initial)
            if result is not NotImplemented:
                _patch.counts["cumulative_trapezoid", "fast"] += 1
                return result
        _patch.counts["cumulative_trapezoid", "scipy"] += 1
        return original(y, x, dx, axis, initial)

    cumulative_trapezoid.fast = _cumulative_trapezoid
    return cumulative_trapezoid


def _cumulative_simpson(y, *, x=None, dx=1.0, axis=-1, initial=None):
    """Evaluate only the interval estimates retained by SciPy's interleaving."""
    # Bounds cover arithmetic in discarded estimates too, which must not emit
    # warnings that the selected estimates would miss. Underflow reporting is
    # left entirely to SciPy because skipped calculations may underflow.
    if not (
        type(y) is np.ndarray
        and y.dtype == np.float64
        and type(axis) is int
        and -y.ndim <= axis < y.ndim
        and y.shape[axis] >= 3
        and type(dx) in (float, int)
        and (type(dx) is not int or -(2**63) <= dx < 2**64)
        and (initial is None or type(initial) in (float, int))
        and -1e100 < dx < 1e100
        and (initial is None or -1e100 < initial < 1e100)
        and np.geterr()["under"] == "ignore"
        and np.max(y, initial=0) < 1e100
        and np.min(y, initial=0) > -1e100
    ):
        return NotImplemented
    values = np.swapaxes(y, axis, -1)
    n = values.shape[-1]
    try:
        with np.errstate(all="raise"):
            if x is None:
                spacing = np.asarray(dx, dtype=np.float64)
            elif type(x) is np.ndarray and x.dtype == np.float64:
                if x.ndim == 1 and x.size == n:
                    spacing = np.diff(x)
                elif x.shape == y.shape:
                    spacing = np.diff(np.swapaxes(x, axis, -1), axis=-1)
                else:
                    return NotImplemented
                # Besides monotonicity, these bounds keep arithmetic in the
                # discarded unequal-interval estimates finite too.
                if not (np.min(spacing, initial=np.inf) >= 1e-20 and np.max(spacing, initial=0) <= 1e20):
                    return NotImplemented
            else:
                return NotImplemented
            sub = np.empty(values.shape[:-1] + (n - 1,))
            f1, f2, f3 = values[..., :-2:2], values[..., 1:-1:2], values[..., 2::2]
            if x is None:
                sub[..., :-1:2] = spacing / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4)
                sub[..., 1::2] = spacing / 3 * (5 * f3 / 4 + 2 * f2 - f1 / 4)
                if n % 2 == 0:
                    last = values[..., -3:]
                    sub[..., -1:] = spacing / 3 * (5 * last[..., 2:] / 4 + 2 * last[..., 1:2] - last[..., :1] / 4)
            else:
                d1, d2 = spacing[..., :-1:2], spacing[..., 1::2]
                sub[..., :-1:2] = _unequal_interval(f1, f2, f3, d1, d2)
                sub[..., 1::2] = _unequal_interval(f3, f2, f1, d2, d1)
                if n % 2 == 0:
                    sub[..., -1:] = _unequal_interval(
                        values[..., -1:], values[..., -2:-1], values[..., -3:-2], spacing[..., -1:], spacing[..., -2:-1]
                    )
            out = np.cumsum(sub, axis=-1)
            if initial is not None:
                start = np.full(values.shape[:-1] + (1,), initial, dtype=np.float64)
                out += start
                out = np.concatenate((start, out), axis=-1)
    except (FloatingPointError, OverflowError):
        return NotImplemented
    return np.swapaxes(out, -1, axis)


def _unequal_interval(f1, f2, f3, x21, x32):
    """SciPy's unequal-interval formula, with its exact operation order."""
    x31 = x21 + x32
    x21_x31 = x21 / x31
    x21_x32 = x21 / x32
    product = x21_x31 * x21_x32
    coeff1, coeff2, coeff3 = 3 - x21_x31, 3 + product + x21_x31, -product
    return x21 / 6 * (coeff1 * f1 + coeff2 * f2 + coeff3 * f3)


def _romb(y, dx=1.0, axis=-1, show=False):
    """Two rows of the same Richardson table instead of a dictionary of all rows."""
    if not (
        type(y) is np.ndarray
        and y.dtype in (np.float64, np.float32, np.complex128)
        and type(axis) is int
        and -y.ndim <= axis < y.ndim
        and type(dx) in (float, int)
        and show is False
    ):
        return NotImplemented
    n = y.shape[axis] - 1
    if n < 1 or n & (n - 1):
        return NotImplemented
    selectors = [slice(None)] * y.ndim
    selectors[axis] = 0
    first = y[tuple(selectors)]
    selectors[axis] = -1
    last = y[tuple(selectors)]
    try:
        with np.errstate(all="raise"):
            h = n * np.asarray(dx, dtype=np.float64)
            previous = [(first + last) / 2.0 * h]
            start = step = n
            for i in range(1, n.bit_length()):
                start >>= 1
                selectors[axis] = slice(start, n, step)
                step >>= 1
                current = [0.5 * (previous[0] + h * np.sum(y[tuple(selectors)], axis=axis))]
                for j in range(1, i + 1):
                    prev = current[j - 1]
                    current.append(prev + (prev - previous[j - 1]) / ((1 << (2 * j)) - 1))
                previous = current
                h /= 2.0
    except (FloatingPointError, OverflowError):
        return NotImplemented
    return previous[-1]


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
        ("simpson", _simpson),
        ("cumulative_simpson", _cumulative_simpson),
        ("romb", _romb),
    ):
        old = getattr(_quadrature, name)
        _patch.rebind(old, _patch.wrap(old, fast))
    old = _quadrature.cumulative_trapezoid
    _patch.rebind(old, _cumulative_trapezoid_wrapper(old))
    old = _quadrature.fixed_quad
    _patch.rebind(old, _fixed_quad(old))


def smoke():
    y = np.linspace(0.0, 1.0, 20)
    integrate.trapezoid(y)
    integrate.trapezoid(y, y * y)
    integrate.cumulative_trapezoid(y, initial=0)
    integrate.simpson(y[:19])
    integrate.simpson(y)
    integrate.simpson(y, x=y * y)
    integrate.simpson(y.astype(np.float32))
    integrate.simpson(y.astype(np.complex64))
    integrate.cumulative_simpson(y)
    integrate.cumulative_simpson(y, x=y * y)
    integrate.romb(y[:17])
    integrate.romb(y[:17].astype(np.float32))
    integrate.fixed_quad(np.cos, 0.0, 1.0)
