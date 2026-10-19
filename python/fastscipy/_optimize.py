"""Finite-difference gradients (``approx_fprime``, and the numerical Jacobians of
``minimize``, ``least_squares``...) call the user function through ``_Fun_Wrapper``,
which re-derives the array namespace and copies the point on every call. For a
float64 NumPy point that is only overhead, so the wrapper skips it. The point is still
copied when it is ``x0`` itself, as the user function may modify its argument."""

import numpy as np
from scipy.optimize import _numdiff

from . import _patch


def patch():
    _patch.require_validated("1.18.")
    original = _numdiff._Fun_Wrapper.__call__

    # Not _patch.wrap: a RuntimeError from the user's function must not trigger a second call.
    def call(self, x):
        x0 = self.x0
        if (
            not _patch.enabled
            or type(x) is not np.ndarray
            or x is x0
            or type(x0) is not np.ndarray
            or x.dtype != np.float64
            or x0.dtype != np.float64
        ):
            return original(self, x)
        _patch.counts["_Fun_Wrapper", "fast"] += 1
        f = np.atleast_1d(self.fun(x, *self.args, **self.kwargs))
        if f.ndim > 1:
            raise RuntimeError("`fun` return value has more than 1 dimension.")
        return f

    _patch.replace(_numdiff._Fun_Wrapper, "__call__", call)


def smoke():
    from scipy import optimize

    optimize.approx_fprime(np.ones(5), optimize.rosen, 1e-8)
