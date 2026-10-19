"""Dense unbounded linear least squares and Rosenbrock Hessians, plus numerical gradients.

The dense least-squares path uses SciPy's exact NumPy solve and result packing.
Rosenbrock Hessians allocate their three diagonals directly in one output matrix.

Finite-difference gradients (``approx_fprime``, and the numerical Jacobians of
``minimize``, ``least_squares``...) call the user function through ``_Fun_Wrapper``,
which re-derives the array namespace and copies the point on every call. For a
float64 NumPy point that is only overhead, so the wrapper skips it. The point is still
copied when it is ``x0`` itself, as the user function may modify its argument."""

import numpy as np
import scipy.optimize as optimize
from scipy.optimize import _numdiff
from scipy.optimize._lsq.lsq_linear import TERMINATION_MESSAGES, OptimizeResult

from . import _patch


def _dense(a, ndim):
    return type(a) is np.ndarray and a.dtype == np.float64 and a.ndim == ndim


def _rosen_hess_fast(x):
    if not (_dense(x, 1) and x.size >= 2 and np.abs(x).max() < 1e100):
        return NotImplemented
    n = x.size
    diagonal = np.empty(n)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = np.zeros((n, n))
    # SciPy adds a zero diagonal matrix to the off-diagonals, so -0 becomes +0.
    off = -400 * x[:-1] + 0.0
    H.flat[1 :: n + 1] = off
    H.flat[n :: n + 1] = off
    H.flat[:: n + 1] = diagonal
    return H


def _lsq_linear_fast(
    A,
    b,
    bounds=(-np.inf, np.inf),
    method="trf",
    tol=1e-10,
    lsq_solver=None,
    lsmr_tol=None,
    max_iter=None,
    verbose=0,
    *,
    lsmr_maxiter=None,
):
    if not (
        _dense(A, 2)
        and _dense(b, 1)
        and A.shape[0] == b.size
        and A.shape[1] > 0
        and type(bounds) is tuple
        and len(bounds) == 2
        and all(type(v) in (int, float, np.float64) for v in bounds)
        and bounds[0] == -np.inf
        and bounds[1] == np.inf
        and type(method) is str
        and method in ("trf", "bvls")
        and (lsq_solver is None or (type(lsq_solver) is str and lsq_solver == "exact"))
        and lsmr_tol is None
        and max_iter is None
        and type(verbose) is int
        and verbose == 0
        and lsmr_maxiter is None
        and np.isfinite(A).all()
        and np.isfinite(b).all()
    ):
        return NotImplemented
    solution = np.linalg.lstsq(A, b, rcond=-1)
    x = solution[0]
    if np.isnan(x).any():
        return NotImplemented  # SciPy's in_bounds check also rejects NaN solutions.
    r = A @ x - b
    cost = 0.5 * np.dot(r, r)
    optimality = np.linalg.norm(A.T.dot(r), ord=np.inf)
    return OptimizeResult(
        x=x,
        fun=r,
        cost=cost,
        optimality=optimality,
        active_mask=np.zeros(A.shape[1]),
        unbounded_sol=solution,
        nit=0,
        status=3,
        message=TERMINATION_MESSAGES[3],
        success=True,
    )


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
    for name, fast in (("rosen_hess", _rosen_hess_fast), ("lsq_linear", _lsq_linear_fast)):
        old = getattr(optimize, name)
        _patch.rebind(old, _patch.wrap(old, fast))


def smoke():
    from scipy import optimize

    optimize.approx_fprime(np.ones(5), optimize.rosen, 1e-8)
    A = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    b = np.array([1.0, 2.0, 1.0])
    optimize.rosen_hess(b), optimize.lsq_linear(A, b)
