"""Algorithmic shortcuts under scipy.stats: discrete distributions on big arrays with few
distinct values, ``mode`` by ``bincount``, and ``gaussian_kde`` evaluated on several threads."""
import numpy as np
import scipy.stats as stats
import scipy.stats._kde as _kde
import scipy.stats._stats_py as sp
from scipy.stats import _discrete_distns, rv_discrete

from . import _patch, _threads

MIN_N = 1000          # arrays below this are not worth a table
RANGE_FACTOR = 0.5    # integer table when its range <= n * factor
DUP_RATIO = 0.3       # np.unique table when a 1024-sample is mostly duplicates
KDE_WORK = 2_000_000  # kernel evaluations before threads pay
EPS = np.finfo(np.float64).eps


def _scalar_params(args, kwds):
    return all(isinstance(v, (bool, int, float, np.number)) or (isinstance(v, np.ndarray) and v.ndim == 0)
               for v in (*args, *kwds.values()))


def _eligible(self, a, args, kwds, dtypes=()):
    return (type(self).__module__ == _discrete_distns.__name__ and type(a) is np.ndarray and a.size >= MIN_N
            and (not dtypes or a.dtype in dtypes) and _scalar_params(args, kwds))


def _pointwise(orig):
    """Evaluate once per distinct k and gather (pointwise, so identical)."""
    def fast(self, k, *args, **kwds):
        if not _eligible(self, k, args, kwds, (np.dtype("int64"), np.dtype("int32"), np.dtype("float64"))):
            return NotImplemented
        n = k.size
        if k.dtype.kind == "i":
            lo, hi = int(k.min()), int(k.max())
            if hi - lo + 1 > n * RANGE_FACTOR:
                return NotImplemented
            idx = k.astype(np.int64, copy=False) - lo
            tab = orig(self, np.arange(lo, hi + 1, dtype=k.dtype), *args, **kwds)
        else:
            sample = k.ravel()[::max(1, n // 1024)]
            if np.unique(sample).size > DUP_RATIO * sample.size:
                return NotImplemented
            u, idx = np.unique(k.ravel(), return_inverse=True)
            tab = orig(self, u, *args, **kwds)
        out = tab[idx.reshape(k.shape)]
        return out if out.shape == k.shape else NotImplemented
    return fast


def _ppf(orig):
    """A cdf table over the needed range plus ``searchsorted``. Quantiles within a tolerance of a
    table value, or outside it, are left to SciPy's own ppf so its rounding decides ties."""
    def fast(self, q, *args, **kwds):
        if not (_eligible(self, q, args, kwds, (np.dtype("float64"),)) and kwds.get("loc", 0) == 0
                and not any(args[self.numargs:self.numargs + 1])):
            return NotImplemented
        qf, n = q.ravel(), q.size
        if not (qf.min() > 0 and qf.max() < 1):             # also rejects NaN
            return NotImplemented
        ext = orig(self, np.array([qf.min(), qf.max()]), *args, **kwds)
        if not np.isfinite(ext).all():
            return NotImplemented
        k0, k1 = int(ext[0]) - 1, int(ext[1]) + 1
        if k1 - k0 + 1 > n * RANGE_FACTOR or k1 - k0 > 10_000_000:
            return NotImplemented
        ks = np.arange(k0, k1 + 1, dtype=np.int64)
        cdfs = self.cdf(ks, *args, **kwds)
        if cdfs.shape != ks.shape or np.isnan(cdfs).any() or not (np.diff(cdfs) >= 0).all():
            return NotImplemented
        m = ks.size
        idx = np.searchsorted(cdfs, qf, side="left")
        ic, im = np.minimum(idx, m - 1), np.maximum(idx - 1, 0)
        tol = 1e-7 * np.minimum(qf, 1.0 - qf) + 16 * EPS
        flag = (idx >= m) | (idx == 0) | (np.abs(qf - cdfs[ic]) <= tol) | (np.abs(qf - cdfs[im]) <= tol)
        out = ks[ic].astype(np.float64)
        if flag.any():
            out[flag] = orig(self, qf[flag], *args, **kwds)
        return out.reshape(q.shape)
    return fast


_MODE_INTS = tuple(np.dtype(t) for t in (np.int64, np.int32, np.int16, np.int8, np.uint8))


def _mode(orig):
    def fast(a, axis=0, nan_policy="propagate", keepdims=False):
        if not (type(a) is np.ndarray and a.ndim == 1 and a.size >= 256 and a.dtype in _MODE_INTS
                and axis in (0, -1, None) and keepdims is False and nan_policy == "propagate"):
            return NotImplemented
        lo, hi = int(a.min()), int(a.max())
        if hi - lo + 1 > 4 * a.size + 1024:
            return NotImplemented
        cnt = np.bincount(a.astype(np.intp, copy=False) - lo)
        i = int(cnt.argmax())
        return sp.ModeResult(a.dtype.type(i + lo), cnt.dtype.type(cnt[i]))
    return fast


def _kde_evaluate(self, points):
    """``gaussian_kernel_estimate`` (it releases the GIL) on chunks of the points."""
    threads = _threads.get_num_threads()
    points = np.atleast_2d(np.asarray(points))
    d, m = points.shape
    if threads == 1 or m < 2 or d != self.d or m * self.n * d < KDE_WORK:
        return NotImplemented
    dtype, spec = _kde._get_output_dtype(self.covariance, points)
    fn = _kde.gaussian_kernel_estimate[spec]
    ds, w, pt = self.dataset.T, self.weights[:, None], points.T
    res = np.empty((m, 1), dtype=dtype)

    def run(s):
        res[s] = fn(ds, w, pt[s], self.cho_cov, dtype)
    _threads.parallel_map(run, _threads.chunks(m, 1000, threads))
    return res[:, 0]


def patch():
    for name in ("pmf", "logpmf", "cdf", "logcdf", "sf", "logsf"):
        old = rv_discrete.__dict__[name]
        _patch.replace(rv_discrete, name, _patch.wrap(old, _pointwise(old), f"discrete.{name}"))
    old = rv_discrete.__dict__["ppf"]
    _patch.replace(rv_discrete, "ppf", _patch.wrap(old, _ppf(old), "discrete.ppf"))
    cell = stats.mode.__closure__[stats.mode.__code__.co_freevars.index("hypotest_fun_in")]
    inner = cell.cell_contents
    _patch.replace(cell, "cell_contents", _patch.wrap(inner, _mode(inner), "mode"))
    old = stats.gaussian_kde.evaluate
    new = _patch.wrap(old, _kde_evaluate, "gaussian_kde.evaluate")
    _patch.replace(stats.gaussian_kde, "evaluate", new)
    _patch.replace(stats.gaussian_kde, "__call__", new)
