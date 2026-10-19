"""scipy.interpolate lookups that are slow for random queries, plus ``Delaunay.find_simplex``.

SciPy locates each query by walking from the previous query's knot interval or
simplex, so random queries pay for long walks. Spline intervals depend only on
the query, so they are found by binary search (Rust) or by sorting the queries
around SciPy's own call. Simplices can depend on the walk's start when a query
lies on a shared face, so queries are visited in grid-cell order and the
queries near a face are replayed in their original order (``_locate``).
"""

import numpy as np

from . import _native, _patch, _threads

_k = _native.interp
_orig = {}

# Break-evens measured on one thread against SciPy 1.18.1 (Apple M-series).
SPLINE_MIN_QUERIES = 300
SPLINE_MIN_JUMP = 100  # mean knot-index jump between consecutive queries (SciPy's walk length)
KERNEL_PARALLEL_FROM = 1 << 16  # with threads the kernel also pays on sorted queries
SIMPLEX_MIN_QUERIES = 300
SIMPLEX_MIN_WORK = 5e4  # queries * sqrt(simplices): the walk length
SAMPLE = 1024  # consecutive query pairs sampled to judge the query order
LOCAL_JUMP = 4.0  # mean jump between consecutive queries, in grid cells, below which SciPy wins
TIE_BAND = 1e-9  # barycentric distance to a face that triggers a replay
MAX_REPLAY = 0.25  # replay more than this fraction of the queries: use SciPy
MAX_REPLAY_RANGES = 20000
CHUNK = 4000  # smallest per-thread find_simplex chunk


def _is_array(a, kinds):
    return type(a) is np.ndarray and a.dtype.kind in kinds


def _jump(x, t):
    """Mean knot-index jump between consecutive queries, from up to SAMPLE pairs."""
    i = np.arange(len(x) - 1) if len(x) <= SAMPLE + 1 else np.linspace(0, len(x) - 2, SAMPLE).astype(np.intp)
    return np.abs(np.searchsorted(t, x[i + 1]) - np.searchsorted(t, x[i])).mean()


def _on_axis(out, nd, axis):
    """Move the ``nd`` query axes of ``out`` (leading) to ``axis``, as BSpline does."""
    if not axis:
        return out
    perm = list(range(out.ndim))
    return out.transpose(perm[nd : nd + axis] + perm[:nd] + perm[nd + axis :])


def _spline(flat, shape, t, c, k, kernel_ok, axis, call):
    """Evaluate at the flat float64 queries by the Rust kernel, or by ``call`` on sorted queries.
    ``c`` has the coefficient axis first; the kernel gets it as (n, columns) float64."""
    n = len(flat)
    if n < SPLINE_MIN_QUERIES:
        return NotImplemented
    parallel = kernel_ok and n >= KERNEL_PARALLEL_FROM and _threads.get_num_threads() > 1
    # A searchsorted index is in [0, len(t)], so short knot vectors cannot
    # have a mean jump large enough to justify reordering.
    if len(t) < SPLINE_MIN_JUMP and not parallel:
        return NotImplemented
    jump = _jump(flat, t)
    if kernel_ok and (jump >= SPLINE_MIN_JUMP or parallel):
        cc = c.reshape(len(c), -1)
        out = _k.bspline_eval(t, cc.view(np.float64) if c.dtype.kind == "c" else cc, k, flat)
        if out is not None:
            out = out.view(complex) if c.dtype.kind == "c" else out
            return _on_axis(out.reshape(shape + c.shape[1:]), len(shape), axis)
    if not jump >= SPLINE_MIN_JUMP:
        return NotImplemented
    order = np.argsort(flat)
    r = np.moveaxis(np.asarray(call(flat[order])), axis, 0)
    pre = np.empty(r.shape, r.dtype)
    pre[order] = r
    return _on_axis(pre.reshape(shape + pre.shape[1:]), len(shape), axis)


def _bspline_call(self, x, nu=0, extrapolate=None):
    d = self._delegate_to
    if type(d) is not _orig["numpy_bspline"] or not isinstance(x, np.ndarray) or x.dtype.kind not in "fiu":
        return NotImplemented
    ext = d.extrapolate if extrapolate is None else extrapolate
    t, c, k = d.t, d.c, d.k
    if not isinstance(ext, (bool, np.bool_)) or not _is_array(t, "f") or not _is_array(c, "fc"):
        return NotImplemented
    kernel_ok = (
        nu == 0
        and ext
        and c.dtype in (np.float64, np.complex128)
        and t.dtype == np.float64
        and t.flags.c_contiguous
        and c.flags.c_contiguous
    )
    flat = np.ascontiguousarray(x.reshape(-1), dtype=np.float64)
    return _spline(flat, x.shape, t, c, k, kernel_ok, d.axis, lambda xs: _orig["bspline"](self, xs, nu, extrapolate))


def _splev(x, tck, der=0, ext=0):
    if not (isinstance(tck, (tuple, list)) and len(tck) == 3 and all(isinstance(a, np.ndarray) for a in tck[:2])):
        return NotImplemented
    (t, c, k), x = tck, np.asarray(x)
    if (
        type(k) is not int
        or not 1 <= k <= 5
        or t.ndim != 1
        or c.ndim != 1
        or len(t) <= k + 1
        or x.dtype.kind not in "fiu"
        or t.dtype.kind != "f"
        or c.dtype.kind != "f"
    ):
        return NotImplemented
    kernel_ok = (
        der == 0 and ext == 0 and c.dtype == t.dtype == np.float64 and t.flags.c_contiguous and c.flags.c_contiguous
    )
    flat = np.ascontiguousarray(x.reshape(-1), dtype=np.float64)
    return _spline(flat, x.shape, t, c, k, kernel_ok, 0, lambda xs: _orig["splev"](xs, tck, der, ext))


# ---------------------------------------------------------------- Delaunay
def _grid(tri):
    g = max(int(np.ceil(max(tri.nsimplex // 2, 1) ** (1 / tri.ndim))), 2)
    lo = np.asarray(tri.min_bound, float)
    hi = np.asarray(tri.max_bound, float)
    return g, lo, np.where(hi > lo, hi - lo, 1.0)


def _eligible(tri, x):
    return len(x) >= SIMPLEX_MIN_QUERIES and len(x) * tri.nsimplex**0.5 >= SIMPLEX_MIN_WORK


def _is_local(x, tri):
    """Consecutive queries already close (sorted or raster order): SciPy's walk is short."""
    g, lo, span = _grid(tri)
    i = np.linspace(0, len(x) - 2, SAMPLE).astype(np.intp)
    return (np.abs(x[i + 1] - x[i]) / span * g).max(axis=1).mean() < LOCAL_JUMP  # NaN is not local


def _locate(tri, x):
    """``(simplex per query, order, ranges)`` or None if too many queries are ties.

    The simplex is exact except inside ``ranges`` (inclusive index ranges in
    the original order), which the caller must recompute sequentially.
    """
    find = _orig["find_simplex"]
    n = len(x)
    g, lo, span = _grid(tri)
    order = _k.cell_order(x, lo, span, g)
    xs = x[order]
    _ = tri.transform  # lazily built (scipy#23854): materialise before threads
    res_s = np.empty(n, np.intc)
    flag_s = np.zeros(n, bool)

    def work(s):
        r = res_s[s] = find(tri, xs[s])
        out = np.flatnonzero((r < 0) & np.isfinite(xs[s]).all(axis=1))
        if len(out):  # outside the hull but within tolerance of a simplex
            flag_s[s.start + out[find(tri, xs[s][out], tol=TIE_BAND) >= 0]] = True

    _threads.parallel_map(work, _threads.chunks(n, CHUNK))
    _k.tie_flags(xs, res_s, tri.transform, TIE_BAND, flag_s)
    res = np.empty(n, np.intc)
    res[order] = res_s
    flagged = np.empty(n, bool)
    flagged[order] = flag_s
    ranges = _k.replay_intervals(res, flagged, int(MAX_REPLAY * n), MAX_REPLAY_RANGES)
    return None if ranges is None else (res, order, ranges)


def _find_simplex(self, xi, bruteforce=False, tol=None):
    xi = np.asanyarray(xi)
    if bruteforce or tol is not None or type(xi) is not np.ndarray or xi.dtype.kind not in "fiu":
        return NotImplemented
    if xi.ndim < 1 or xi.shape[-1] != self.ndim or xi.size == 0:
        return NotImplemented
    x = np.ascontiguousarray(xi.reshape(-1, self.ndim), dtype=np.double)
    if not _eligible(self, x) or _is_local(x, self):
        return NotImplemented
    loc = _locate(self, x)
    if loc is None:
        return NotImplemented
    res, _, ranges = loc
    for a, b in ranges:
        res[a : b + 1] = _orig["find_simplex"](self, x[a : b + 1])
    return res.reshape(xi.shape[:-1])


def _make_evaluate(cls, name):
    original = getattr(cls, name)
    linear = cls.__name__ == "LinearNDInterpolator"
    # CloughTocher2D's evaluate releases the GIL; Linear's (and both complex ones) do not.
    threaded = not linear and name == "_evaluate_double"

    def evaluate(self, xi, simplex_tolerance=1.0):
        tri = self.tri
        x = np.ascontiguousarray(xi, dtype=np.double)
        if simplex_tolerance != 1.0 or x.ndim != 2 or not _eligible(tri, x) or _is_local(x, tri):
            return NotImplemented
        loc = _locate(tri, x)
        if loc is None:
            return NotImplemented
        res, order, ranges = loc
        v = self.values
        if linear and name == "_evaluate_double" and _is_array(v, "f") and v.dtype == np.float64:
            for a, b in ranges:
                res[a : b + 1] = _orig["find_simplex"](tri, x[a : b + 1])
            return _k.linear_eval(res, tri.transform, tri.simplices, np.ascontiguousarray(v), x, float(self.fill_value))
        xs = x[order]
        slices = _threads.chunks(len(x), CHUNK) if threaded else [slice(0, len(x))]
        parts = _threads.parallel_map(lambda s: original(self, xs[s]), slices)
        first = np.asarray(parts[0])
        out = np.empty((len(x),) + first.shape[1:], dtype=first.dtype)
        # Scatter chunks directly: concatenating would copy the full output.
        for s, part in zip(slices, parts):
            out[order[s]] = part
        for a, b in ranges:
            out[a : b + 1] = original(self, x[a : b + 1])
        return out

    return original, evaluate


def _evaluate_spline(self, xi, method):
    g, v = self._grid, self._values
    if (
        method != "pchip"
        or not _is_array(xi, "f")
        or xi.ndim != 2
        or xi.dtype != np.float64
        or not len(xi)
        or not _is_array(v, "f")
        or v.dtype != np.float64
        or v.ndim != len(g)
        or not all(_is_array(a, "f") and a.dtype == np.float64 and a.ndim == 1 for a in g)
    ):
        return NotImplemented
    out = _k.pchip_eval(list(g), np.ascontiguousarray(v).reshape(-1), np.ascontiguousarray(xi))
    return NotImplemented if out is None else out


def patch():
    _patch.require_validated("1.18.")
    import scipy.interpolate as si
    from scipy.interpolate import _bsplines, _fitpack_py, _interpnd
    from scipy.spatial import Delaunay

    _orig.update(
        numpy_bspline=_bsplines._BSpline,
        bspline=si.BSpline.__call__,
        splev=_fitpack_py.splev,
        find_simplex=Delaunay.find_simplex,
    )
    _patch.replace(si.BSpline, "__call__", _patch.wrap(_orig["bspline"], _bspline_call, "BSpline.__call__"))
    _patch.replace(Delaunay, "find_simplex", _patch.wrap(_orig["find_simplex"], _find_simplex, "Delaunay.find_simplex"))
    rgi = si.RegularGridInterpolator
    _patch.replace(
        rgi, "_evaluate_spline", _patch.wrap(rgi._evaluate_spline, _evaluate_spline, "RegularGridInterpolator pchip")
    )
    _patch.rebind(_orig["splev"], _patch.wrap(_orig["splev"], _splev, "splev"))
    for cls in (_interpnd.LinearNDInterpolator, _interpnd.CloughTocher2DInterpolator):
        for name in ("_evaluate_double", "_evaluate_complex"):
            original, fast = _make_evaluate(cls, name)
            _patch.replace(cls, name, _patch.wrap(original, fast, f"{cls.__name__}.{name[1:]}"))


def smoke():
    import scipy.interpolate as si
    import scipy.spatial as sp

    rng = np.random.default_rng(0)
    x = np.sort(rng.random(400))
    spl = si.make_interp_spline(x, np.sin(x * 6), k=3)
    q = rng.uniform(x[0], x[-1], 1000)
    spl(q)
    si.splev(q, spl.tck)
    si.interp1d(x, np.sin(x * 6), kind="cubic")(q)
    pts = rng.random((1500, 2))
    tri = sp.Delaunay(pts)
    xq = rng.random((2000, 2))
    tri.find_simplex(xq)
    si.LinearNDInterpolator(tri, np.sin(pts[:, 0]))(xq)
    si.CloughTocher2DInterpolator(tri, np.sin(pts[:, 0]))(xq)
    g = np.linspace(0, 1, 12)
    si.RegularGridInterpolator((g, g), np.add.outer(g, g) ** 2, method="pchip")(rng.random((16, 2)))
