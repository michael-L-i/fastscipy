import warnings

import numpy as np
import pytest
import scipy.interpolate as si
import scipy.spatial as sp

import fastscipy
from fastscipy.testing import assert_identical, check

ALL = fastscipy.get_num_threads()


@pytest.fixture(params=[1, ALL], ids=["1t", "allt"], autouse=True)
def threads(request):
    fastscipy.set_num_threads(request.param)
    yield request.param
    fastscipy.set_num_threads(ALL)


def rng(seed=0):
    return np.random.default_rng(seed)


def knots(kind, n, k, r):
    t = np.sort(r.random(n))
    if kind == "repeated":  # interior knots of multiplicity 2
        i = np.arange(1, n - 2, 7)
        t[i + 1] = t[i]
    if kind == "clamped":
        t = np.r_[[t[0]] * k, t, [t[-1]] * k]
    elif kind == "unclamped":
        t = np.r_[t[0] - np.arange(k, 0, -1) * 0.01, t, t[-1] + np.arange(1, k + 1) * 0.01]
    return t


# ---------------------------------------------------------------- BSpline / splev
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("kind", ["clamped", "unclamped", "repeated"])
def test_bspline_kernel(k, kind):
    r = rng(k)
    t = knots(kind, 400, k, r)
    spl = si.BSpline(t, r.standard_normal(len(t) - k - 1), k)
    lo, hi = t[k], t[-k - 1]
    q = np.r_[r.uniform(lo - 0.2, hi + 0.2, 5000), t, np.nextafter(t, 0), np.nextafter(t, 2), lo, hi]
    r.shuffle(q)
    check(lambda: spl(q))
    check(lambda: spl(q[: len(q) // 8 * 8].reshape(-1, 8)))
    check(lambda: spl(q.astype(np.float32)))


def test_bspline_sort_path():
    r = rng(1)
    k = 3
    t = knots("clamped", 400, k, r)
    q = r.uniform(t[0], t[-1], 4000)
    spl = si.BSpline(t, r.standard_normal((len(t) - k - 1, 3)), k)
    check(lambda: spl(q))  # 2-D coefficients
    check(lambda: spl(q.reshape(40, 100)))
    spl1 = si.BSpline(t, r.standard_normal(len(t) - k - 1), k)
    check(lambda: spl1(q, nu=1))
    check(lambda: spl1(q, extrapolate=False))
    cs = si.BSpline(t, r.standard_normal(len(t) - k - 1) + 1j, k)
    check(lambda: cs(q))
    spl_ax = si.BSpline(t, r.standard_normal((2, len(t) - k - 1, 3)), k, axis=1)
    check(lambda: spl_ax(q))
    check(lambda: spl_ax(q.reshape(20, 200)))
    check(lambda: spl_ax(q, nu=2))


def test_bspline_declined():
    r = rng(2)
    k = 3
    t = knots("clamped", 400, k, r)
    spl = si.BSpline(t, r.standard_normal(len(t) - k - 1), k)
    q = r.uniform(t[0], t[-1], 4000)
    bad = q.copy()
    bad[::50] = np.nan
    bad[1::50] = np.inf
    check(lambda: spl(bad), fast=False)
    check(lambda: spl(q, extrapolate="periodic"), fast=False)
    check(lambda: spl(q[:50]), fast=False)
    check(lambda: spl(np.sort(q)[:200]), fast=False)
    check(lambda: spl(list(q[:10])), fast=False)
    degenerate = np.r_[[0.0] * 6, np.sort(r.random(300)), [1.0] * 6]  # end knots of multiplicity k + 3
    dq = r.uniform(-0.1, 1.1, 3000)
    dspl = si.BSpline(degenerate, r.standard_normal(len(degenerate) - 4), 3)
    check(lambda: dspl(dq), fast=False)


def test_bspline_sorted_queries():
    r = rng(3)
    t = knots("clamped", 2000, 3, r)
    spl = si.BSpline(t, r.standard_normal(len(t) - 4), 3)
    q = np.sort(r.uniform(t[0], t[-1], 20000))
    check(lambda: spl(q), fast=False)  # SciPy's walk wins on one thread; either way identical


@pytest.mark.parametrize("nknots", [12, 90, 100])
def test_short_knot_vectors(nknots):
    r = rng(nknots)
    x = np.linspace(0, 1, nknots)
    spl = si.make_interp_spline(x, r.standard_normal(nknots))
    for n in (1000, 70000):
        q = r.uniform(-0.1, 1.1, n)
        check(lambda: spl(q), fast=False)
        check(lambda: spl(np.sort(q)), fast=False)
        check(lambda: spl(q, nu=1), fast=False)


@pytest.mark.parametrize("kind", ["cubic", "quadratic", "slinear", "zero", "linear", "nearest"])
def test_interp1d(kind):
    r = rng(4)
    x = np.sort(r.random(2000))
    f = si.interp1d(
        x,
        np.sin(9 * x),
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate" if kind in ("cubic", "quadratic", "slinear", "linear") else np.nan,
    )
    q = r.uniform(x[0], x[-1], 30000)
    check(lambda: f(q), fast=kind in ("cubic", "quadratic", "slinear", "zero"))
    check(
        lambda: si.interp1d(x, np.cos(x[:, None] * np.arange(1, 4)), kind=kind, axis=0)(q.clip(x[0], x[-1])),
        fast=kind in ("cubic", "quadratic", "slinear", "zero"),
    )


@pytest.mark.parametrize("k", [1, 3, 5])
def test_splev(k):
    r = rng(5)
    x = np.sort(r.random(600))
    tck = si.splrep(x, np.sin(7 * x), k=k, s=0)
    q = r.uniform(-0.1, 1.1, 20000)
    check(lambda: si.splev(q, tck))
    check(lambda: si.splev(q.reshape(100, 200), tck))
    check(lambda: si.splev(np.clip(q, 0, 1), tck, der=1))
    for ext in (1, 3):
        check(lambda: si.splev(q, tck, ext=ext))
    check(lambda: si.splev(np.clip(q, x[0], x[-1]), tck, ext=2))
    check(lambda: si.splev(q, (tck[0], tck[1], tck[2])), fast=True)
    check(lambda: si.splev(q.astype(np.float32), tck))
    check(lambda: si.splev(list(q[:20]), tck), fast=False)
    check(lambda: si.splev(q, si.BSpline(*tck)), fast=True)  # through BSpline.__call__
    with pytest.raises(ValueError, match="domain"):
        si.splev(q, tck, ext=2)


def test_univariate_spline():
    r = rng(6)
    x = np.sort(r.random(1000))
    s = si.UnivariateSpline(x, np.sin(8 * x) + 0.01 * r.standard_normal(1000), k=3, s=0.01)
    q = r.uniform(0, 1, 30000)
    check(lambda: s(q), fast=False)  # evaluates through _fitpack, not hooked


# ---------------------------------------------------------------- Delaunay
def points(kind, n, ndim, r):
    if kind == "random":
        return r.random((n, ndim))
    m = int(round(n ** (1 / ndim)))
    g = np.stack(np.meshgrid(*[np.arange(m)] * ndim, indexing="ij"), -1).reshape(-1, ndim) / (m - 1)
    return g + (1e-9 * r.random(g.shape) if kind == "jittered" else 0)  # jittered: nearly degenerate


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("kind", ["random", "lattice", "jittered"])
def test_find_simplex(ndim, kind):
    r = rng(ndim)
    pts = points(kind, 3000 if ndim == 2 else 2000, ndim, r)
    tri = sp.Delaunay(pts)
    n = 6000
    q = r.random((n, ndim)) * 1.2 - 0.1  # 15% outside the hull
    k = n // 10
    q[:k] = pts[r.integers(0, len(pts), k)]  # vertices
    q[k : 2 * k] = pts[r.integers(0, len(pts), k)] / 2 + pts[r.integers(0, len(pts), k)] / 2  # edge-ish points
    q[2 * k : 2 * k + 5] = np.nan
    q[2 * k + 5 : 2 * k + 8] = np.inf
    r.shuffle(q)
    check(lambda: tri.find_simplex(q), fast=kind == "random")  # lattices: heavy ties, replayed or declined
    check(lambda: tri.find_simplex(q[:5400].reshape(60, 90, ndim)), fast=False)


def test_find_simplex_ties_replayed():
    r = rng(7)
    pts = r.random((4000, 2))
    tri = sp.Delaunay(pts)
    q = r.random((20000, 2))
    q[::40] = pts[r.integers(0, len(pts), 500)]  # 2.5% vertices: replayed, still fast
    check(lambda: tri.find_simplex(q))
    q[:] = pts[r.integers(0, len(pts), len(q))]
    check(lambda: tri.find_simplex(q), fast=False)  # all ties: declined


def test_find_simplex_shapes_and_declines():
    r = rng(8)
    tri = sp.Delaunay(r.random((3000, 2)))
    q = r.random((50, 400, 2))
    check(lambda: tri.find_simplex(q))
    check(lambda: tri.find_simplex(q.astype(np.float32)))
    check(lambda: tri.find_simplex(q.reshape(-1, 2)[:100]), fast=False)
    check(lambda: tri.find_simplex(q.reshape(-1, 2), bruteforce=False, tol=1e-6), fast=False)
    check(lambda: tri.find_simplex(q.reshape(-1, 2)[:300], bruteforce=True), fast=False)
    check(lambda: tri.find_simplex(np.sort(q.reshape(-1, 2), axis=0)), fast=False)  # already local
    one = r.random(2)
    check(lambda: tri.find_simplex(one), fast=False)
    check(lambda: tri.find_simplex(np.empty((0, 2))), fast=False)
    small, sq = sp.Delaunay(r.random((200, 2))), r.random((5000, 2))
    check(lambda: small.find_simplex(sq), fast=False)  # few simplices


# ---------------------------------------------------------------- LinearND / CloughTocher / griddata
def scattered(n=3000, ndim=2, seed=9):
    r = rng(seed)
    pts = r.random((n, ndim))
    return r, pts, np.sin(3 * pts[:, 0]) + pts[:, -1] ** 2


@pytest.mark.parametrize("ndim", [2, 3])
def test_linear_nd(ndim):
    r, pts, v = scattered(3000, ndim)
    q = r.random((8000, ndim)) * 1.2 - 0.1
    q[:200] = pts[:200]
    r.shuffle(q)
    f = si.LinearNDInterpolator(pts, v)
    check(lambda: f(q))
    check(lambda: f(q[:, 0], q[:, 1]) if ndim == 2 else f(q))
    check(lambda: si.LinearNDInterpolator(pts, np.c_[v, 2 * v, v**2], fill_value=-1.5)(q))
    check(lambda: si.LinearNDInterpolator(pts, v * (1 + 1j))(q))  # complex: reordered SciPy
    check(lambda: si.LinearNDInterpolator(pts, v.astype(np.float32))(q))
    check(lambda: si.LinearNDInterpolator(pts, v, rescale=True)(q))
    check(lambda: f(q, simplex_tolerance=2.0), fast=False)
    check(lambda: f(q[:100]), fast=False)


def test_linear_nd_nan_and_ties():
    r, pts, v = scattered()
    f = si.LinearNDInterpolator(pts, v)
    q = r.random((9000, 2))
    q[::30] = pts[r.integers(0, len(pts), 300)]
    q[5::97] = np.nan
    check(lambda: f(q))


def test_clough_tocher():
    r, pts, v = scattered()
    q = r.random((8000, 2)) * 1.2 - 0.1
    q[:100] = pts[:100]
    r.shuffle(q)
    check(lambda: si.CloughTocher2DInterpolator(pts, v)(q))
    check(lambda: si.CloughTocher2DInterpolator(pts, np.c_[v, v**2], fill_value=0.5)(q))
    check(lambda: si.CloughTocher2DInterpolator(pts, v * (1 - 2j))(q))
    check(lambda: si.CloughTocher2DInterpolator(pts, v, tol=1e-4, maxiter=50)(q))


@pytest.mark.parametrize("complex_values", [False, True])
def test_clough_tocher_many_outputs(complex_values):
    r, pts, v = scattered()
    values = (v[:, None] * np.arange(1, 25)).reshape(-1, 4, 6)
    if complex_values:
        values = values * (1 - 2j)
    f = si.CloughTocher2DInterpolator(pts, values, fill_value=-1.5)
    # Uneven chunks, multiple output axes, extrapolation, NaNs, and tie replay.
    q = r.uniform(-0.1, 1.1, (8501, 2))
    q[:50] = pts[:50]
    q[50:53] = np.nan
    r.shuffle(q)
    check(lambda: f(q))


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_griddata(method):
    r, pts, v = scattered(4000)
    q = r.random((9000, 2)) * 1.1
    check(lambda: si.griddata(pts, v, q, method=method))
    check(lambda: si.griddata(pts, v, q, method=method, fill_value=0.0, rescale=True))
    check(lambda: si.griddata(pts, v, (q[:, 0], q[:, 1]), method=method))
    check(lambda: si.griddata(pts, v, q[:100], method=method), fast=False)
    check(lambda: si.griddata(pts, v, q, method="nearest"), fast=False)


# ---------------------------------------------------------------- RegularGridInterpolator pchip
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("uniform", [True, False])
def test_rgi_pchip(ndim, uniform):
    r = rng(10 + ndim)
    shape = (60, 25, 12)[:ndim] if ndim > 1 else (200,)
    grid = [np.linspace(0, 1, s) if uniform else np.sort(r.random(s)) for s in shape]
    v = r.standard_normal(shape)
    v[..., 0] = 1.0  # flat runs
    if ndim > 1:
        v[:, 3] = v[:, 2]
    f = si.RegularGridInterpolator(grid, v, method="pchip", bounds_error=False, fill_value=None)
    rf = lambda vals, **kw: si.RegularGridInterpolator(grid, vals, method="pchip", **kw)
    lo = [g[0] for g in grid]
    hi = [g[-1] for g in grid]
    q = r.uniform(lo, hi, (500, ndim))
    q[:50] = [[g[r.integers(len(g))] for g in grid] for _ in range(50)]  # on nodes
    q[50:60] = r.uniform(np.array(lo) - 0.1, np.array(hi) + 0.1, (10, ndim))  # outside, extrapolated
    check(lambda: f(q[:60]))
    qn = q.copy()
    qn[60:63] = np.nan
    if ndim == 1:
        check(lambda: f(qn), fast=False)
    else:
        with pytest.raises(ValueError, match="finite"):  # SciPy refits on the NaN it produced
            f(qn)
    check(lambda: si.interpn(grid, v, q, method="pchip", bounds_error=False, fill_value=-1.0))
    inside = q[63:].clip(lo, hi)
    check(lambda: rf(np.round(v, 1))(inside))
    check(lambda: f(q[:5]), fast=False)
    check(lambda: f(q[0]), fast=False)
    check(lambda: rf(v.astype(np.float32))(inside), fast=False)
    check(lambda: si.RegularGridInterpolator(grid, v, method="cubic")(inside), fast=False)


def test_rgi_pchip_declined():
    r = rng(11)
    g = (np.linspace(0, 1, 10), np.linspace(0, 1, 9))
    q = r.random((50, 2))
    trailing = si.RegularGridInterpolator(g, r.random((10, 9, 3)), method="pchip")
    check(lambda: trailing(q), fast=False)
    gs = (np.linspace(0, 1, 4),) * 4
    four, q4 = si.RegularGridInterpolator(gs, r.random((4,) * 4), method="pchip"), r.random((50, 4))
    check(lambda: four(q4), fast=False)
    v = r.random((10, 9))
    v[2, 2] = np.nan
    nan = si.RegularGridInterpolator(g, v, method="pchip")
    with pytest.raises(ValueError, match="finite"):
        nan(q)


# NumPy PPoly/BPoly keep SciPy's arithmetic while avoiding namespace adapters.
@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("descending", [False, True])
def test_piecewise_numpy_dispatch(cls, complex_, axis, descending):
    r = rng(205)
    c = r.standard_normal((4, 13, 2, 3))
    if complex_:
        c = c + 1j * r.standard_normal(c.shape)
    x = np.linspace(-2, 3, 14)
    if descending:
        x = x[::-1]
    c = np.moveaxis(c, (0, 1), (axis, axis + 1))
    p = cls(c, x, axis=axis)
    q = np.r_[r.uniform(-3, 4, 73), x, np.nan, np.inf, -np.inf, -0.0]
    q.flags.writeable = False
    for query in (q, q[:84].reshape(7, 12), np.array(0.0), np.empty((0, 2))):
        for nu in (0, 1, 3, 5):
            for ext in (None, True, False):
                check(lambda: p(query, nu=nu, extrapolate=ext))


@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
def test_piecewise_mutated_coefficients(cls):
    r = rng(206)
    p = cls(r.standard_normal((4, 13, 2)), np.linspace(0, 1, 14))
    p.c = p.c[:, :, ::-1]
    p.x = p.x.copy()[::-1]
    q = r.random(11)
    check(lambda: p(q))
    assert p._delegate_to.c.flags.c_contiguous
    assert p._delegate_to.x.flags.c_contiguous


@pytest.mark.parametrize("cls", [si.CubicSpline, si.PchipInterpolator, si.Akima1DInterpolator])
def test_piecewise_derived_evaluation(cls):
    x = np.linspace(0, 1, 14)
    p = cls(x, np.sin(x * 6))
    q = rng(207).random(17)
    for obj in (p, p.derivative(), p.antiderivative()):
        check(lambda: obj(q))


@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
def test_piecewise_declined_options(cls):
    p = cls(np.ones((4, 3)), np.arange(4.0))
    for q in ([0.2, 0.8], np.array([0.2], dtype=np.longdouble)):
        check(lambda: p(q), fast=False)
    with pytest.warns(np.exceptions.ComplexWarning):
        check(lambda: p(np.array([0.2 + 0.1j])), fast=False)
    for opts in ({"nu": -1}, {"nu": 0.5}, {"extrapolate": "periodic"}, {"extrapolate": np.bool_(False)}):
        check(lambda: p(np.array([0.2, 3.2]), **opts), fast=False)
    p._xp = np
    check(lambda: p(np.array([0.2])), fast=False)


@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
def test_piecewise_custom_evaluator_not_retried(cls):
    p = cls(np.ones((4, 3)), np.arange(4.0))
    calls = []

    def evaluate(*args):
        calls.append(1)
        raise RuntimeError("custom evaluator")

    p._delegate_to._evaluate = evaluate
    with pytest.raises(RuntimeError, match="custom evaluator"):
        p(np.array([0.2]))
    assert calls == [1]


@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int8, np.int32, np.uint64])
@pytest.mark.parametrize("axis", [0, 1])
def test_piecewise_query_conversion(cls, dtype, axis):
    r = rng(301)
    c = r.standard_normal((4, 13, 2)) + 1j * r.standard_normal((4, 13, 2))
    p = cls(np.moveaxis(c, (0, 1), (axis, axis + 1)), np.linspace(-2, 3, 14), axis=axis)
    q = r.uniform(0, 4, (12, 16)).astype(dtype)
    for query in (q, q[::-1, ::-2], np.asfortranarray(q), q[:0]):
        query.flags.writeable = False
        before = query.copy()
        for nu in (0, 1, 5):
            check(lambda: p(query, nu=nu))
        assert np.array_equal(query, before)


@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
@pytest.mark.parametrize("scalar", [0.0, -0.0, 0.25, np.float32(0.25), np.float64(0.25), np.nan, np.inf, -np.inf])
def test_piecewise_scalar_queries(cls, scalar):
    p = cls(rng(302).standard_normal((4, 13, 2)), np.linspace(-2, 3, 14))
    check(lambda: p(scalar))
    check(lambda: p(scalar, nu=1, extrapolate=False))


@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
def test_piecewise_queries_and_coefficients_not_cached(cls):
    p = cls(rng(303).standard_normal((4, 13)), np.linspace(-2, 3, 14))
    q = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    check(lambda: p(q))
    q[:] += 0.25
    p.c[:] *= 2
    check(lambda: p(q))


@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
def test_piecewise_signaling_nan_conversion_warning(cls):
    p = cls(np.ones((4, 3)), np.arange(4.0))
    q = np.array([0x7F800001, 0x3F000000], dtype=np.uint32).view(np.float32)
    before = q.tobytes()
    results, messages = [], []
    for enabled in (False, True):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            if enabled:
                result = p(q)
            else:
                with fastscipy.disabled():
                    result = p(q)
        results.append(result)
        messages.append([(type(w.message), str(w.message)) for w in seen])
    assert_identical(*results)
    assert messages[0] == messages[1]
    assert len(messages[0]) == 1
    assert q.tobytes() == before


@pytest.mark.parametrize("cls", [si.PPoly, si.BPoly])
@pytest.mark.parametrize("mode", ["call", "log"])
def test_piecewise_raising_seterr_handler_once(cls, mode):
    p = cls(np.ones((4, 3)), np.arange(4.0))
    q = np.array([0x7F800001], dtype=np.uint32).view(np.float32)
    calls = []

    def handler(*args):
        calls.append(args)
        raise RuntimeError("conversion handler")

    class Logger:
        write = staticmethod(handler)

    old = np.seterrcall(handler if mode == "call" else Logger())
    try:
        with np.errstate(invalid=mode):
            with pytest.raises(RuntimeError, match="conversion handler"):
                p(q)
    finally:
        np.seterrcall(old)
    assert len(calls) == 1
