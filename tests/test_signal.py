import warnings

import numpy as np
import pytest
import scipy.signal as sg

import fastscipy
from fastscipy import _threads
from fastscipy.testing import identical

THREADS = [1, 8]
ALIAS = {"butter": "iirfilter", "cheby1": "iirfilter"}


def _run(name, args, kwargs):
    args = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = "ok", getattr(sg, name)(*args, **kwargs)
        except Exception as e:
            out = "err", type(e), str(e)
    return out, [(x.category, str(x.message)) for x in w], args


def cmp(name, *args, fast=True, **kwargs):
    """The patched call matches SciPy's: result bytes, exception, warnings, mutated arguments."""
    with fastscipy.disabled():
        ref = _run(name, args, kwargs)
    key = ALIAS.get(name, name), "fast"
    before = fastscipy.stats().get(key, 0)
    got = _run(name, args, kwargs)
    assert ref[0][0] == got[0][0], (ref[0], got[0])
    assert ref[0][1:] == got[0][1:] if ref[0][0] == "err" else identical(ref[0][1], got[0][1])
    assert ref[1] == got[1]
    assert all(identical(a, b) for a, b in zip(ref[2], got[2]))
    if fast is not None:
        assert (fastscipy.stats().get(key, 0) > before) == fast, "fast path ran or not as expected"
    return got[0]


@pytest.fixture(params=THREADS, autouse=True)
def threads(request):
    n = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield request.param
    fastscipy.set_num_threads(n)


def signals(rng, n):
    x = rng.standard_normal(n)
    return [x, x * 1e-300, x * 1e300, np.zeros(n), np.arange(n, dtype=float), x.astype(np.float32)]


def run_all(sos, b, a, X):
    # short or extreme inputs may legitimately decline, so only SciPy's result is asserted here
    for f in (lambda: cmp("sosfiltfilt", sos, X, fast=None), lambda: cmp("filtfilt", b, a, X, fast=None),
              lambda: cmp("sosfilt", sos, X, fast=None), lambda: cmp("lfilter", b, a, X.astype(np.float64), fast=None),
              lambda: cmp("detrend", X, fast=None), lambda: cmp("hilbert", X.astype(np.float64), fast=None)):
        f()


FILTERS = [(N, wn, bt) for N in (1, 2, 4, 8) for wn, bt in ((0.1, "low"), (0.3, "high"), ([0.1, 0.4], "band"))]


@pytest.mark.parametrize("N,wn,bt", FILTERS)
def test_filters_1d_2d(N, wn, bt):
    before = fastscipy.stats()
    rng = np.random.default_rng(N)
    sos = sg.butter(N, wn, bt, output="sos")
    b, a = sg.butter(N, wn, bt)
    for n in (40, 100, 1000):
        for x in signals(rng, n):
            for X in (x, np.stack([x, 2 * x[::-1], x + 1])):
                run_all(sos, b, a, X)
    after = fastscipy.stats()
    for name in ("sosfiltfilt", "filtfilt", "sosfilt", "lfilter", "detrend", "hilbert"):
        assert after[name, "fast"] > before.get((name, "fast"), 0), name


def test_filter_zi_and_caches():
    rng = np.random.default_rng(1)
    for N, wn, bt in FILTERS:
        sos = sg.butter(N, wn, bt, output="sos")
        b, a = sg.butter(N, wn, bt)
        cmp("sosfilt_zi", sos)
        cmp("lfilter_zi", b, a)
        for _ in range(2):     # second call is a cache hit
            for out in ("ba", "sos", "zpk"):
                cmp("iirfilter", N, wn, None, None, bt, False, "butter", out)
                cmp("butter", N, wn, bt, False, out)
            cmp("cheby1", N, 1, wn, bt, output="sos")
        x = rng.standard_normal(200)
        zi = sg.lfilter_zi(b, a)
        cmp("lfilter", b, a, x, zi=zi * x[0])
        zs = sg.sosfilt_zi(sos)
        cmp("sosfilt", sos, x, zi=zs * 0.7)
        X = rng.standard_normal((3, 200))
        cmp("sosfilt", sos, X, zi=np.broadcast_to(zs[:, None, :], (sos.shape[0], 3, 2)).copy())
    r1 = sg.sosfilt_zi(sos)
    r1[...] = 7
    assert not (sg.sosfilt_zi(sos) == 7).any()      # callers cannot corrupt the cache


def test_detrend_options():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((5, 60, 3))
    for axis in (0, 1, 2, -1):
        cmp("detrend", X, axis=axis)
        cmp("detrend", X, axis=axis, overwrite_data=True)
    cmp("detrend", X, type="constant", fast=False)
    cmp("detrend", X, bp=[20, 40], fast=False)
    cmp("detrend", np.array([1.0, np.nan, 3, 4]), fast=False)
    cmp("detrend", np.array([1e200, 2, 3, 4]), fast=False)
    cmp("detrend", np.array([1.0]), fast=False)


def test_welch_csd_coherence():
    before = fastscipy.stats()
    rng = np.random.default_rng(3)
    for n in (64, 257, 1000, 3000):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        X = np.stack([x, y, x + y])
        for kw in ({}, {"nperseg": 64}, {"nperseg": 50, "noverlap": 10}, {"nperseg": 64, "nfft": 100},
                   {"window": ("kaiser", 5.0), "nperseg": 32}, {"window": np.hanning(40), "fs": 100},
                   {"detrend": False, "scaling": "spectrum"}, {"nperseg": 33, "noverlap": 0}):
            cmp("welch", x, fast=None, **kw)
            cmp("welch", X, fast=None, **kw)
            cmp("csd", x, y, fast=None, **kw)
            cmp("csd", X, X[::-1].copy(), fast=None, **kw)
            cmp("csd", x, x, fast=None, **kw)
            cmp("coherence", x, y, fast=None, **kw)
        cmp("periodogram", x, fast=None)
        cmp("periodogram", X, window="hann", fast=None)
    after = fastscipy.stats()
    for name in ("welch", "csd"):
        assert after[name, "fast"] > before.get((name, "fast"), 0) + 20, name
    x = rng.standard_normal(100)
    for kw in ({"nperseg": 200}, {"detrend": "linear"}, {"return_onesided": False}, {"average": "median"},
               {"scaling": "bogus"}, {"nperseg": 0}, {"noverlap": 100}, {"nfft": 5, "nperseg": 20}):
        cmp("welch", x, fast=False, **kw)
    cmp("welch", x.astype(np.float32), fast=False)
    cmp("welch", np.empty(0), fast=False)
    cmp("csd", x, x[:50], fast=False)


def test_nonfinite_and_odd_inputs_match_scipy():
    rng = np.random.default_rng(4)
    sos = sg.butter(3, 0.2, output="sos")
    b, a = sg.butter(3, 0.2)
    for n in (30, 200):
        for bad in (np.nan, np.inf, -np.inf, 1e308):
            x = rng.standard_normal(n)
            x[n // 2] = bad
            for name, args in (("sosfiltfilt", (sos, x)), ("filtfilt", (b, a, x)), ("sosfilt", (sos, x)),
                               ("lfilter", (b, a, x)), ("hilbert", (x,)), ("welch", (x,)), ("detrend", (x,))):
                cmp(name, *args, **({"nperseg": 16} if name == "welch" else {}), fast=None)
    x = rng.standard_normal(100)
    for name, args, kw in (
        ("sosfiltfilt", (sos, x), {"padtype": "even"}), ("sosfiltfilt", (sos, x), {"padlen": 5}),
        ("sosfiltfilt", (sos, x), {"padlen": 1000}), ("sosfiltfilt", (sos, x[:5]), {}),
        ("sosfiltfilt", (sos, x), {"axis": 3}), ("sosfiltfilt", (sos.astype(np.float32), x), {}),
        ("filtfilt", (b, a, x), {"method": "gust"}), ("filtfilt", (b, a, x), {"padtype": None}),
        ("filtfilt", (b, a, x[:3]), {}), ("filtfilt", (b, np.array([0.0, 1]), x), {}),
        ("sosfilt", (sos, x.astype(np.int64)), {}), ("sosfilt", (sos, x), {"zi": np.zeros((9, 2))}),
        ("lfilter", (b, a, x.astype(np.float32)), {}), ("lfilter", (b, a, x), {"axis": 4}),
        ("lfilter", (b, [0.0, 1.0], x), {}), ("lfilter", (b, [1.0], x), {}),
        ("lfilter", (b, a, x.astype(complex)), {}), ("lfilter", ([], a, x), {}),
        ("hilbert", (x,), {"N": 120}), ("hilbert", (x.astype(complex),), {}),
        ("sosfilt_zi", (sos[:, :5],), {}), ("lfilter_zi", (b, a[:2]), {}), ("lfilter_zi", ([0.0], a), {}),
        ("butter", (3, 1.5), {}), ("butter", (0, 0.5), {}), ("butter", (3, [0.1, 0.1], "band"), {}),
    ):
        cmp(name, *args, fast=None, **kw)
    cmp("sosfiltfilt", sos, np.asfortranarray(rng.standard_normal((6, 80))))
    cmp("sosfiltfilt", sos, rng.standard_normal((6, 160))[:, ::2])
    cmp("filtfilt", b, a, rng.standard_normal((6, 160))[:, ::2])
    cmp("sosfiltfilt", sos, rng.standard_normal((80, 6)), axis=0, fast=False)
    cmp("sosfiltfilt", sos, rng.standard_normal((2, 3, 80)), fast=False)
    cmp("sosfiltfilt", sos, x.tolist(), fast=False)


@pytest.fixture
def pmap(monkeypatch, threads):
    """Counts the calls that really went to the thread pool."""
    calls = []
    real = _threads.parallel_map
    monkeypatch.setattr(_threads, "parallel_map", lambda fn, items: calls.append(1) or real(fn, items))
    return lambda: len(calls)


def cmp_threaded(pmap, threads, name, *args, expect=True, **kwargs):
    before = pmap()
    cmp(name, *args, fast=None, **kwargs)
    if expect is not None:     # None: not threaded itself, but may call a threaded function inside
        assert (pmap() > before) == (expect and threads > 1), (name, kwargs)


LAYOUTS = [
    ((6, 80000), lambda x: x), ((6, 80000), np.asfortranarray), ((80000, 6), lambda x: x),
    ((12, 4, 14000), lambda x: x), ((5, 3, 30000), lambda x: x[:, ::-1]), ((5, 160000), lambda x: x[:, ::2]),
    ((6, 80000), lambda x: x.astype(np.float32)), ((8, 60000), lambda x: x.astype(np.int16)),
    ((3, 2, 60000), lambda x: x + 1j * x[::-1]),
]


@pytest.mark.parametrize("shape,prep", LAYOUTS)
def test_threaded_filters(pmap, threads, shape, prep):
    x = prep(np.random.default_rng(5).standard_normal(shape) * 100)
    b, a = sg.butter(6, 0.2)
    sos = sg.butter(6, 0.2, output="sos")
    for axis in range(-x.ndim, x.ndim):
        run = lambda *a, **k: cmp_threaded(pmap, threads, *a, axis=axis, **k)
        run("lfilter", b, a, x)
        run("sosfilt", sos, x)
        if x.shape[axis] > 60:
            run("filtfilt", b, a, x)
            run("sosfiltfilt", sos, x)
            run("filtfilt", b, a, x, method="gust", expect=None)
    if x.dtype.kind == "c":
        return
    for ax in (0, -1):
        if x.shape[ax] > 60:
            cmp_threaded(pmap, threads, "decimate", x, 4, axis=ax)
            cmp_threaded(pmap, threads, "decimate", x, 3, ftype="fir", axis=ax)
            cmp_threaded(pmap, threads, "decimate", x, 5, zero_phase=False, axis=ax)
            cmp_threaded(pmap, threads, "resample_poly", x, 3, 2, axis=ax)
            cmp_threaded(pmap, threads, "resample_poly", x, 3, 2, axis=ax, padtype="line", expect=None)
            cmp_threaded(pmap, threads, "resample_poly", x, 1, 3, axis=ax, padtype="mean", expect=None)
            cmp_threaded(pmap, threads, "upfirdn", np.ones(16), x, 2, 3, axis=ax)


def test_threaded_initial_conditions(pmap, threads):
    rng = np.random.default_rng(6)
    b, a = sg.butter(3, 0.3)
    sos = sg.butter(3, 0.3, output="sos")
    x = rng.standard_normal((6, 100000))
    zi = sg.lfilter_zi(b, a) * x[:, :1]
    cmp_threaded(pmap, threads, "lfilter", b, a, x, zi=zi)
    zs = np.broadcast_to(sg.sosfilt_zi(sos)[:, None, :], (sos.shape[0], 6, 2)) * rng.random((1, 6, 1))
    cmp_threaded(pmap, threads, "sosfilt", sos, x, zi=zs)
    cmp_threaded(pmap, threads, "lfilter", b, a, np.ascontiguousarray(x.T), axis=0, zi=np.ascontiguousarray(zi.T))
    sos1 = sg.butter(2, 0.3, output="sos")      # one section: a size-1 dimension in zf
    x1 = rng.standard_normal((6, 200000))
    cmp_threaded(pmap, threads, "sosfilt", sos1, x1, zi=rng.standard_normal((1, 6, 2)))
    x3 = rng.standard_normal((3, 4, 100000))
    for axis in (0, 1, 2):
        shape = list(x3.shape)
        shape[axis] = 2
        cmp_threaded(pmap, threads, "sosfilt", sos1, x3, axis=axis, zi=rng.standard_normal((1, *shape)))
    cmp_threaded(pmap, threads, "lfilter", b, a, x, zi=zi[:3], expect=False)     # wrong zi shape: SciPy raises
    cmp_threaded(pmap, threads, "lfilter", b, a, x, axis=7, expect=False)
    cmp_threaded(pmap, threads, "sosfiltfilt", sos, x[:, :8], expect=False)      # too short: SciPy raises


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("ks", [3, 5, [3, 5], [7, 3], 4, [3, 4]])
def test_threaded_medfilt2d(pmap, threads, dtype, ks):
    img = (np.random.default_rng(7).random((700, 500)) * 255).astype(dtype)
    odd = bool(np.all(np.asarray(ks) % 2 == 1))
    cmp_threaded(pmap, threads, "medfilt2d", img, ks, expect=odd)
    cmp_threaded(pmap, threads, "medfilt2d", img[:, ::2], ks, expect=odd)


def test_unpatch_restores():
    f = sg.sosfiltfilt
    fastscipy.unpatch(["signal"])
    assert sg.sosfiltfilt is not f and not hasattr(sg.sosfiltfilt, "fast")
    fastscipy.patch(["signal"])
    assert hasattr(sg.sosfiltfilt, "fast")
