"""Byte-identity of fastscipy's scipy.ndimage fast paths against plain SciPy."""
import numpy as np
import pytest
import scipy.ndimage as ndi

import fastscipy
from fastscipy.testing import check

MODES = ["reflect", "constant", "nearest", "mirror", "wrap", "grid-wrap", "grid-constant", "grid-mirror"]
DTYPES = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
          np.float32, np.float64]
SEEDS = range(12)


@pytest.fixture(params=[1, 4], ids=["1t", "4t"], autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield request.param
    fastscipy.set_num_threads(old)


def array(rng, shape, dtype, special=False):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        a = rng.standard_normal(shape) * rng.choice([1.0, 100.0, 1e6])
        if special and a.size:
            idx = rng.integers(0, a.size, size=max(1, a.size // 20))
            a.flat[idx] = rng.choice([np.nan, np.inf, -np.inf, -0.0, 1e300, 5e-324], size=idx.size)
        with np.errstate(over="ignore"):
            return a.astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(max(info.min, -2**40), min(info.max, 2**40), size=shape, endpoint=True).astype(dtype)


def shape_of(rng, ndim=None, maxlen=40, minlen=1):
    ndim = ndim or int(rng.integers(1, 4))
    return tuple(max(minlen, int(rng.choice([1, 2, 3, int(rng.integers(1, maxlen))]))) for _ in range(ndim))


def view_of(rng, a):
    """``a`` or a Fortran-ordered / strided copy with the same values."""
    k = rng.integers(0, 3)
    if k == 1 and a.ndim >= 2:
        return np.asfortranarray(a)
    if k == 2:
        big = np.empty(tuple(2 * s for s in a.shape), a.dtype)
        v = big[(slice(None, None, 2),) * a.ndim]
        v[...] = a
        return v
    return a


def weights(rng, size, kind):
    half = rng.standard_normal(size // 2)
    mid = rng.standard_normal(1)
    w = {"sym": np.concatenate([half, mid, half[::-1]]),
         "anti": np.concatenate([half, mid, -half[::-1]])}.get(kind, rng.standard_normal(size))
    return w[:size]


def pick(rng, seq):
    return seq[rng.integers(len(seq))]


# --- correlate family -------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_correlate1d(seed):
    rng = np.random.default_rng(seed)
    for _ in range(25):
        x = view_of(rng, array(rng, shape_of(rng), pick(rng, DTYPES), rng.random() < 0.2))
        size = int(pick(rng, [1, 2, 3, 4, 5, 7, 9, int(rng.integers(1, 30)), int(rng.integers(19, 70))]))
        w = weights(rng, size, pick(rng, ["sym", "anti", "gen"]))
        origin = int(rng.integers(-(size // 2), (size - 1) // 2 + 1))
        fn = pick(rng, [ndi.correlate1d, ndi.convolve1d])
        if fn is ndi.convolve1d and size % 2 == 0 and origin == (size - 1) // 2:
            origin = 0
        kw = dict(axis=int(rng.integers(-x.ndim, x.ndim)), mode=pick(rng, MODES),
                  cval=float(pick(rng, [0.0, 1.5, -7.25, 1e5])), origin=origin,
                  output=None if rng.random() < 0.6 else pick(rng, DTYPES))
        check(lambda: getattr(ndi, fn.__name__)(x, w, **kw))


@pytest.mark.parametrize("seed", SEEDS)
def test_uniform_gaussian_filters(seed):
    rng = np.random.default_rng(100 + seed)
    for _ in range(12):
        x = view_of(rng, array(rng, shape_of(rng, maxlen=30), pick(rng, DTYPES), rng.random() < 0.2))
        mode = pick(rng, MODES)
        out = None if rng.random() < 0.6 else pick(rng, DTYPES)
        size = int(rng.integers(2, 12))
        check(lambda: ndi.uniform_filter(x, size, mode=mode, output=out, origin=0))
        check(lambda: ndi.uniform_filter1d(x, size, axis=-1, mode=mode, output=out))
        sigma, gorder = float(rng.uniform(0.3, 3)), int(rng.integers(0, 3))
        if x.dtype.kind == "f":
            check(lambda: ndi.gaussian_filter(x, sigma, mode=mode, order=gorder, truncate=3.0))


@pytest.mark.parametrize("seed", SEEDS)
def test_nd_correlate_derivatives(seed):
    rng = np.random.default_rng(200 + seed)
    for _ in range(10):
        x = view_of(rng, array(rng, shape_of(rng, rng.integers(1, 4), 25), pick(rng, DTYPES)))
        fshape = tuple(int(rng.integers(1, 6)) for _ in x.shape)
        w = rng.standard_normal(fshape)
        mode = pick(rng, MODES)
        origin = tuple(int(rng.integers(-(f // 2), (f - 1) // 2 + 1)) for f in fshape)
        check(lambda: ndi.correlate(x, w, mode=mode, origin=origin, cval=2.5))
        check(lambda: ndi.convolve(x, w, mode=mode, cval=-1.0))
        if x.ndim >= 2:
            check(lambda: ndi.laplace(x, mode=mode))
            check(lambda: ndi.sobel(x, axis=0, mode=mode))
            check(lambda: ndi.prewitt(x, axis=-1, mode=mode))


def test_correlate_inplace_and_out():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 40))
    for fn in (lambda a: ndi.correlate1d(a, [1.0, 2.0, 3.0], output=a),
               lambda a: ndi.uniform_filter1d(a, 5, output=a),
               lambda a: ndi.gaussian_filter(a, 2, output=a)):
        a, b = x.copy(), x.copy()
        with fastscipy.disabled():
            fn(a)
        fn(b)
        assert a.tobytes() == b.tobytes()
    out = np.zeros((50, 40), np.float32)[:, ::1]
    check(lambda: ndi.convolve(x, np.ones((3, 3)), output=out))


# --- interpolation ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_spline_zoom_shift(seed):
    rng = np.random.default_rng(300 + seed)
    for _ in range(12):
        x = view_of(rng, array(rng, shape_of(rng, rng.integers(1, 4), 24), pick(rng, DTYPES)))
        order, mode = int(rng.integers(0, 6)), pick(rng, MODES)
        gm = bool(rng.integers(2))
        kw = dict(order=order, mode=mode, cval=0.5, grid_mode=gm and mode in ("grid-wrap", "grid-constant"))
        factor, delta = float(rng.uniform(0.5, 2.2)), float(rng.uniform(-3, 3))
        check(lambda: ndi.zoom(x, factor, **kw))
        check(lambda: ndi.shift(x, [delta] * x.ndim, order=order, mode=mode))
        if x.dtype.kind == "f":
            check(lambda: ndi.spline_filter(x, order=max(order, 2), mode=mode))
            check(lambda: ndi.spline_filter1d(x, order=max(order, 2), axis=0, mode=mode))


@pytest.mark.parametrize("seed", SEEDS)
def test_affine_rotate_map_coordinates(seed):
    rng = np.random.default_rng(400 + seed)
    for _ in range(8):
        x = view_of(rng, array(rng, shape_of(rng, 2, 40), pick(rng, DTYPES)))
        order, mode = int(rng.integers(0, 6)), pick(rng, MODES[:5])
        m = np.eye(2) + 0.2 * rng.standard_normal((2, 2))
        check(lambda: ndi.affine_transform(x, m, offset=[0.3, -1.2], order=order, mode=mode))
        angle, reshape = float(rng.uniform(-180, 180)), bool(rng.integers(2))
        check(lambda: ndi.rotate(x, angle, order=order, mode=mode, reshape=reshape))
        coords = rng.uniform(-3, 43, size=(2, 5, 7))
        check(lambda: ndi.map_coordinates(x, coords, order=order, mode=mode, cval=1.0))
        check(lambda: ndi.geometric_transform(x, lambda c: c, order=order), fast=False)


# --- rank, min/max ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_rank_filters(seed):
    rng = np.random.default_rng(500 + seed)
    for _ in range(14):
        x = view_of(rng, array(rng, shape_of(rng, rng.integers(1, 4), 24, 2), pick(rng, DTYPES), rng.random() < 0.2))
        mode = pick(rng, MODES[:7])
        size = tuple(int(rng.integers(2, 6)) for _ in x.shape)
        fp = rng.random(size) < 0.7
        fp.flat[rng.integers(fp.size)] = True
        out = None if rng.random() < 0.7 else pick(rng, DTYPES)
        kw = dict(mode=mode, cval=1.0, output=out)
        f1d = x.ndim > 1 and out is None    # 1-D rank filters go to SciPy's own _rank_filter_1d
        check(lambda: ndi.median_filter(x, size=size, **kw), fast=f1d)
        check(lambda: ndi.minimum_filter(x, footprint=fp, **kw), fast=out is None)
        check(lambda: ndi.maximum_filter(x, size=size, **kw), fast=out is None)
        rank, pct, s1, s2 = int(rng.integers(-fp.sum(), fp.sum())), float(rng.uniform(0, 100)), *rng.integers(1, 9, 2)
        check(lambda: ndi.rank_filter(x, rank, footprint=fp, **kw), fast=f1d)
        check(lambda: ndi.percentile_filter(x, pct, footprint=fp, **kw), fast=f1d)
        check(lambda: ndi.minimum_filter1d(x, int(s1), axis=-1, mode=mode, output=out), fast=out is None)
        check(lambda: ndi.maximum_filter1d(x, int(s2), axis=0, mode=mode, output=out), fast=out is None)


def test_rank_filter_bool_and_inplace():
    rng = np.random.default_rng(2)
    b = rng.random((40, 30)) < 0.5
    check(lambda: ndi.maximum_filter(b, size=3))
    check(lambda: ndi.median_filter(b, size=3))
    x = rng.standard_normal((30, 30))
    a, c = x.copy(), x.copy()
    with fastscipy.disabled():
        ndi.maximum_filter1d(a, 5, output=a)
    ndi.maximum_filter1d(c, 5, output=c)
    assert a.tobytes() == c.tobytes()


# --- large inputs (parallel paths) ------------------------------------------------------------

def test_large_inputs():
    x = np.random.default_rng(3).random((600, 500)).astype(np.float32)
    check(lambda: ndi.gaussian_filter(x, 2.0))
    check(lambda: ndi.median_filter(x, 5))
    check(lambda: ndi.uniform_filter(x, 7))
    check(lambda: ndi.zoom(x, 1.3, order=3))
    check(lambda: ndi.rotate(x, 20, order=1))
    v = np.random.default_rng(4).random((60, 60, 60))
    check(lambda: ndi.gaussian_filter(v, 1.5))
    check(lambda: ndi.maximum_filter(v, 3))


# --- declined cases still match SciPy ---------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda r: r.standard_normal((20, 20)).astype(np.float16),
    lambda r: r.standard_normal((20, 20)).astype(">f8"),
    lambda r: (r.standard_normal((20, 20)) + 1j * r.standard_normal((20, 20))),
    lambda r: r.standard_normal((0, 5)),
    lambda r: r.standard_normal(()),
], ids=["f16", "bigendian", "complex", "empty", "0d"])
def test_declined_inputs(make):
    x = make(np.random.default_rng(5))
    for call in (lambda: ndi.gaussian_filter(x, 1.0), lambda: ndi.uniform_filter(x, 3),
                 lambda: ndi.median_filter(x, 3), lambda: ndi.zoom(x, 1.5)):
        try:
            with fastscipy.disabled():
                expected = call()
        except Exception as e:
            with pytest.raises(type(e)):
                call()
        else:
            check(call, fast=False)


def test_generic_callbacks_and_errors():
    x = np.random.default_rng(6).random((20, 20))
    check(lambda: ndi.generic_filter(x, np.ptp, size=3), fast=False)
    for call in (lambda: ndi.gaussian_filter(x, 1, mode="bogus"), lambda: ndi.uniform_filter(x, 3, origin=5),
                 lambda: ndi.median_filter(x, 3, origin=5), lambda: ndi.zoom(x, 2, order=7)):
        errors = []
        for off in (True, False):
            with pytest.raises(Exception) as e:
                if off:
                    with fastscipy.disabled():
                        call()
                else:
                    call()
            errors.append((type(e.value), str(e.value)))
        assert errors[0] == errors[1]


def test_disabled_runs_scipy_and_counts():
    x = np.random.default_rng(7).random((30, 30))
    before = fastscipy.stats().get(("correlate1d", "scipy"), 0)
    with fastscipy.disabled():
        ndi.gaussian_filter(x, 1.0)
    assert fastscipy.stats()[("correlate1d", "scipy")] > before
