"""Byte-identity of fastscipy's scipy.ndimage fast paths against plain SciPy."""

import os
import math

import numpy as np
import pytest
import scipy.ndimage as ndi

import fastscipy
from fastscipy import _native
from fastscipy.testing import check

MODES = ["reflect", "constant", "nearest", "mirror", "wrap", "grid-wrap", "grid-constant", "grid-mirror"]
DTYPES = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64]
SEEDS = range(int(os.environ.get("FASTSCIPY_TEST_SEEDS", 16)))  # raise for a soak run


@pytest.fixture(params=[1, 4], ids=["1t", "4t"], autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    _native.ndimage.set_par_min(1 if request.param > 1 else 1 << 18)  # split even tiny label/fill inputs
    yield request.param
    _native.ndimage.set_par_min(1 << 18)
    fastscipy.set_num_threads(old)


def array(rng, shape, dtype, special=False):
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return rng.random(shape) < 0.5
    if dtype.kind == "f":
        a = rng.standard_normal(shape) * rng.choice([1.0, 100.0, 1e6])
        if special and a.size:
            idx = rng.integers(0, a.size, size=max(1, a.size // 20))
            a.flat[idx] = rng.choice([np.nan, np.inf, -np.inf, -0.0, 1e300, 5e-324], size=idx.size)
        with np.errstate(over="ignore"):
            return a.astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(max(info.min, -(2**40)), min(info.max, 2**40), size=shape, endpoint=True).astype(dtype)


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
    w = {"sym": np.concatenate([half, mid, half[::-1]]), "anti": np.concatenate([half, mid, -half[::-1]])}.get(
        kind, rng.standard_normal(size)
    )
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
        kw = dict(
            axis=int(rng.integers(-x.ndim, x.ndim)),
            mode=pick(rng, MODES),
            cval=float(pick(rng, [0.0, 1.5, -7.25, 1e5])),
            origin=origin,
            output=None if rng.random() < 0.6 else pick(rng, DTYPES),
        )
        check(lambda: getattr(ndi, fn.__name__)(x, w, **kw))


@pytest.mark.parametrize("kind", ["sym", "anti", "general"])
def test_correlate_four_tap_prefix(kind):
    from fastscipy._ndimage_seam import core
    from tools.probe_ndimage_platform import windows_arm64_reference

    rng = np.random.default_rng(913)
    x = rng.standard_normal(67) * 100
    channels = np.vstack([x, x[::-1], x * 0.5])
    # Around the six-iteration vectorization threshold and four/eight-tap
    # boundaries, including filters that use the generic kernel family.
    for iterations in (4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 31, 32):
        if kind == "general":
            w = rng.standard_normal(iterations + 1)
        else:
            half = rng.standard_normal(iterations)
            w = np.r_[half, rng.standard_normal(1), half[::-1] if kind == "sym" else -half[::-1]]
        check(lambda: ndi.correlate1d(x, w))
        check(lambda: ndi.correlate1d(channels, w, axis=1))
        check(lambda: ndi.correlate1d(channels.T, w, axis=0))
        if hasattr(math, "fma"):
            expected = windows_arm64_reference(x, w, kind)
            got = np.empty_like(x)
            assert core.corr_correlate1d(x, w, 0, got, 2, 0.0, 0, 3)
            assert expected.tobytes() == got.tobytes()


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
    for fn in (
        lambda a: ndi.correlate1d(a, [1.0, 2.0, 3.0], output=a),
        lambda a: ndi.uniform_filter1d(a, 5, output=a),
        lambda a: ndi.gaussian_filter(a, 2, output=a),
    ):
        a, b = x.copy(), x.copy()
        with fastscipy.disabled():
            fn(a)
        fn(b)
        assert a.tobytes() == b.tobytes()
    out = np.zeros((50, 40), np.float32)[:, ::1]
    check(lambda: ndi.convolve(x, np.ones((3, 3)), output=out))


@pytest.mark.parametrize("dtype", DTYPES)
def test_filter_compiler_cast_fallback(monkeypatch, dtype):
    from fastscipy import _ndimage_correlate as corr

    # Exercise the conversion-only fallback even on compilers whose direct
    # native cast already matches. Include vector prefixes and scalar tails.
    cal = dict(corr.calibration(), line_types=frozenset(), safe_types=frozenset())
    monkeypatch.setattr(corr, "_cal", cal)
    x = np.resize(corr._probe_values(), (7, 9))
    for axis in (0, 1):
        for fn in (ndi.correlate1d, ndi.uniform_filter1d):
            w = [0.3, -0.2, 1.1] if fn is ndi.correlate1d else 3
            check(lambda: fn(x, w, axis=axis, output=dtype))
            check(lambda: fn(x, w, axis=axis, output=np.empty((14, 18), dtype)[::2, ::2]))
    check(lambda: ndi.correlate(x, [[0.3, -0.2], [1.1, 0.7]], output=dtype))


@pytest.mark.parametrize("rows", [1, 7, 8, 9])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("length", [1, 2, 7, 8, 9])
def test_uniform_filter_nan_payloads(rows, axis, length):
    # Last-axis groups below/at/above eight lines take scalar, SIMD, and
    # remainder paths. Transposing exercises non-last-axis SIMD and tails.
    bits = np.array([0x7FF8002460000000, 0xFFF800ACE0000000, 0, 0, 0, 0, 0, 0, 0], np.uint64)
    for values in (
        [np.inf, -np.inf, np.nan, 0.0, -0.0, 0.4, -0.4, 0.9999, -0.9999],
        bits.view(np.float64),
        # Repeated distinct payloads exercise the final rolling update too:
        # MacIntel's paired body and even-length final tail use opposite
        # operand orders. Length 1 initializes only; length 2 has only a tail.
        bits[:2].view(np.float64),
    ):
        x = np.tile(np.resize(values, length), (rows, 1))
        if axis == 0:
            x = x.T.copy()
        x.flags.writeable = False
        for size in (3, 4):
            for dtype in (np.float32, np.float64):
                check(lambda: ndi.uniform_filter1d(x, size, axis=axis, output=dtype, mode="reflect"))


def test_edt_sampled_ties():
    rng = np.random.default_rng(901)
    x = rng.random((7, 11, 13)) > 0.8
    for sampling in ([0.7, 1.3, 2.1], [1.0, 1.3, 1.3], [0.1, 0.3, 0.9]):
        check(lambda: ndi.distance_transform_edt(x, sampling=sampling, return_indices=True))


@pytest.mark.parametrize("dtype", DTYPES[:8])
def test_interpolation_compiler_cast_fallback(monkeypatch, dtype):
    from fastscipy import _ndimage_interp as interp

    monkeypatch.setattr(interp, "_SCIPY_INT_CAST", True)
    x = np.resize(np.array([0.0, -0.0, 1.5, -1.5, 1e30, -1e30, np.nan, np.inf]), (7, 9))
    for order in (0, 1, 3):
        check(lambda: ndi.shift(x, [0.3, -1.2], order=order, output=dtype))
        check(lambda: ndi.affine_transform(x, [[0.9, 0.1], [-0.1, 1.1]], order=order, output=dtype))
        check(lambda: ndi.map_coordinates(x, np.mgrid[:3, :4] + 0.3, order=order, output=dtype))
    check(lambda: ndi.spline_filter1d(x, 3, axis=0, output=dtype))
    check(lambda: ndi.spline_filter1d(x, 3, axis=1, output=np.empty((14, 18), dtype)[::2, ::2]))


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
        size = tuple(
            min(int(rng.integers(2, 6)), n) for n in x.shape
        )  # SciPy corrupts memory when a footprint exceeds the array
        fp = rng.random(size) < 0.7
        fp.flat[rng.integers(fp.size)] = True
        out = None if rng.random() < 0.7 or x.ndim == 1 else pick(rng, DTYPES)  # 1-D + output dtype: SciPy bug
        kw = dict(mode=mode, cval=1.0, output=out)
        f1d = x.ndim > 1 and out is None  # 1-D rank filters go to SciPy's own _rank_filter_1d
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


# --- binary morphology ------------------------------------------------------------------------


def structure_of(rng, ndim, center=True):
    st = rng.random(tuple(int(rng.integers(1, 5)) for _ in range(ndim))) < 0.6
    if center:
        st[tuple(s // 2 for s in st.shape)] = True
    return st


def blobs(rng, shape, p=0.5):
    return ndi.gaussian_filter(rng.random(shape), 1.0) > np.quantile(rng.random(1000), 1 - p) * 0.5 + 0.25


@pytest.mark.parametrize("seed", SEEDS)
def test_binary_morphology(seed):
    rng = np.random.default_rng(600 + seed)
    for _ in range(10):
        ndim = int(rng.integers(1, 4))
        x = blobs(rng, tuple(int(rng.integers(8, 28)) for _ in range(ndim)))
        if rng.random() < 0.3:
            x = x.astype(pick(rng, [np.uint8, np.int32, np.float32]))
        st = structure_of(rng, ndim, center=bool(rng.integers(2)))
        it = int(rng.integers(1, 4))
        mask = rng.random(x.shape) < 0.8 if rng.random() < 0.4 else None
        origin = tuple(int(rng.integers(-(f // 2), (f - 1) // 2 + 1)) for f in st.shape)
        kw = dict(structure=st, iterations=it, mask=mask, border_value=int(rng.integers(2)), origin=origin)
        check(lambda: ndi.binary_erosion(x, **kw))
        check(lambda: ndi.binary_dilation(x, **kw))
        check(lambda: ndi.binary_opening(x, structure=st, iterations=it))
        check(lambda: ndi.binary_closing(x, structure=st, iterations=it))
        out = np.zeros(x.shape, pick(rng, [np.bool_, np.uint8, np.int8, np.int32]))
        check(lambda: ndi.binary_erosion(x, structure=st, output=out.copy()))
        if all(f % 2 for f in st.shape):
            check(lambda: ndi.binary_hit_or_miss(x, st, ~st))


@pytest.mark.parametrize("seed", SEEDS)
def test_binary_fixed_point(seed):
    rng = np.random.default_rng(700 + seed)
    for _ in range(10):
        ndim = int(rng.integers(1, 4))
        x = blobs(rng, tuple(int(rng.integers(8, 40)) for _ in range(ndim)), 0.45)
        st = (
            structure_of(rng, ndim)
            if rng.random() < 0.5
            else ndi.generate_binary_structure(ndim, int(rng.integers(1, ndim + 1)))
        )
        origin = tuple(int(rng.integers(-(f // 2), (f - 1) // 2 + 1)) for f in st.shape)
        if not ndi._morphology._center_is_true(st, list(origin)):
            continue  # SciPy's own fixed-point loop may hang or crash on these
        mask = ndi.binary_dilation(x, iterations=2) if rng.random() < 0.7 else None
        check(lambda: ndi.binary_fill_holes(x, structure=st if st.shape == (3,) * ndim else None))
        seeds = x & (rng.random(x.shape) < 0.1)
        check(lambda: ndi.binary_propagation(seeds, structure=st, mask=mask, origin=origin))
        bv = int(rng.integers(2))
        check(lambda: ndi.binary_dilation(x, structure=st, iterations=-1, mask=mask, border_value=bv))
        check(lambda: ndi.binary_erosion(x, structure=st, iterations=0, mask=mask, origin=origin))


def test_binary_declines():
    x = np.random.default_rng(8).random((12, 12)) > 0.5
    no_center = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)
    check(lambda: ndi.binary_dilation(x, no_center, iterations=2), fast=False)
    check(lambda: ndi.binary_fill_holes(x, axes=(0,)), fast=False)


# --- distance transform, generic_filter -------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_distance_transform_edt(seed):
    rng = np.random.default_rng(800 + seed)
    for _ in range(8):
        shape = tuple(int(rng.integers(2, 30)) for _ in range(int(rng.integers(1, 4))))
        x = array(rng, shape, pick(rng, [np.uint8, np.float32, np.float64, np.int64])) > 0
        sampling = None if rng.random() < 0.5 else [float(rng.uniform(0.2, 3))] * x.ndim
        ri, rd = bool(rng.integers(2)), True
        check(lambda: ndi.distance_transform_edt(x, sampling=sampling, return_indices=ri, return_distances=rd))
        check(lambda: ndi.distance_transform_edt(x != 0, return_distances=False, return_indices=True))
    big = rng.random((300, 200)) > 0.999
    check(lambda: ndi.distance_transform_edt(~big))
    check(lambda: ndi.distance_transform_edt(~big, return_indices=True))


def test_edt_declined_and_errors():
    x = np.random.default_rng(9).random((10, 10)) > 0.5
    d, i = np.empty((10, 10)), np.empty((2, 10, 10), np.int32)
    check(lambda: ndi.distance_transform_edt(x, distances=d), fast=False)
    check(lambda: ndi.distance_transform_edt(x, return_indices=True, indices=i), fast=False)
    with pytest.raises(RuntimeError):
        ndi.distance_transform_edt(x, return_distances=False)


@pytest.mark.parametrize("seed", SEEDS)
def test_generic_filter_reducers(seed):
    rng = np.random.default_rng(900 + seed)
    for _ in range(6):
        x = view_of(
            rng,
            array(rng, shape_of(rng, rng.integers(1, 4), 14, 2), pick(rng, DTYPES + [np.bool_]), rng.random() < 0.2),
        )
        fp = rng.random(tuple(int(rng.integers(1, 5)) for _ in x.shape)) < 0.7
        fp.flat[rng.integers(fp.size)] = True
        origin = tuple(int(rng.integers(-(f // 2), (f - 1) // 2 + 1)) for f in fp.shape)
        mode = pick(rng, MODES)
        for fn in (np.mean, np.sum, np.max, np.min, np.median, np.std, np.var, np.ptp):
            # integer outputs of out-of-range doubles are undefined in C; SciPy's float output is always safe
            check(
                lambda: ndi.generic_filter(x, fn, footprint=fp, mode=mode, cval=1.5, origin=origin, output=np.float64),
                fast=True,
            )
        check(lambda: ndi.generic_filter(x, np.mean, footprint=fp, mode=mode, cval=0.0, origin=origin), fast=False)


# --- labelling and measurements ---------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_label(seed):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(12):
        ndim = int(rng.integers(1, 5))
        x = array(rng, shape_of(rng, ndim, 14), pick(rng, [np.bool_, np.uint8, np.int16, np.float32, np.float64]))
        x = x > 0 if rng.random() < 0.6 else x
        if x.dtype == np.float32 and rng.random() < 0.3:
            x = x * (rng.random(x.shape) < 0.4)
        st = ndi.generate_binary_structure(ndim, int(rng.integers(1, ndim + 1))) if rng.random() < 0.5 else None
        out = pick(rng, [None, np.int32, np.int64, np.uint16, np.int16, np.uint8, np.int8])
        try:
            with fastscipy.disabled():
                ndi.label(x, st, out)
        except Exception:
            continue  # e.g. too many labels for the output dtype: SciPy decides
        check(lambda: ndi.label(x, st, out))
        if rng.random() < 0.3:
            check(lambda: ndi.label(x, st, np.zeros(x.shape, np.int32)))


def test_label_large_and_declined():
    rng = np.random.default_rng(10)
    x = ndi.gaussian_filter(rng.random((400, 300)), 2) > 0.5
    check(lambda: ndi.label(x))
    check(lambda: ndi.label(x, np.ones((3, 3))))
    vol = rng.random((30, 20, 10)) > 0.6
    check(lambda: ndi.label(vol, ndi.generate_binary_structure(3, 3)))
    check(lambda: ndi.label(x[:50, :50], output=np.zeros((50, 50), np.int32, order="F")), fast=False)
    check(lambda: ndi.label(np.zeros((0, 4), bool)), fast=False)


def test_label_structure_truth_values_and_mutation():
    x = np.eye(5, dtype=bool)
    # Bool arrays can contain different nonzero bytes with identical truth values.
    storage = np.array([[0, 2, 0], [1, 255, 3], [0, 7, 0]], dtype=np.uint8)
    structure = storage.view(np.bool_)
    check(lambda: ndi.label(x, structure))
    structure[0, 0] = structure[2, 2] = True
    check(lambda: ndi.label(x, structure))
    structure = structure[::-1, ::-1]
    structure.flags.writeable = False
    check(lambda: ndi.label(x, structure))
    asymmetric = np.eye(3, dtype=bool)
    asymmetric[0, 0] = False
    check(lambda: ndi.label(x, asymmetric), fast=False)


@pytest.mark.parametrize("origin", [[0, -1], (0, -1), np.array([0, -1]), [np.int64(0), np.int64(-1)]])
def test_rank_filter_origin_sequence_types(origin):
    x = np.arange(42, dtype=np.float64).reshape(6, 7)
    footprint = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
    for fn in (ndi.median_filter, ndi.minimum_filter, ndi.maximum_filter):
        check(lambda: fn(x, footprint=footprint, origin=origin))


@pytest.mark.parametrize("shape", [(17,), (7, 9), (3, 5, 7)])
@pytest.mark.parametrize("border", [0, 1])
def test_fixed_point_default_structure_and_outputs(shape, border):
    x = np.zeros(shape, dtype=bool)
    x[(slice(1, -1),) * len(shape)] = True
    x[tuple(s // 2 for s in shape)] = False
    check(lambda: ndi.binary_fill_holes(x))
    for kind in ("default", "int32", "strided"):

        def run():
            output = None
            if kind == "int32":
                output = np.empty(shape, np.int32)
            elif kind == "strided":
                output = np.empty(tuple(2 * s for s in shape), bool)[(slice(None, None, 2),) * len(shape)]
            return ndi.binary_dilation(x, iterations=0, border_value=border, output=output)

        check(run)


@pytest.mark.parametrize("seed", SEEDS)
def test_find_objects_and_watershed(seed):
    rng = np.random.default_rng(1100 + seed)
    for _ in range(8):
        ndim = int(rng.integers(1, 4))
        shape = shape_of(rng, ndim, 20, 2)
        lab = rng.integers(0, 8, shape).astype(pick(rng, [np.int32, np.int64, np.uint8, np.uint16]))
        check(lambda: ndi.find_objects(lab), fast=lab.max() > 0)
        check(lambda: ndi.find_objects(lab, 12))
        img = rng.integers(0, 256, shape).astype(pick(rng, [np.uint8, np.uint16]))
        mk = np.zeros(shape, pick(rng, [np.int8, np.int16, np.int32, np.int64]))
        for k in range(1, 5):
            mk[tuple(int(rng.integers(0, s)) for s in shape)] = k
        mk[tuple(int(rng.integers(0, s)) for s in shape)] = -1
        st = ndi.generate_binary_structure(ndim, int(rng.integers(1, ndim + 1)))
        check(lambda: ndi.watershed_ift(img, mk, st))


@pytest.mark.parametrize("seed", SEEDS)
def test_labelled_measurements(seed):
    rng = np.random.default_rng(1200 + seed)
    for _ in range(8):
        shape = shape_of(rng, rng.integers(1, 4), 24, 2)
        lab = rng.integers(0, int(rng.integers(2, 30)), shape)
        lab = lab.astype(pick(rng, [np.int32, np.int64, np.uint8, np.int16]))
        x = array(rng, shape, pick(rng, DTYPES + [np.bool_]), rng.random() < 0.3)
        if x.dtype.kind == "f":
            x = x + rng.random(shape).astype(x.dtype) * 1e-3  # mostly tie-free
        index = np.arange(0, lab.max() + 3) if rng.random() < 0.5 else np.arange(1, lab.max() + 1)
        index = index.astype(pick(rng, [np.int64, np.int32]))
        for fn in (ndi.minimum, ndi.maximum, ndi.extrema, ndi.minimum_position, ndi.maximum_position):
            check(lambda: fn(x, lab, index), fast=False)  # ties and absent labels: SciPy answers
        check(lambda: ndi.center_of_mass(x.astype(np.float64), lab, index[index > 0]), fast=False)
    big = rng.random((300, 300))
    blob = ndi.label(ndi.gaussian_filter(big, 3) > 0.5)[0]
    ids = np.arange(1, blob.max() + 1)
    for fn in (ndi.minimum, ndi.maximum, ndi.extrema, ndi.minimum_position, ndi.maximum_position, ndi.center_of_mass):
        check(lambda: fn(big, blob, ids))
    check(lambda: ndi.maximum(big, blob, 1), fast=False)
    check(lambda: ndi.median(big, blob, ids), fast=False)


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


@pytest.mark.parametrize(
    "make",
    [
        lambda r: r.standard_normal((20, 20)).astype(np.float16),
        lambda r: r.standard_normal((20, 20)).astype(">f8"),
        lambda r: r.standard_normal((20, 20)) + 1j * r.standard_normal((20, 20)),
        lambda r: r.standard_normal((0, 5)),
        lambda r: r.standard_normal(()),
    ],
    ids=["f16", "bigendian", "complex", "empty", "0d"],
)
def test_declined_inputs(make):
    x = make(np.random.default_rng(5))
    for call in (
        lambda: ndi.gaussian_filter(x, 1.0),
        lambda: ndi.uniform_filter(x, 3),
        lambda: ndi.median_filter(x, 3),
        lambda: ndi.zoom(x, 1.5),
    ):
        try:
            with fastscipy.disabled():
                call()
        except Exception as e:
            with pytest.raises(type(e)):
                call()
        else:
            check(call, fast=False)


def test_generic_callbacks_and_errors():
    x = np.random.default_rng(6).random((20, 20))
    check(lambda: ndi.generic_filter(x, np.ptp, size=3), fast=False)
    for call in (
        lambda: ndi.gaussian_filter(x, 1, mode="bogus"),
        lambda: ndi.uniform_filter(x, 3, origin=5),
        lambda: ndi.median_filter(x, 3, origin=5),
        lambda: ndi.zoom(x, 2, order=7),
    ):
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


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("dtype", [np.int16, np.uint16])
@pytest.mark.parametrize("shape", [(73,), (17, 19), (5, 7, 9)])
def test_center_of_mass_two_byte_integers(dtype, shape):
    rng = np.random.default_rng(331)
    x = rng.integers(1, 300, shape, dtype=dtype)
    labels = np.arange(x.size).reshape(shape) % 7 + 1
    index = np.array([7, 2, 1, 3, 3, 6, 5, 4])
    for negative in (False, True) if dtype is np.int16 else (False,):
        if negative:
            x = -x
        for values, labs in ((x, labels), (x[..., ::-1], labels[..., ::-1]), (np.asfortranarray(x), labels)):
            values.flags.writeable = False
            before = values.tobytes()
            check(lambda: ndi.center_of_mass(values, labs, index))
            assert values.tobytes() == before


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("dtype", [np.int16, np.uint16])
def test_center_of_mass_zero_mass_handler_once(dtype):
    calls = []

    def handler(*args):
        calls.append(args)
        raise RuntimeError("zero mass handler")

    old = np.seterrcall(handler)
    try:
        with np.errstate(invalid="call"):
            with pytest.raises(RuntimeError, match="zero mass handler"):
                ndi.center_of_mass(np.zeros((3, 3), dtype=dtype), np.ones((3, 3), int), [1])
    finally:
        np.seterrcall(old)
    assert len(calls) == 1
