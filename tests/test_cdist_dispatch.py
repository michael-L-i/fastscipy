"""Compatibility coverage for small cdist calls and future dispatch experiments."""

import warnings

import numpy as np
import pytest
import scipy.spatial.distance as dist

import fastscipy
from fastscipy import _patch, _spatial
from fastscipy.testing import assert_identical, check


@pytest.fixture(params=[1, 2], ids=["1t", "2t"], autouse=True)
def threads(request):
    previous = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield
    fastscipy.set_num_threads(previous)


@pytest.mark.parametrize("metric", sorted(_spatial._DIRECT_PDIST))
@pytest.mark.parametrize("shape", [(0, 3, 2), (2, 0, 3), (4, 3, 0), (7, 5, 3), (32, 16, 16)])
def test_small_cdist_metrics(metric, shape):
    rows_a, rows_b, dims = shape
    rng = np.random.default_rng(926)
    a, b = rng.normal(size=(rows_a, dims)), rng.normal(size=(rows_b, dims))
    a[::3] = 0
    b[::4] = 0
    check(lambda: dist.cdist(a, b, metric), fast=False)
    check(lambda: dist.cdist(a, b, metric=metric), fast=False)


@pytest.mark.parametrize("metric", sorted(_spatial._DIRECT_PDIST))
@pytest.mark.parametrize("layout", ["C", "F", "reverse", "stride", "unaligned"])
def test_small_cdist_layout_nonfinite(metric, layout):
    rng = np.random.default_rng(927)
    a, b = rng.normal(size=(16, 6)), rng.normal(size=(9, 6))
    a[0] = [0.0, -0.0, np.nan, np.inf, -np.inf, 1e308]
    b[0] = [-0.0, 0.0, np.inf, np.nan, 1e308, -np.inf]
    a.view(np.uint64)[1, 0] = 0x7FF0000000000001
    b.view(np.uint64)[1, 0] = 0x7FF0000000000002

    def prepare(x):
        if layout == "F":
            x = np.asfortranarray(x)
        elif layout == "reverse":
            x = x[::-1]
        elif layout == "stride":
            x = x[:, ::2]
        elif layout == "unaligned":
            unaligned = np.ndarray(x.shape, dtype=np.float64, buffer=bytearray(x.nbytes + 1), offset=1)
            unaligned[:] = x
            x = unaligned
        x.flags.writeable = False
        return x

    a, b = prepare(a), prepare(b)
    before = a.tobytes(), b.tobytes()

    def outcome(call):
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="warn"):
            warnings.simplefilter("always")
            result = call(a, b, metric)
        return result, [(w.category, str(w.message)) for w in caught]

    assert_identical(outcome(dist.cdist.__wrapped__), outcome(dist.cdist))
    assert (a.tobytes(), b.tobytes()) == before
    check(lambda: dist.cdist(a, b, metric), fast=False)


def test_small_cdist_independent_properties():
    a = np.array([[0.0, 0.0], [3.0, 0.0]])
    b = np.array([[3.0, 4.0], [0.0, 4.0], [0.0, 0.0]])
    assert_identical(dist.cdist(a, b), np.array([[5.0, 4.0, 0.0], [4.0, 5.0, 3.0]]))
    assert_identical(dist.cdist(a, b), dist.cdist(b, a).T.copy())
    assert_identical(dist.cdist(a, b, "sqeuclidean"), dist.cdist(a, b) ** 2)
    calls = []

    def metric(x, y):
        calls.append((x.copy(), y.copy()))
        return 2.0

    assert_identical(dist.cdist(a, b, metric), np.full((2, 3), 2.0))
    assert len(calls) == 6
    for (x, y), (i, j) in zip(calls, ((i, j) for i in range(2) for j in range(3))):
        assert_identical(x, a[i])
        assert_identical(y, b[j])


def test_small_cdist_declines_inputs_and_options():
    a, b = np.arange(24.0).reshape(8, 3), np.arange(12.0).reshape(4, 3)
    for value in (a.tolist(), a.astype(np.float32), a.astype(np.int64), a.astype(">f8"), a.view(np.ma.MaskedArray)):
        for x, y in ((value, b), (b, value)):
            assert dist.cdist.fast(x, y) is NotImplemented
            check(lambda: dist.cdist(x, y), fast=False)
    for args, kw in (
        (("EUCLIDEAN",), {}),
        (("eu",), {}),
        (("euclidean",), {"metric": "cityblock"}),
        (("euclidean", "extra"), {}),
        ((), {"out": np.empty((8, 4))}),
        ((), {"w": np.ones(3)}),
        (("minkowski",), {"p": 3.0}),
        (("seuclidean",), {}),
        (("mahalanobis",), {}),
        ((), {"unexpected": 1}),
    ):
        assert dist.cdist.fast(a, b, *args, **kw) is NotImplemented
        check(lambda: dist.cdist(a, b, *args, **kw), fast=False)
    for value in (a[0], np.zeros((1, 2, 3)), a[:, :2]):
        for x, y in ((value, b), (b, value)):
            assert dist.cdist.fast(x, y) is NotImplemented
            check(lambda: dist.cdist(x, y), fast=False)
    for out in (np.empty((8, 4)), np.empty((8, 3)), np.empty((8, 4), dtype=np.float32), np.empty((8, 8))[:, ::2]):
        check(lambda: dist.cdist(a, b, out=out), fast=False)
    out = np.empty((8, 4))
    assert dist.cdist(a, b, out=out) is out
    assert_identical(out, dist.cdist.__wrapped__(a, b))


def test_small_cdist_disabled_and_restored():
    a, b = np.ones((4, 2)), np.ones((8, 2))
    before = _patch.counts["cdist", "fast"]
    with fastscipy.disabled():
        dist.cdist(a, b)
    assert _patch.counts["cdist", "fast"] == before
    original = dist.cdist.__wrapped__
    fastscipy.unpatch(["spatial"])
    try:
        assert dist.cdist is original
    finally:
        fastscipy.patch(["spatial"])
    check(lambda: dist.cdist(a, b), fast=False)
