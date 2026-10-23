"""Small pdist dispatch keeps SciPy's compiled arithmetic and argument contract."""

import warnings

import numpy as np
import pytest
import scipy.spatial.distance as dist

import fastscipy
from fastscipy import _patch, _spatial
from fastscipy.testing import assert_identical, check


@pytest.fixture(params=[1, 2], ids=["1t", "2t"], autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield
    fastscipy.set_num_threads(old)


@pytest.mark.parametrize("metric", sorted(_spatial._DIRECT_PDIST))
@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (4, 0), (7, 3), (32, 16)])
def test_small_pdist_metrics(metric, shape):
    x = np.random.default_rng(903).normal(size=shape)
    x[::3] = 0
    check(lambda: dist.pdist(x, metric))
    check(lambda: dist.pdist(x, metric=metric))


@pytest.mark.parametrize("metric", sorted(_spatial._DIRECT_PDIST))
@pytest.mark.parametrize("layout", ["C", "F", "reverse", "stride", "unaligned"])
def test_small_pdist_layout_nonfinite(metric, layout):
    x = np.random.default_rng(904).normal(size=(16, 6))
    x[0] = [0.0, -0.0, np.nan, np.inf, -np.inf, 1e308]
    # Preserve even signaling NaN payloads through the same SciPy kernel.
    x.view(np.uint64)[1, 0] = 0x7FF0000000000001
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1]
    elif layout == "stride":
        x = x[:, ::2]
    elif layout == "unaligned":
        storage = bytearray(x.nbytes + 1)
        unaligned = np.ndarray(x.shape, dtype=np.float64, buffer=storage, offset=1)
        unaligned[:] = x
        x = unaligned
    x.flags.writeable = False
    before = x.tobytes()
    original = dist.pdist.__wrapped__

    def outcome(call):
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="warn"):
            warnings.simplefilter("always")
            result = call(x, metric)
        return result, [(w.category, str(w.message)) for w in caught]

    assert_identical(outcome(original), outcome(dist.pdist))
    assert x.tobytes() == before
    check(lambda: dist.pdist(x, metric))


def test_small_pdist_independent_properties():
    x = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]])
    assert_identical(dist.pdist(x), np.array([3.0, 5.0, 4.0, 4.0, 5.0, 3.0]))
    square = dist.squareform(dist.pdist(x))
    assert_identical(square, square.T.copy())
    assert np.all(square.diagonal() == 0)
    assert np.all(square[:, :, None] <= square[:, None, :] + square[None, :, :])
    assert_identical(dist.pdist(x, "sqeuclidean"), dist.pdist(x) ** 2)
    # Independently exercise condensed ordering and custom callback count.
    calls = []

    def metric(a, b):
        calls.append((a.copy(), b.copy()))
        return 2.0

    assert_identical(dist.pdist(x, metric), np.full(6, 2.0))
    assert len(calls) == 6
    for (a, b), (i, j) in zip(calls, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
        assert_identical(a, x[i])
        assert_identical(b, x[j])


def test_small_pdist_declines_options_and_foreign_inputs():
    x = np.arange(24.0).reshape(8, 3)
    fast = dist.pdist.fast
    for value in (x.tolist(), x.astype(np.float32), x.astype(np.int64), x.astype(">f8"), x.view(np.ma.MaskedArray)):
        assert fast(value) is NotImplemented
        check(lambda: dist.pdist(value), fast=False)
    for args, kw in (
        (("EUCLIDEAN",), {}),
        (("eu",), {}),
        (("euclidean",), {"metric": "cityblock"}),
        (("euclidean", "extra"), {}),
        ((), {"out": np.empty(28)}),
        ((), {"w": np.ones(3)}),
        (("minkowski",), {"p": 3.0}),
        (("seuclidean",), {}),
        (("mahalanobis",), {}),
        ((), {"unexpected": 1}),
    ):
        assert fast(x, *args, **kw) is NotImplemented
        check(lambda: dist.pdist(x, *args, **kw), fast=False)
    for value in (x[0], np.zeros((1, 2, 3))):
        assert fast(value) is NotImplemented
        check(lambda: dist.pdist(value), fast=False)
    for out in (np.empty(28), np.empty(27), np.empty(28, dtype=np.float32), np.empty(56)[::2]):
        check(lambda: dist.pdist(x, out=out), fast=False)
    out = np.empty(28)
    assert dist.pdist(x, out=out) is out
    assert_identical(out, dist.pdist.__wrapped__(x))


def test_small_pdist_disabled_and_restored():
    x = np.ones((4, 3))
    before = _patch.counts["pdist", "fast"]
    with fastscipy.disabled():
        dist.pdist(x)
    assert _patch.counts["pdist", "fast"] == before
    original = dist.pdist.__wrapped__
    fastscipy.unpatch(["spatial"])
    try:
        assert dist.pdist is original
    finally:
        fastscipy.patch(["spatial"])
    check(lambda: dist.pdist(x))


def test_small_pdist_threshold(monkeypatch):
    monkeypatch.setattr(_spatial, "_WORK", 64)
    x = np.arange(16.0).reshape(8, 2)
    # Exactly at the boundary the existing threaded/fallback path owns the call.
    # One thread makes this test independent of the large-call block scheduler.
    fastscipy.set_num_threads(1)
    assert dist.pdist.fast(x) is NotImplemented
    check(lambda: dist.pdist(x), fast=False)
    check(lambda: dist.pdist(x[:-1]))
