"""Keep cheap two-thread distances on SciPy before allocating tile storage."""

import numpy as np
import pytest
import scipy.spatial.distance as dist

import fastscipy
from fastscipy import _patch, _spatial
from fastscipy.testing import check


@pytest.fixture
def two_threads():
    previous = fastscipy.get_num_threads()
    fastscipy.set_num_threads(2)
    try:
        yield
    finally:
        fastscipy.set_num_threads(previous)


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "cityblock"])
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "float32"])
def test_low_dimensional_two_thread_pdist_declines(metric, dims, layout, two_threads):
    x = np.random.default_rng(527).normal(size=(1500, dims * 2))
    x = x[:, ::2] if layout == "strided" else x[:, :dims].copy()
    if layout == "float32":
        x = x.astype(np.float32)
    before = x.tobytes()
    assert dist.pdist.fast(x, metric) is NotImplemented
    check(lambda: dist.pdist(x, metric), fast=False)
    check(lambda: dist.pdist(x, metric=metric), fast=False)
    assert x.tobytes() == before


def test_low_dimensional_decline_precedes_preparation(two_threads, monkeypatch):
    x = np.ones((1500, 3), dtype=np.float32)

    def forbidden(*args, **kwargs):
        pytest.fail("declined problem reached tiled distance preparation")

    fast = _spatial._pdist_fast(forbidden)
    monkeypatch.setattr(_spatial.np, "ascontiguousarray", forbidden)
    monkeypatch.setattr(_spatial, "_parameters", forbidden)
    assert fast(x) is NotImplemented


def test_low_dimensional_decline_calls_original_once(two_threads):
    x = np.ones((1500, 3))
    calls = []
    error = ValueError("original error")

    def original(value, **kwargs):
        assert value is x and kwargs == {"metric": "cityblock"}
        calls.append(value)
        raise error

    wrapper = _patch.wrap(original, _spatial._pdist_fast(dist.cdist.__wrapped__), "dispatch_original_error")
    before = _patch.counts["dispatch_original_error", "fast"]
    with pytest.raises(ValueError) as raised:
        wrapper(x, metric="cityblock")
    assert raised.value is error
    assert len(calls) == 1
    assert _patch.counts["dispatch_original_error", "fast"] == before


@pytest.mark.parametrize("scenario", ["small", "zero_dims", "four_threads", "weighted", "expensive", "four_dims"])
def test_other_distance_paths_remain_available(scenario, two_threads):
    x = np.random.default_rng(528).random((1500, 4 if scenario == "four_dims" else 3))
    options = {}
    metric = "euclidean"
    if scenario == "small":
        x = x[:16]
    elif scenario == "zero_dims":
        x = x[:, :0].copy()
    elif scenario == "four_threads":
        fastscipy.set_num_threads(4)
    elif scenario == "weighted":
        options["w"] = np.array([1.0, 0.5, 2.0])
    elif scenario == "expensive":
        metric, options = "minkowski", {"p": 3.5}
    check(lambda: dist.pdist(x, metric, **options))


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "cityblock"])
def test_small_nonfinite_inputs_keep_direct_path(metric, two_threads):
    x = np.array([[0.0, -0.0], [np.nan, 1.0], [np.inf, -np.inf]])
    check(lambda: dist.pdist(x, metric))


def test_low_dimensional_nonfinite_inputs_delegate(two_threads):
    x = np.resize(np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0]), (1500, 3))
    check(lambda: dist.pdist(x), fast=False)
