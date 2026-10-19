import warnings

import numpy as np
import pytest
from scipy.cluster import hierarchy, vq

import fastscipy
from fastscipy.testing import assert_identical, check


@pytest.fixture(scope="module", autouse=True)
def cluster_patched():
    was_on = fastscipy.status().get("cluster") == "on"
    assert fastscipy.patch(["cluster"])["cluster"] == "on"
    yield
    if not was_on:
        fastscipy.unpatch(["cluster"])


@pytest.fixture(params=[1, 2], autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield
    fastscipy.set_num_threads(old)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
@pytest.mark.parametrize("check_finite", [True, False])
def test_vq(dtype, layout, check_finite):
    rng = np.random.default_rng(10)
    for shape in ((31,), (23, 1), (19, 7)):
        obs = rng.standard_normal(shape).astype(dtype)
        if layout == "F":
            obs = np.asfortranarray(obs)
        elif layout == "strided":
            obs = np.repeat(obs, 2, axis=0)[::2]
        elif layout == "reversed":
            obs = obs[::-1]
        book = obs[::4].copy()
        book[-1] = book[0]  # Duplicate centroids exercise tie ordering.
        obs.flags.writeable = book.flags.writeable = False
        check(lambda: vq.vq(obs, book, check_finite=check_finite))


def test_vq_declines():
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    for obs, book in (
        (x.astype(np.int32), x[:3].astype(np.int32)),
        (x.astype(np.float32), x[:3]),
        (x.tolist(), x[:3]),
        (x.astype(">f8"), x[:3].astype(">f8")),
        (x.astype(np.float16), x[:3].astype(np.float16)),
        (x, x[:3, :1]),
        (x.reshape(2, 2, 5), x.reshape(2, 2, 5)),
        (x[:0], x[:3]),
    ):
        check(lambda: vq.vq(obs, book), fast=False)
    for value in (np.nan, np.inf, -np.inf):
        bad = x.copy()
        bad[0, 0] = value
        check(lambda: vq.vq(bad, x[:3]), fast=False)
        check(lambda: vq.vq(x, bad[:3]), fast=False)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_whiten(dtype, layout):
    rng = np.random.default_rng(11)
    for _ in range(8):
        x = rng.standard_normal((int(rng.integers(2, 40)), int(rng.integers(1, 10)))).astype(dtype)
        if layout == "F":
            x = np.asfortranarray(x)
        elif layout == "strided":
            x = np.repeat(x, 2, axis=0)[::2]
        x.flags.writeable = False
        for finite in (None, True, False):
            check(lambda: vq.whiten(x, check_finite=finite))
    zero_columns = np.ones((8, 3), dtype=dtype)
    check(lambda: vq.whiten(zero_columns, check_finite=False))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(31,), (19, 3, 4)])
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_whiten_nd_and_complex(dtype, shape, layout):
    rng = np.random.default_rng(850)
    x = rng.standard_normal(shape).astype(dtype)
    if np.iscomplexobj(x):
        x += 1j * rng.standard_normal(shape)
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reversed":
        x = x[::-1]
    x.flags.writeable = False
    before = x.tobytes()
    for finite in (None, True, False):
        check(lambda: vq.whiten(x, check_finite=finite))
    assert x.tobytes() == before
    zero = np.zeros(shape, dtype=dtype)
    check(lambda: vq.whiten(zero, check_finite=False))
    for value in (np.nan, np.inf, -np.inf, 1e30, 1e-30):
        a = x.copy()
        a.flat[0] = value
        for finite in (None, False):
            for err in ({}, {"all": "warn"}, {"all": "raise"}):
                with np.errstate(**err):
                    with fastscipy.disabled():
                        expected = _outcome(lambda: vq.whiten(a, check_finite=finite))
                    assert_identical(expected, _outcome(lambda: vq.whiten(a, check_finite=finite)))


def _outcome(call):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            value, error = call(), None
        except Exception as exc:
            value, error = None, (type(exc), str(exc))
    return value, error, [(type(w.message), str(w.message)) for w in seen]


def test_whiten_diagnostics():
    for x, finite in (
        (np.ones((4, 3)), None),
        (np.ones((4, 3)), True),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), True),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), False),
        (np.array([[1e308, 2.0], [-1e308, 3.0]]), True),
        (np.empty((0, 2)), False),
        (np.ones((1, 2)), False),
        (np.arange(8).reshape(4, 2), True),
    ):
        with fastscipy.disabled():
            expected = _outcome(lambda: vq.whiten(x, check_finite=finite))
        assert_identical(expected, _outcome(lambda: vq.whiten(x, check_finite=finite)))
    with np.errstate(over="raise"):
        check(lambda: vq.whiten(np.array([[1e308], [-1e308]])), fast=False)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_linkage_validation_and_workflows(seed, method):
    x = np.random.default_rng(seed).standard_normal((20, 4))
    z = hierarchy.linkage(x, method=method)
    for linkage in (z, np.asfortranarray(z), np.repeat(z, 2, axis=0)[::2]):
        linkage.flags.writeable = False
        check(lambda: hierarchy.is_valid_linkage(linkage))
        check(lambda: hierarchy.num_obs_linkage(linkage))
        check(lambda: hierarchy.leaves_list(linkage))
        check(lambda: hierarchy.is_monotonic(linkage))
        check(lambda: hierarchy.fcluster(linkage, 3, criterion="maxclust"))
        check(lambda: hierarchy.cophenet(linkage))


def test_linkage_unusual_and_diagnostics():
    base = np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 3.0]])
    unusual = [np.empty((0, 4)), base.astype(np.float32), base[:, :3], base.ravel()]
    for column, value in (
        (0, -1),
        (1, 9),
        (1, 1e308),
        (1, float(2**63)),
        (2, -1),
        (3, 9),
        (0, np.nan),
        (2, np.inf),
        (0, 0.5),
    ):
        z = base.copy()
        z[0, column] = value
        unusual.append(z)
    repeated = base.copy()
    repeated[1, 0] = repeated[0, 0]
    unusual.append(repeated)
    for z in unusual:
        for warning, throw in ((False, False), (True, False), (True, True)):
            call = lambda: hierarchy.is_valid_linkage(z, warning=warning, throw=throw, name="test")
            with fastscipy.disabled():
                expected = _outcome(call)
            assert_identical(expected, _outcome(call))
    # SciPy deliberately accepts all contents when there is only one merge.
    check(lambda: hierarchy.is_valid_linkage(np.full((1, 4), np.nan)))


def test_kmeans_workflows():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((60, 4))
    guess = x[[1, 10, 30]].copy()
    check(lambda: vq.kmeans(x, guess, iter=1))
    check(lambda: vq.kmeans2(x, guess, minit="matrix", iter=8, rng=12))
