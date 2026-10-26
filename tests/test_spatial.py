import contextlib
import pickle

import numpy as np
import pytest
import scipy.spatial as sp
import scipy.spatial.distance as dist

import fastscipy
from fastscipy import _patch, _spatial
from fastscipy.testing import assert_identical, check

RNG = np.random.default_rng(0)


@pytest.fixture(params=[1, 4], ids=["1t", "4t"], autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield request.param
    fastscipy.set_num_threads(old)


def data(m, d, round_some=True):
    a = RNG.random((m, d))
    if round_some:
        a[::7] = np.round(a[::7])  # exact zeros and ones exercise the boolean metrics
    return a


@pytest.mark.parametrize("metric", sorted(_spatial._CDIST))
@pytest.mark.parametrize("d", [3, 16, 17])
def test_cdist(metric, d, threads):
    A, B = data(900, d), data(700, d)
    check(lambda: dist.cdist(A, B, metric), fast=threads > 1)


@pytest.mark.parametrize("metric", sorted(_spatial._PDIST))
@pytest.mark.parametrize("d", [3, 16, 17])
def test_pdist(metric, d, threads):
    X = data(1500, d)
    check(lambda: dist.pdist(X, metric), fast=threads > 1)


@pytest.mark.parametrize("threads", [2], indirect=True)
@pytest.mark.parametrize("metric", sorted(_spatial._PDIST))
@pytest.mark.parametrize("tile_entries", [19, 512])
def test_pdist_tiles(metric, tile_entries, threads, monkeypatch):
    # Exercise both a tile smaller than one row and multirow tiles, including
    # a final partial tile. The spy checks the actual SciPy temporary shapes.
    monkeypatch.setattr(_spatial, "_WORK", 1)
    monkeypatch.setattr(_spatial, "_BLOCK", 100_000)
    monkeypatch.setattr(_spatial, "_PDIST_TILE", tile_entries)
    x = np.random.default_rng(845).random((53, 7))
    x[::7] = np.round(x[::7])
    original = dist.cdist.__wrapped__
    shapes = []
    parameters = []

    def cdist_spy(a, b, *args, **kwargs):
        shapes.append((len(a), len(b)))
        parameters.append(kwargs)
        return original(a, b, *args, **kwargs)

    result = _spatial._pdist_fast(cdist_spy)(x, metric)
    with fastscipy.disabled():
        expected = dist.pdist(x, metric)
    assert_identical(expected, result)
    assert len(shapes) > 2
    assert all(rows * cols <= max(tile_entries, len(x)) for rows, cols in shapes)
    assert sum(rows for rows, _ in shapes) == len(x) - 1
    if metric in _spatial._STATISTICS:
        key = _spatial._STATISTICS[metric]
        assert all(p[key] is parameters[0][key] for p in parameters)


def test_pdist_shapes_and_default_metric(threads):
    data_wide = data(5000, 400)
    for m in (1000, 1001, 1500):
        X = data(m, 8)
        check(lambda: dist.pdist(X), fast=threads > 1)
    tall, few, wide = data(5000, 4), data(3, 4), data(3, 400)
    check(lambda: dist.cdist(tall, few), fast=False)
    check(lambda: dist.cdist(wide, data_wide), fast=False)  # 3 rows: nothing to split


def test_declined_still_correct():
    A, B = data(900, 16), data(700, 16)
    out = np.empty((900, 700))
    r = dist.cdist(A, B, out=out)
    assert r is out and np.array_equal(out, dist.cdist.__wrapped__(A, B))
    for call in (
        lambda: dist.cdist(A[:5], B[:5], lambda u, v: 1.0),
        lambda: dist.cdist(A.astype(np.int32), B),
        lambda: dist.cdist(A.T, A.T),
        lambda: dist.cdist(A[:10], B[:10]),
        lambda: dist.cdist(A.tolist(), B.tolist()),
        lambda: dist.pdist(A, "dice"),
        lambda: dist.pdist(A[:20]),
    ):
        check(call, fast=False)


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("metric", sorted(_spatial._WEIGHTED))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_distances(metric, dtype, threads, monkeypatch):
    monkeypatch.setattr(_spatial, "_WORK", 1000)
    monkeypatch.setattr(_spatial, "_BLOCK", 2000)
    rng = np.random.default_rng(820)
    a, b = rng.random((97, 7)).astype(dtype), rng.random((81, 7)).astype(dtype)
    a[::7], b[::7] = np.round(a[::7]), np.round(b[::7])
    w = rng.random(14)[::2]
    w[0] = -0.0
    w.flags.writeable = False
    for aa, bb in ((a, b), (a[::-1], np.asfortranarray(b))):
        for ww in (None, w, np.zeros(7)):
            before = aa.tobytes(), bb.tobytes(), None if ww is None else ww.tobytes()
            check(lambda: dist.cdist(aa, bb, metric, w=ww), fast=threads > 1)
            check(lambda: dist.pdist(aa, metric, w=ww), fast=threads > 1 and metric in _spatial._PDIST)
            assert before == (aa.tobytes(), bb.tobytes(), None if ww is None else ww.tobytes())


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("metric", sorted(_spatial._CDIST))
def test_float32_distances(metric, threads, monkeypatch):
    monkeypatch.setattr(_spatial, "_WORK", 1000)
    monkeypatch.setattr(_spatial, "_BLOCK", 2000)
    rng = np.random.default_rng(821)
    a, b = rng.random((97, 7)).astype(np.float32), rng.random((81, 7)).astype(np.float32)
    a[0, 0], b[0, 0] = -0.0, np.nan
    for aa, bb in ((a, b), (np.asfortranarray(a), b[::-1])):
        check(lambda: dist.cdist(aa, bb, metric), fast=threads > 1 and metric not in _spatial._STATISTICS)
        check(lambda: dist.pdist(aa, metric), fast=threads > 1 and metric in _spatial._PDIST)


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("p", [0.5, 1, 2, 3.5, np.inf])
def test_minkowski_options(p, threads, monkeypatch):
    monkeypatch.setattr(_spatial, "_WORK", 1000)
    monkeypatch.setattr(_spatial, "_BLOCK", 2000)
    rng = np.random.default_rng(822)
    a, b = rng.normal(size=(97, 7)), rng.normal(size=(81, 7))
    for options in ({"p": p}, {"p": p, "w": rng.random(7)}):
        check(lambda: dist.cdist(a, b, "minkowski", **options), fast=threads > 1)
        check(lambda: dist.pdist(a, "minkowski", **options), fast=threads > 1)


def test_distance_options_invalid(monkeypatch):
    monkeypatch.setattr(_spatial, "_WORK", 1000)
    monkeypatch.setattr(_spatial, "_BLOCK", 2000)
    a, b = np.ones((97, 7)), np.ones((81, 7))
    for w in ([1.0] * 7, np.ones(8), np.ones(7, dtype=np.float32), np.full(7, -1.0), np.full(7, np.inf)):
        check(lambda: dist.cdist(a, b, "euclidean", w=w), fast=False)
        check(lambda: dist.pdist(a, "euclidean", w=w), fast=False)
    for p in (0, -1, np.nan, "invalid", np.array([2.0])):
        check(lambda: dist.cdist(a, b, "minkowski", p=p), fast=False)
        check(lambda: dist.pdist(a, "minkowski", p=p), fast=False)


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_float32_cast_warning_events(threads, monkeypatch):
    import warnings

    from fastscipy import _patch

    monkeypatch.setattr(_spatial, "_WORK", 1000)
    monkeypatch.setattr(_spatial, "_BLOCK", 2000)
    monkeypatch.setattr(_patch, "STRICT", False)
    a = np.ones((97, 7), dtype=np.float32)
    a.view(np.uint32)[0, 0] = 0x7F800001
    b = np.zeros((81, 7), dtype=np.float32)
    before = a.tobytes()
    for call in (lambda: dist.cdist(a, b), lambda: dist.pdist(a)):

        def outcome():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = call()
            return result, [(w.category, str(w.message)) for w in caught]

        with fastscipy.disabled():
            expected = outcome()
        assert_identical(expected, outcome())
        for enabled in (False, True):
            events = []

            def handler(error, flag):
                events.append((error, flag))
                raise RuntimeError("cast callback")

            context = fastscipy.disabled() if not enabled else contextlib.nullcontext()
            with context, np.errstate(invalid="call", call=handler):
                with pytest.raises(RuntimeError, match="cast callback"):
                    call()
            assert len(events) == 1
    assert a.tobytes() == before


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("metric,key", [("seuclidean", "V"), ("mahalanobis", "VI")])
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_distance_statistics(metric, key, layout, threads):
    rng = np.random.default_rng(812)
    a, b = rng.normal(size=(850, 7)), rng.normal(size=(650, 7))
    if layout == "F":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif layout == "reversed":
        a, b = a[::-1], b[::-1]
    v = np.arange(1, 8, dtype=np.float64)
    supplied = v if key == "V" else np.diag(v)
    supplied.flags.writeable = False
    exceptional = supplied.copy()
    exceptional.flat[0] = np.nan
    for options in ({}, {key: None}, {key: supplied}, {key: supplied[::-1]}, {key: exceptional}):
        check(lambda: dist.cdist(a, b, metric=metric, **options), fast=threads > 1)
        check(lambda: dist.pdist(a, metric, **options), fast=threads > 1)


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_distance_statistics_declined(threads):
    import warnings

    rng = np.random.default_rng(813)
    a, b = rng.random((850, 7)), rng.random((650, 7))
    for metric, key in (("seuclidean", "V"), ("mahalanobis", "VI")):
        supplied = np.ones(7) if key == "V" else np.eye(7)
        for value in (supplied.tolist(), supplied.astype(np.float32)):
            assert dist.cdist.fast(a, b, metric, **{key: value}) is NotImplemented
            assert dist.pdist.fast(a, metric, **{key: value}) is NotImplemented
            check(lambda: dist.cdist(a, b, metric, **{key: value}), fast=False)
            check(lambda: dist.pdist(a, metric, **{key: value}), fast=False)
        check(lambda: dist.cdist(a, b, metric, unexpected=True), fast=False)
    # SciPy validates V's shape, but does not validate VI before its native loop.
    check(lambda: dist.cdist(a, b, "seuclidean", V=np.ones((7, 7, 1))), fast=False)
    singular = np.ones((850, 7))
    check(lambda: dist.pdist(singular, "mahalanobis"), fast=False)
    for bad in (np.nan, np.inf, 1e308):
        aa = a.copy()
        aa[0] = bad
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check(lambda: dist.cdist(aa, b, "seuclidean"), fast=False)


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("metric,key", [("seuclidean", "V"), ("mahalanobis", "VI")])
@pytest.mark.parametrize("kind", ["list", "float32", "rank3"])
@pytest.mark.parametrize("name", ["cdist", "pdist"])
def test_distance_statistics_decline_delegation(metric, key, kind, name, threads):
    # SciPy 1.18.1 reads n*n entries from VI without checking its shape
    # (spatial/src/distance_impl.h::mahalanobis_distance). A short VI is
    # unsafe even in check()'s unpatched reference call. Test our guards and
    # untouched delegation with spies, never passing malformed VI to native C.
    a, b = np.ones((850, 7)), np.zeros((650, 7))
    value = {"list": [1.0] * 7, "float32": np.ones(7, dtype=np.float32), "rank3": np.ones((7, 7, 1))}[kind]
    arrays = (a, b) if name == "cdist" else (a,)
    args, kw = (*arrays, metric), {key: value}

    def forbidden_kernel(*args, **kw):
        pytest.fail("unsupported statistics reached the distance kernel")

    factory = _spatial._cdist_fast if name == "cdist" else _spatial._pdist_fast
    fast = factory(forbidden_kernel)
    assert fast(*args, **kw) is NotImplemented
    sentinel, calls = object(), []

    def original_spy(*args, **kw):
        calls.append((args, kw))
        return sentinel

    wrapper = _patch.wrap(original_spy, fast, "statistics_decline_spy")
    assert wrapper(*args, **kw) is sentinel
    assert len(calls) == 1
    received_args, received_kw = calls[0]
    assert len(received_args) == len(args)
    assert all(received is original for received, original in zip(received_args, args))
    assert received_kw.keys() == kw.keys() and received_kw[key] is value


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("metric", sorted(_spatial._CDIST))
def test_distance_noncontiguous(metric, threads):
    rng = np.random.default_rng(810)
    A, B = rng.random((800, 16)), rng.random((600, 16))
    A[::7] = np.round(A[::7])
    for a, b in ((np.asfortranarray(A), np.asfortranarray(B)), (A[::-1], B[::-1]), (A[:, ::2], B[:, ::2])):
        before_a, before_b = a.tobytes(), b.tobytes()
        check(lambda: dist.cdist(a, b, metric), fast=threads > 1)
        check(lambda: dist.pdist(a, metric), fast=threads > 1 and metric in _spatial._PDIST)
        assert a.tobytes() == before_a and b.tobytes() == before_b


@pytest.fixture(scope="module")
def tree():
    return sp.cKDTree(RNG.random((50_000, 3))), sp.KDTree(RNG.random((50_000, 3)))


@pytest.mark.parametrize("k", [1, 8, [1, 3]])
def test_query(tree, k, threads):
    q = RNG.random((20_000, 3))
    for t in tree:
        check(lambda: t.query(q, k), fast=threads > 1)
        check(lambda: t.query(q, k, 0.1, 1.0), fast=threads > 1)
        check(lambda: t.query(q, k, distance_upper_bound=0.05), fast=threads > 1)


def test_query_ball_point(tree, threads):
    q = RNG.random((20_000, 3))
    for t in tree:
        check(lambda: t.query_ball_point(q, 0.03).tolist(), fast=threads > 1)
        check(lambda: t.query_ball_point(q, 0.03, return_length=True), fast=threads > 1)
        check(lambda: t.query_ball_point(q, 0.03, return_sorted=False).tolist(), fast=threads > 1)


def test_tree_declined_still_correct(tree):
    q = RNG.random((20_000, 3))
    for t in tree:
        for call in (
            lambda: t.query(q, 2, workers=2),
            lambda: t.query(q, 2, 0.0, 2.0, np.inf, 1),
            lambda: t.query(q[:100], 2),
            lambda: t.query(q[0], 2),
            lambda: t.query_ball_point(q, 0.02, workers=-1).tolist(),
            lambda: t.query_ball_point(q[0], 0.1),
        ):
            check(call, fast=False)


def test_tree_types_and_pickle(tree):
    c, k = tree
    assert isinstance(c, sp.cKDTree) and isinstance(sp.cKDTree(RNG.random((10, 2))), sp.cKDTree)
    c2 = pickle.loads(pickle.dumps(c))
    q = RNG.random((50, 3))
    assert np.array_equal(c2.query(q)[1], c.query(q)[1])
    assert isinstance(k, sp.cKDTree)


def test_unpatch_restores():
    fastscipy.unpatch(["spatial"])
    assert not hasattr(dist.cdist, "fast") and not hasattr(sp.cKDTree.query, "fast")
    fastscipy.patch(["spatial"])
    assert hasattr(dist.cdist, "fast")
