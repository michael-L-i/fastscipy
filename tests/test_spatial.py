import pickle

import numpy as np
import pytest
import scipy.spatial as sp
import scipy.spatial.distance as dist

import fastscipy
from fastscipy import _spatial
from fastscipy.testing import check

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
        a[::7] = np.round(a[::7])    # exact zeros and ones exercise the boolean metrics
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


def test_pdist_shapes_and_default_metric(threads):
    data_wide = data(5000, 400)
    for m in (1000, 1001, 1500):
        X = data(m, 8)
        check(lambda: dist.pdist(X), fast=threads > 1)
    tall, few, wide = data(5000, 4), data(3, 4), data(3, 400)
    check(lambda: dist.cdist(tall, few), fast=False)
    check(lambda: dist.cdist(wide, data_wide), fast=False)   # 3 rows: nothing to split


def test_declined_still_correct():
    A, B = data(900, 16), data(700, 16)
    out = np.empty((900, 700))
    r = dist.cdist(A, B, out=out)
    assert r is out and np.array_equal(out, dist.cdist.__wrapped__(A, B))
    for call in (lambda: dist.cdist(A, B, "mahalanobis"),
                 lambda: dist.cdist(A, B, "seuclidean"),
                 lambda: dist.cdist(A, B, "minkowski", p=3.0),
                 lambda: dist.cdist(A, B, "euclidean", w=np.ones(16)),
                 lambda: dist.cdist(A[:5], B[:5], lambda u, v: 1.0),
                 lambda: dist.cdist(A.astype(np.float32), B),
                 lambda: dist.cdist(A.T, A.T),
                 lambda: dist.cdist(A[::2], B[::2]),
                 lambda: dist.cdist(A[:10], B[:10]),
                 lambda: dist.cdist(A.tolist(), B.tolist()),
                 lambda: dist.pdist(A, "dice"),
                 lambda: dist.pdist(np.asfortranarray(A)),
                 lambda: dist.pdist(A[:20])):
        check(call, fast=False)


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
        for call in (lambda: t.query(q, 2, workers=2), lambda: t.query(q, 2, 0.0, 2.0, np.inf, 1),
                     lambda: t.query(q[:100], 2), lambda: t.query(q[0], 2),
                     lambda: t.query_ball_point(q, 0.02, workers=-1).tolist(),
                     lambda: t.query_ball_point(q[0], 0.1)):
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
