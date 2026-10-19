import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as cg

import fastscipy
from fastscipy.testing import assert_identical, check

THREADS = [1, 4]


@pytest.fixture(params=THREADS, autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield request.param
    fastscipy.set_num_threads(old)


def graph(seed, n=60, density=0.08, ints=True, negative=False, fmt="csr", sort=True):
    rng = np.random.default_rng(seed)
    m = max(int(n * n * density), 1)
    r, c = rng.integers(0, n, m), rng.integers(0, n, m)
    w = rng.integers(1, 6, m).astype(float) if ints else rng.random(m) * 10 + 0.1  # integers give many ties
    if negative:  # a random potential keeps cycles non-negative
        h = rng.integers(0, 5, n)
        w = w + h[r] - h[c]
    g = sp.coo_array((w, (r, c)), (n, n)).tocsr()
    if not sort:
        g.has_sorted_indices = False
        for i in range(n):
            lo, hi = g.indptr[i], g.indptr[i + 1]
            g.indices[lo:hi] = g.indices[lo:hi][::-1]
            g.data[lo:hi] = g.data[lo:hi][::-1]
    return g.asformat(fmt)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("ints", [True, False])
def test_dijkstra(seed, directed, ints):
    g = graph(seed, ints=ints, sort=seed % 2 == 0)
    for kw in (
        {},
        {"return_predecessors": True},
        {"unweighted": True, "return_predecessors": True},
        {"indices": [3, 7, 3], "return_predecessors": True},
        {"indices": -1},
        {"indices": [[1, 2], [3, 4]]},
        {"limit": 4.0, "return_predecessors": True},
        {"unweighted": True, "limit": 3, "indices": [0, 5]},
        {"indices": [0, 9, 17], "min_only": True, "return_predecessors": True},
        {"indices": [4, 2], "min_only": True, "unweighted": True, "return_predecessors": True},
        {"indices": [], "return_predecessors": True},
    ):
        check(lambda: cg.dijkstra(g, directed, **kw))


@pytest.mark.parametrize("fmt", ["csc", "coo", "lil", "dia", "bsr"])
def test_input_formats(fmt):
    g = graph(1, fmt=fmt)
    for f in (cg.dijkstra, cg.bellman_ford, cg.johnson):
        check(lambda: f(g, directed=False, return_predecessors=True))
    check(lambda: cg.connected_components(g, directed=True, connection="strong"))
    d = graph(1).toarray()
    d[d == 0] = np.inf
    check(lambda: cg.dijkstra(d, indices=[1, 2]))
    check(lambda: cg.connected_components(graph(1).astype(np.int8)))
    check(lambda: cg.connected_components(graph(1).toarray()))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("directed", [True, False])
def test_bellman_ford_johnson(seed, directed):
    for neg in (False, True):
        g = graph(
            seed, negative=neg, ints=seed % 2 == 0 or not directed, density=0.05
        )  # float undirected rounding trips SciPy's cycle check
        for f in (cg.bellman_ford, cg.johnson):
            for kw in (
                {},
                {"return_predecessors": True},
                {"unweighted": True, "return_predecessors": True},
                {"indices": [0, 11, -2], "return_predecessors": True},
                {"indices": 5},
            ):
                if neg and (
                    not directed or kw.get("unweighted")
                ):  # a negative undirected edge is a cycle; unweighted Dijkstra declines
                    continue
                check(lambda: f(g, directed, **kw))


def test_padded_graph_converges_early():
    g = graph(0, n=40)
    big = sp.csr_array((g.data, g.indices, np.concatenate([g.indptr, np.full(2000, g.indptr[-1])])), (2040, 2040))
    for f in (cg.bellman_ford, cg.johnson):
        check(lambda: f(big, indices=[0, 1], return_predecessors=True))


def test_negative_cycle_and_negative_dijkstra_decline():
    g = sp.csr_array(np.array([[0, 1, 0], [0, 0, -3], [1, 0, 0.0]]))
    for f in (cg.bellman_ford, cg.johnson):
        with pytest.raises(cg.NegativeCycleError) as e:
            f(g)
        with fastscipy.disabled(), pytest.raises(cg.NegativeCycleError) as e2:
            f(g)
        assert str(e.value) == str(e2.value)
    with pytest.warns(UserWarning, match="negative weights"):
        check(lambda: cg.dijkstra(sp.csr_array(np.array([[0, 2, 0], [0, 0, -1], [3, 0, 0.0]]))), fast=False)
    check(lambda: cg.johnson(sp.csr_array(np.array([[0, 2, 0], [0, 0, -1], [3, 0, 0.0]])), return_predecessors=True))


def test_nan_and_inf_weights():
    g = graph(2, ints=False)
    g.data[::7] = np.nan
    g.data[1::11] = np.inf
    check(lambda: cg.dijkstra(g, False, return_predecessors=True))
    check(lambda: cg.bellman_ford(g, True, return_predecessors=True))


@pytest.mark.parametrize("seed", range(4))
def test_shortest_path(seed):
    g = graph(seed, density=0.03)
    for kw in (
        {},
        {"unweighted": True, "indices": 0},
        {"method": "D", "return_predecessors": True, "directed": False},
        {"method": "BF", "indices": [0, 3]},
        {"method": "J", "unweighted": True, "return_predecessors": True},
        {"indices": [1, 2], "directed": False},
    ):
        check(lambda: cg.shortest_path(g, **kw))
    check(lambda: cg.shortest_path(graph(seed, negative=True, density=0.03), indices=[0, 4], return_predecessors=True))
    check(lambda: cg.shortest_path(graph(seed, density=0.5), method="FW"), fast=False)
    check(lambda: cg.shortest_path(graph(seed, density=0.5)), fast=False)


def test_unweighted_grid():
    n = 60
    idx = np.arange(n * n).reshape(n, n)
    r = np.concatenate([idx[:, :-1].ravel(), idx[:-1].ravel()])
    c = np.concatenate([idx[:, 1:].ravel(), idx[1:].ravel()])
    g = sp.csr_array((np.ones(len(r)), (r, c)), (n * n, n * n))
    check(lambda: cg.shortest_path(g, unweighted=True, indices=0, return_predecessors=True))
    check(lambda: cg.shortest_path(g, directed=False, unweighted=True, indices=[7, 1000, 7], return_predecessors=True))


@pytest.mark.parametrize("seed", range(8))
def test_connected_components(seed):
    g = graph(seed, n=200, density=0.004 + 0.002 * seed, sort=seed % 2 == 0)
    for kw in (
        {},
        {"directed": False},
        {"connection": "strong"},
        {"connection": "STRONG", "return_labels": False},
        {"directed": False, "connection": "strong"},
        {"return_labels": False},
    ):
        check(lambda: cg.connected_components(g, **kw))
    check(lambda: cg.connected_components(sp.csr_array((0, 0))), fast=False)
    for gg in (sp.csr_array((5, 5)), sp.csr_array(np.ones((4, 4)))):
        check(lambda: cg.connected_components(gg, connection="strong"))
        check(lambda: cg.connected_components(gg))


def test_connected_components_one_big_cycle():
    n = 5000
    r = np.arange(n)
    g = sp.csr_array((np.ones(n), (r, (r + 1) % n)), (n, n))
    check(lambda: cg.connected_components(g, connection="strong"))
    check(lambda: cg.connected_components(g[::-1, ::-1].tocsr(), connection="strong"))


def test_declined_cases_match_scipy():
    g = graph(0)
    for bad in ({"indices": 999}, {"limit": -1.0}):
        with pytest.raises(ValueError) as e1:
            cg.dijkstra(g, **bad)
        with fastscipy.disabled(), pytest.raises(ValueError) as e2:
            cg.dijkstra(g, **bad)
        assert str(e1.value) == str(e2.value)
    big = g.copy()
    big.indices, big.indptr = big.indices.astype(np.int64), big.indptr.astype(np.int64)
    check(lambda: cg.connected_components(big, connection="strong"))
    check(lambda: cg.dijkstra(big, indices=[0, 1], return_predecessors=True))
    assert_identical(cg.connected_components(sp.csr_matrix((3, 3))), (3, np.arange(3, dtype=np.int32)))
