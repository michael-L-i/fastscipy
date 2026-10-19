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


@pytest.mark.parametrize("source", [0, -1, 3, -4, np.int32(0), [0]])
def test_scalar_sources_preserve_output_shape(source):
    g = graph(0, n=4, density=0.5)
    for f in (cg.dijkstra, cg.bellman_ford, cg.johnson):
        check(lambda: f(g, indices=source, return_predecessors=True))


def test_unweighted_transpose_ignores_stored_weights():
    g = sp.csr_array(
        (np.array([np.nan, np.inf, 0.0, 7.0]), (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0]))), shape=(4, 4)
    )
    original = g.data.copy()
    for sources, min_only in ((0, False), ([0, 3], True)):
        check(
            lambda: cg.dijkstra(
                g, directed=False, indices=sources, unweighted=True, min_only=min_only, return_predecessors=True
            )
        )
    assert_identical(original, g.data)


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("cls", [sp.csr_array, sp.csr_matrix])
@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int16, np.int64, np.uint8, np.uint64])
def test_integer_unit_graph_ignores_weight_buffers(cls, dtype):
    from fastscipy import _csgraph

    g = cls(graph(330, n=37, density=0.12).astype(dtype))
    g.indices, g.indptr = g.indices.astype(np.int32), g.indptr.astype(np.int32)
    g.data[::7] = 0  # Stored zeros still represent edges in unweighted mode.
    g.data.flags.writeable = g.indices.flags.writeable = g.indptr.flags.writeable = False
    before = tuple(a.tobytes() for a in (g.data, g.indices, g.indptr))
    assert _csgraph._unit_csr(g)
    for directed in (True, False):
        for min_only in (True, False):
            for limit in (0, 1, np.inf):
                check(
                    lambda: cg.dijkstra(
                        g,
                        directed=directed,
                        indices=[0, 3],
                        unweighted=True,
                        min_only=min_only,
                        limit=limit,
                        return_predecessors=True,
                    )
                )
        check(lambda: cg.johnson(g, directed=directed, indices=0, unweighted=True, return_predecessors=True))
        check(lambda: cg.shortest_path(g, directed=directed, indices=0, unweighted=True, return_predecessors=True))
    assert tuple(a.tobytes() for a in (g.data, g.indices, g.indptr)) == before


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_integer_unit_graph_canonical_validation():
    from fastscipy import _csgraph

    # Deliberately stale flags: astype(float64) reorders/merges this input.
    for columns in ([2, 1], [1, 1]):
        g = sp.csr_array(
            (np.array([3, 4], np.int16), np.array(columns, np.int32), np.array([0, 2, 2, 2], np.int32)), shape=(3, 3)
        )
        g.has_sorted_indices = g.has_canonical_format = True
        assert not _csgraph._unit_csr(g)
        for directed in (True, False):
            check(lambda: cg.dijkstra(g, directed=directed, unweighted=True, indices=0, return_predecessors=True))
    for pointer, index in (([0, 2, 1, 2], [0, 1]), ([0, 1, 1, 1], [3]), ([0, 1, 1, 1], [-1])):
        g = sp.csr_array((3, 3), dtype=np.int16)
        g.indptr, g.indices, g.data = (
            np.array(pointer, np.int32),
            np.array(index, np.int32),
            np.ones(len(index), np.int16),
        )
        assert not _csgraph._unit_csr(g)  # Decline before calling the C++ canonical checker.
    assert not _csgraph._unit_csr(sp.csr_array((2, 3), dtype=np.int16))
    g = sp.csr_array(np.eye(3, dtype=np.int16))
    g.data = g.data.reshape(-1, 1)
    assert not _csgraph._unit_csr(g)
    g.data = g.data.reshape(-1).view(type("ArraySubclass", (np.ndarray,), {}))
    assert not _csgraph._unit_csr(g)


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_integer_unit_graph_negative_warnings():
    import warnings

    g = sp.csr_array((np.array([-3, 1], np.int16), ([0, 1], [1, 2])), shape=(3, 3))
    outputs, warnings_seen = [], []
    for enabled in (False, True):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            if enabled:
                out = cg.dijkstra(g, unweighted=True, indices=0)
            else:
                with fastscipy.disabled():
                    out = cg.dijkstra(g, unweighted=True, indices=0)
        outputs.append(out)
        warnings_seen.append([(type(w.message), str(w.message)) for w in seen])
    assert_identical(*outputs)
    assert warnings_seen[0] == warnings_seen[1]


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_integer_unit_graph_passes_empty_native_weight_buffers(monkeypatch):
    from fastscipy import _csgraph

    original = _csgraph._k.dijkstra
    calls = []

    def capture(*args):
        assert args[3].dtype == args[6].dtype == np.float64
        assert args[3].size == args[6].size == 0
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(_csgraph._k, "dijkstra", capture)
    g = sp.csr_matrix((np.array([0, 2, 3], np.uint8), ([0, 1, 2], [1, 2, 3])), shape=(4, 4))
    for directed in (False, True):
        check(lambda: cg.dijkstra(g, directed=directed, unweighted=True, indices=0))
    assert len(calls) == 2
