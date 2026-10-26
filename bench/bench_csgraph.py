import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg


def grid(n):
    idx = np.arange(n * n).reshape(n, n)
    r = np.concatenate([idx[:, :-1].ravel(), idx[:-1].ravel()])
    c = np.concatenate([idx[:, 1:].ravel(), idx[1:].ravel()])
    rng = np.random.default_rng(0)
    return sp.csr_array((rng.integers(1, 10, len(r)).astype(float), (r, c)), (n * n, n * n))


def random_graph(n, m, seed=0, weights=True):
    rng = np.random.default_rng(seed)
    w = rng.random(m) + 0.1 if weights else np.ones(m)
    return sp.csr_array((w, (rng.integers(0, n, m), rng.integers(0, n, m))), (n, n))


def cases():
    # Permuted wide levels make ordering work visible; distance-only BFS does
    # not need the predecessor tie order. Reverse chains stress explicit BF.
    rng = np.random.default_rng(392)
    width = 50000
    columns = rng.permutation(np.arange(1, width + 1))
    wide = sp.csr_array(
        (np.ones(2 * width), (np.r_[np.zeros(width, dtype=int), columns], np.r_[columns, width + columns])),
        shape=(2 * width + 1, 2 * width + 1),
    )
    wide.has_sorted_indices = False
    wide.indices[:width] = columns
    for predecessors in (False, True):
        yield (
            f"dijkstra unweighted wide frontier, 100k nodes, predecessors={predecessors}",
            lambda p=predecessors: cg.dijkstra(wide, indices=0, unweighted=True, return_predecessors=p),
        )
    chain = sp.csr_array((np.ones(4999), (np.arange(1, 5000), np.arange(4999))), shape=(5000, 5000))
    yield "dijkstra unweighted reverse chain, 5k nodes", lambda: cg.dijkstra(chain, indices=4999, unweighted=True)
    yield (
        "bellman_ford unweighted reverse chain, 5k nodes",
        lambda: cg.bellman_ford(chain, indices=4999, unweighted=True),
    )
    unit_rng = np.random.default_rng(332)
    unit_rows, unit_cols = unit_rng.integers(0, 10000, (2, 100000), dtype=np.int32)
    for dtype in (np.uint8, np.int32):
        integer_unit = sp.csr_matrix((np.ones(100000, dtype=dtype), (unit_rows, unit_cols)), shape=(10000, 10000))
        for directed in (True, False):
            yield (
                f"dijkstra unweighted {np.dtype(dtype).name} CSR, 10k nodes, radius 1, directed={directed}",
                lambda g=integer_unit, d=directed: cg.dijkstra(g, directed=d, unweighted=True, indices=0, limit=1),
            )
    scalar_graph = random_graph(200, 400)
    scalar_graph.indices = scalar_graph.indices.astype(np.int32)
    scalar_graph.indptr = scalar_graph.indptr.astype(np.int32)
    yield (
        "dijkstra int32 CSR, scalar source, 200 nodes x100 calls",
        lambda: [cg.dijkstra(scalar_graph, indices=0) for _ in range(100)],
    )
    yield (
        "connected_components int32 CSR, 200 nodes x100 calls",
        lambda: [cg.connected_components(scalar_graph) for _ in range(100)],
    )
    unit_graph = grid(100)
    yield (
        "dijkstra unweighted undirected, grid 100^2, scalar source",
        lambda: cg.dijkstra(unit_graph, directed=False, unweighted=True, indices=0),
    )
    g = grid(600)
    yield "shortest_path unweighted BFS, grid 600^2, 1 source", lambda: cg.shortest_path(g, unweighted=True, indices=0)
    yield (
        "dijkstra unweighted + predecessors, grid 600^2",
        lambda: cg.dijkstra(g, unweighted=True, indices=0, return_predecessors=True),
    )
    g2 = grid(300)
    yield "dijkstra weighted, 18 sources, grid 300^2", lambda: cg.dijkstra(g2, indices=np.arange(18) * 1000)
    yield (
        "dijkstra weighted + predecessors, 12 sources, undirected",
        lambda: cg.dijkstra(g2, False, np.arange(12) * 500, True),
    )
    r = random_graph(2000, 40000)
    yield "bellman_ford 2000 nodes, 40k edges, 20 sources", lambda: cg.bellman_ford(r, indices=np.arange(20))
    yield "johnson 2000 nodes, 40k edges, 100 sources", lambda: cg.johnson(r, indices=np.arange(100))
    big = random_graph(1_000_000, 1_500_000, weights=False)
    yield "connected_components weak, 1M nodes", lambda: cg.connected_components(big)
    yield "connected_components strong, 1M nodes", lambda: cg.connected_components(big, connection="strong")
    small = random_graph(200, 400, 1)
    yield (
        "connected_components, 200 nodes x100 calls",
        lambda: [cg.connected_components(small, directed=False) for _ in range(100)],
    )
