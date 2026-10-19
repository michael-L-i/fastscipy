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
