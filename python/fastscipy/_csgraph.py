"""scipy.sparse.csgraph: Rust ports of the shortest-path searches (sources run in
parallel, Bellman-Ford and Johnson stop once a pass changes nothing) and of
``connected_components``. The wrappers repeat SciPy's argument handling, then
call the kernels instead of Cython; anything unusual (non-CSR input that fails
validation, negative weights in Dijkstra, int64 indices, a negative cycle) goes to SciPy."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse._sputils import safely_cast_index_arrays
from scipy.sparse.csgraph import _shortest_path as _paths, _traversal
from scipy.sparse.csgraph._validation import validate_graph

from . import _native, _patch

_k = _native.csgraph
I32, NULL_IDX = np.int32, -9999
_EMPTY_F, _EMPTY_I = np.empty(0), np.empty(0, I32)


def _csr(g, directed, **kw):
    """The validated CSR graph (what SciPy's ``validate_graph`` returns); None to decline."""
    if sp.issparse(g) and g.format == "csr" and g.dtype == kw.get("dtype", np.float64):
        out = g
    else:
        out = validate_graph(g, directed, dense_output=False, **kw)
    return out if out.shape[0] > 0 else None


def _arrays(g):
    ix, ip = safely_cast_index_arrays(g, I32, msg="csgraph")
    return (ix, ip) if ix.dtype == I32 and ip.dtype == I32 else (None, None)


def _sources(indices, n, wrap_negative=True):
    """SciPy's ``indices`` handling: (flat int32 sources, output shape), or None to decline."""
    if indices is None:
        return np.arange(n, dtype=I32), (n,)
    indices = np.array(indices, order="C", dtype=I32)
    shape = indices.shape
    indices = np.atleast_1d(indices).reshape(-1)
    indices[indices < 0] += n
    if (indices < 0).any() or (indices >= n).any():
        return None
    return indices, shape


def _transposed(g, unweighted):
    t = g.T.tocsr()
    ix, ip = _arrays(t)
    return (ip, ix, np.ones(t.data.shape) if unweighted else t.data) if ix is not None else None


def _dijkstra(
    csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False, limit=np.inf, min_only=False
):
    g = _csr(csgraph, directed)
    min_only, limit = bool(min_only), float(limit)
    if g is None or (g.data < 0).any() or not limit >= 0:
        return NotImplemented
    n = g.shape[0]
    ix, ip = _arrays(g)
    src = _sources(indices, n)
    t = _transposed(g, unweighted) if not directed and ix is not None else (_EMPTY_I, _EMPTY_I, _EMPTY_F)
    if ix is None or src is None or t is None:
        return NotImplemented
    srcs, shape = src
    shape = (n,) if min_only else shape + (n,)
    rows = (n,) if min_only else (len(srcs), n)
    dist = np.full(rows, np.inf)
    pred = np.full(rows, NULL_IDX, I32) if return_predecessors else np.empty(0, I32)
    source = np.full(n, NULL_IDX, I32) if return_predecessors and min_only else np.empty(0, I32)
    _k.dijkstra(srcs, ip, ix, g.data, *t, dist.reshape(-1), pred.reshape(-1), source, limit, min_only, bool(unweighted))
    if not return_predecessors:
        return dist.reshape(shape)
    out = dist.reshape(shape), pred.reshape(shape)
    return out + (source.reshape(shape),) if min_only else out


def _bellman_ford(csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False):
    g = _csr(csgraph, directed)
    if g is None:
        return NotImplemented
    n = g.shape[0]
    ix, ip = _arrays(g)
    src = _sources(indices, n)
    if ix is None or src is None:
        return NotImplemented
    srcs, shape = src
    dist = np.full((len(srcs), n), np.inf)
    dist[np.arange(len(srcs)), srcs] = 0
    pred = np.full((len(srcs), n), NULL_IDX, I32) if return_predecessors else np.empty(0, I32)
    data = np.ones(g.data.shape) if unweighted else g.data
    if _k.bellman_ford(srcs, ip, ix, data, not directed, dist.reshape(-1), pred.reshape(-1)) >= 0:
        return NotImplemented  # SciPy raises NegativeCycleError
    out = dist.reshape(shape + (n,))
    return (out, pred.reshape(shape + (n,))) if return_predecessors else out


def _johnson(csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False):
    if unweighted:
        return _dijkstra(csgraph, directed, indices, return_predecessors, unweighted)
    g = _csr(csgraph, directed)
    if g is None:
        return NotImplemented
    n = g.shape[0]
    ix, ip = _arrays(g)
    src = _sources(indices, n)
    if ix is None or src is None:
        return NotImplemented
    srcs, shape = src
    data, h = g.data.copy(), np.zeros(n)
    if _k.johnson_potentials(ip, ix, data, not directed, h) >= 0:
        return NotImplemented  # SciPy raises NegativeCycleError
    _k.add_weights(ip, ix, data, h)
    t = (_EMPTY_I, _EMPTY_I, _EMPTY_F)
    if not directed:
        gt = sp.csr_array((data, ix, ip), g.shape).T.tocsr()
        tix, tip = _arrays(gt)
        if tix is None or gt.data.dtype != np.float64 or not gt.data.flags.writeable:
            return NotImplemented
        _k.add_weights(tip, tix, gt.data, h)
        t = (tip, tix, gt.data)
    dist = np.full((len(srcs), n), np.inf)
    pred = np.full((len(srcs), n), NULL_IDX, I32) if return_predecessors else np.empty(0, I32)
    _k.dijkstra(srcs, ip, ix, data, *t, dist.reshape(-1), pred.reshape(-1), np.empty(0, I32), np.inf, False, False)
    dist += h
    dist -= h[:, None][srcs]
    out = dist.reshape(shape + (n,))
    return (out, pred.reshape(shape + (n,))) if return_predecessors else out


def _shortest_path(
    csgraph, method="auto", directed=True, return_predecessors=False, unweighted=False, overwrite=False, indices=None
):
    if not (
        sp.issparse(csgraph)
        and csgraph.format == "csr"
        and csgraph.shape[0] == csgraph.shape[1]
        and csgraph.dtype.kind in "biuf"
    ):
        return NotImplemented
    if method == "auto":
        if indices is None and csgraph.nnz >= csgraph.shape[0] ** 2 / 4:
            return NotImplemented  # Floyd-Warshall
        method = "J" if (csgraph.data < 0).any() else "D"
    fast = {"D": _dijkstra, "BF": _bellman_ford, "J": _johnson}.get(method)
    if fast is None:
        return NotImplemented
    return fast(csgraph, directed, indices=indices, return_predecessors=return_predecessors, unweighted=unweighted)


def _connected_components(csgraph, directed=True, connection="weak", return_labels=True):
    if not isinstance(connection, str) or connection.lower() not in ("weak", "strong"):
        return NotImplemented
    weak = connection.lower() == "weak"
    g = _csr(csgraph, directed and not weak, dtype=csgraph.dtype)
    ix, ip = _arrays(g) if g is not None else (None, None)
    if ix is None:
        return NotImplemented
    labels = np.empty(g.shape[0], I32)
    count = (_k.weak if weak or not directed else _k.strong)(ip, ix, labels)
    return (count, labels) if return_labels else count


def patch():
    _patch.require_validated("1.18.")
    for mod, name, fast in (
        (_paths, "dijkstra", _dijkstra),
        (_paths, "bellman_ford", _bellman_ford),
        (_paths, "johnson", _johnson),
        (_paths, "shortest_path", _shortest_path),
        (_traversal, "connected_components", _connected_components),
    ):
        old = getattr(mod, name)
        _patch.rebind(old, _patch.wrap(old, fast))


def smoke():
    import scipy.sparse.csgraph as cg

    g = sp.random_array((30, 30), density=0.2, format="csr", rng=0)
    for directed in (True, False):
        cg.dijkstra(g, directed, [0, 1], return_predecessors=True)
        cg.dijkstra(g, directed, [0, 1], unweighted=True, min_only=True)
        cg.bellman_ford(g, directed, [0, 1])
        cg.johnson(g, directed, [0, 1])
        cg.connected_components(g, directed, "strong")
    cg.shortest_path(g, unweighted=True, indices=0)
