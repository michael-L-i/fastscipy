"""Threaded scipy.spatial: cdist/pdist by row blocks, and KDTree queries.

``cdist`` and ``pdist`` run SciPy's own GIL-releasing kernel on blocks of rows;
every output element comes from the same per-pair loop, so bytes are identical
(checked per metric; mahalanobis and seuclidean estimate statistics once from
the whole input, and pdist/dice differs, so it is left to SciPy). KDTree
queries pass ``workers`` when the caller left the default.
"""

import numpy as np
import scipy.spatial as sp
import scipy.spatial._ckdtree as _ck
import scipy.spatial.distance as dist

from . import _patch
from ._threads import chunks, get_num_threads, parallel_map

_CDIST = {
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "cosine",
    "correlation",
    "minkowski",
    "canberra",
    "braycurtis",
    "hamming",
    "jaccard",
    "jensenshannon",
    "dice",
    "yule",
    "russellrao",
    "sokalsneath",
    "rogerstanimoto",
    "seuclidean",
    "mahalanobis",
}
_PDIST = _CDIST - {"dice"}
_STATISTICS = {"seuclidean": "V", "mahalanobis": "VI"}
_WEIGHTED = _CDIST - {"cosine", "correlation", "jensenshannon", *_STATISTICS}
_OPTIONS = {
    m: {"metric"}
    | ({"w"} if m in _WEIGHTED else set())
    | ({"p"} if m == "minkowski" else set())
    | ({_STATISTICS[m]} if m in _STATISTICS else set())
    for m in _CDIST
}
_WORK = 1_000_000  # pair-coordinates below which a block split loses to hand-off cost
_BLOCK = 200_000  # about 100 us per block
_QUERIES = 2000  # KDTree queries below which threads cost more than they save


def _metric(args, kw, names, *arrays):
    """The metric name when the call has supported options and NumPy inputs."""
    if len(args) > 1 or (args and "metric" in kw) or get_num_threads() <= 1:
        return None
    metric = args[0] if args else kw.get("metric", "euclidean")
    if (
        isinstance(metric, str)
        and metric in names
        and kw.keys() <= _OPTIONS[metric]
        and all(type(a) is np.ndarray and a.dtype in (np.float64, np.float32) and a.ndim == 2 for a in arrays)
    ):
        return metric


def _parameters(metric, kw, *arrays):
    """Use SciPy's full-input statistics before splitting either distance kernel."""
    key = _STATISTICS.get(metric)
    if key is None:
        params = {k: v for k, v in kw.items() if k != "metric"}
        w = params.get("w")
        if w is not None and not (
            type(w) is np.ndarray
            and w.dtype == np.float64
            and w.shape == (arrays[0].shape[1],)
            and np.isfinite(w).all()
            and (w >= 0).all()
        ):
            return None
        p = params.get("p", 2.0)
        if type(p) not in (int, float) or not 0 < p <= np.inf:
            return None
        return params
    value = kw.get(key)
    d = arrays[0].shape[1]
    if value is not None:
        shape = (d,) if key == "V" else (d, d)
        if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape:
            return None
        return {key: value}
    if sum(a.shape[0] for a in arrays) <= max(d, 1) or not all(np.isfinite(a).all() for a in arrays):
        return None
    validator = dist._validate_seuclidean_kwargs if key == "V" else dist._validate_mahalanobis_kwargs
    try:
        with np.errstate(all="raise"):
            return validator(arrays if len(arrays) > 1 else arrays[0], sum(a.shape[0] for a in arrays), d)
    except (FloatingPointError, np.linalg.LinAlgError):
        return None


def _cdist_fast(original):
    def fast(XA, XB, *args, **kw):
        metric = _metric(args, kw, _CDIST, XA, XB)
        if not metric or XA.shape[1] != XB.shape[1]:
            return NotImplemented
        (ma, d), mb = XA.shape, XB.shape[0]
        if ma * mb * d < _WORK:
            return NotImplemented
        blocks = chunks(ma, -(-_BLOCK // max(1, mb * d)))
        if len(blocks) < 2:
            return NotImplemented
        try:
            with np.errstate(all="raise"):
                XA, XB = np.ascontiguousarray(XA, dtype=np.float64), np.ascontiguousarray(XB, dtype=np.float64)
        except FloatingPointError:
            return NotImplemented  # SciPy owns signaling-NaN cast warnings and callbacks.
        params = _parameters(metric, kw, XA, XB)
        if params is None:
            return NotImplemented
        res = np.empty((ma, mb))
        parallel_map(lambda s: original(XA[s], XB, metric, out=res[s], **params), blocks)
        return res

    return fast


def _pdist_fast(cdist):
    def fast(X, *args, **kw):
        metric = _metric(args, kw, _PDIST, X)
        if not metric:
            return NotImplemented
        m, d = X.shape
        if m * m * d < 2 * _WORK:
            return NotImplemented
        try:
            with np.errstate(all="raise"):
                X = np.ascontiguousarray(X, dtype=np.float64)
        except FloatingPointError:
            return NotImplemented
        params = _parameters(metric, kw, X)
        if params is None:
            return NotImplemented
        # Block [a, b) owns condensed entries of rows a..b-1; weight rows by their pair count.
        rows = np.cumsum(m - 1 - np.arange(m))
        res = np.empty(m * (m - 1) // 2)
        starts = np.concatenate(([0], rows))
        k = max(2, min(8 * get_num_threads(), m * m * d // _BLOCK))
        bounds = sorted(set(np.searchsorted(rows, np.linspace(0, rows[-1], k + 1)).tolist()) | {0, m - 1})

        def block(ab):
            a, b = ab
            D = cdist(X[a:b], X[a:], metric, **params)
            # One masked copy, not a Python loop over rows: a loop holds the GIL and serialises the blocks.
            res[starts[a] : starts[b]] = D[np.arange(D.shape[1]) > np.arange(b - a)[:, None]]

        parallel_map(block, list(zip(bounds, bounds[1:])))
        return res

    return fast


def _tree_fast(original, workers_pos, rows):
    """Add ``workers`` to a default-workers query; ``rows(tree, x)`` counts the queries."""

    def fast(self, x, *args, **kw):
        if "workers" in kw or len(args) > workers_pos or get_num_threads() <= 1:
            return NotImplemented
        n = rows(self, x)
        if n < _QUERIES:
            return NotImplemented
        return original(self, x, *args, workers=min(get_num_threads(), n // 500), **kw)

    return fast


def _nq(tree, x):
    x = np.asarray(x)
    return x.size // tree.m if x.ndim > 1 else 0


class _Meta(type):
    """Keep ``isinstance(tree, cKDTree)`` true for trees built from the original class (KDTree)."""

    def __instancecheck__(cls, obj):
        return isinstance(obj, cls.__mro__[1])

    def __subclasscheck__(cls, sub):
        return issubclass(sub, cls.__mro__[1])


def patch():
    _patch.require_validated("1.18.")
    cdist = dist.cdist
    new_cdist = _patch.wrap(cdist, _cdist_fast(cdist), "cdist")
    _patch.rebind(cdist, new_cdist)
    pdist = dist.pdist
    new_pdist = _patch.wrap(pdist, _pdist_fast(cdist), "pdist")
    _patch.rebind(pdist, new_pdist)

    # cKDTree is an immutable extension type: swap in a subclass; KDTree is a Python class.
    base = _ck.cKDTree
    ns = {"__module__": base.__module__, "__qualname__": "cKDTree", "__slots__": ()}
    for name, pos in (("query", 4), ("query_ball_point", 3)):
        ns[name] = _method(base, name, pos, "cKDTree")
    sub = _Meta("cKDTree", (base,), ns)
    _patch.rebind(base, sub)
    for name, pos in (("query", 4), ("query_ball_point", 3)):
        _patch.replace(sp.KDTree, name, _method(sp.KDTree, name, pos, "KDTree"))


def _method(cls, name, pos, label):
    original = getattr(cls, name)
    w = _patch.wrap(original, _tree_fast(original, pos, _nq), f"{label}.{name}")
    return w


def smoke():
    rng = np.random.default_rng(0)
    A = rng.random((300, 16))
    dist.cdist(A, A), dist.pdist(A)
    t = sp.cKDTree(A[:, :3])
    t.query(A[:, :3]), t.query_ball_point(A[:, :3], 0.1)
