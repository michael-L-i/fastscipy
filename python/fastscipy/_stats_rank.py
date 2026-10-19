"""Rust radix `_rankdata` for float64 and integer input (used by rankdata, spearmanr,
mannwhitneyu, wilcoxon, kruskal, ...)."""

import numpy as np
from scipy._lib._array_api import array_namespace

from . import _native, _patch

_METHODS = {"average": 0, "min": 1, "max": 2, "dense": 3, "ordinal": 4}
_KIND = {np.dtype("float64"): 0, np.dtype("int64"): 1, np.dtype("uint64"): 2}
_NP = array_namespace(np.zeros(1))


def _rankdata(x, method, return_sorted=False, return_ties=False, xp=None):
    if (
        type(x) is not np.ndarray
        or x.ndim < 1
        or not x.size
        or method not in _METHODS
        or not (xp is None or xp is _NP)
        or x.shape[-1] >= 2**32
    ):
        return NotImplemented
    dt = x.dtype
    if dt in _KIND:
        v = x
    elif dt.kind in "iu" and dt.itemsize < 8:  # SciPy ranks these as float64 too
        v = x.astype(np.int64)
    else:
        return NotImplemented
    n = x.shape[-1]
    bits = np.ascontiguousarray(v).reshape(-1, n).view(np.uint64)
    ordinal = method == "ordinal"
    ranks, srt, ties = _native.stats.rankdata(
        bits, _KIND[v.dtype], _METHODS[method], bool(return_sorted) and not ordinal, bool(return_ties) and not ordinal
    )
    ranks = ranks.reshape(x.shape)
    if ordinal:
        return ranks, None, None
    # SciPy's second output is the sorted array even when not asked for; no caller reads it then.
    return (
        ranks,
        None if srt is None else srt.view(v.dtype).astype(dt, copy=False).reshape(x.shape),
        None if ties is None else ties.reshape(x.shape),
    )


def patch():
    import scipy.stats._stats_py as sp

    old = sp._rankdata
    _patch.rebind(old, _patch.wrap(old, _rankdata, "_rankdata"))
