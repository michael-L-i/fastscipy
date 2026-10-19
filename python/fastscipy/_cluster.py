"""NumPy clustering paths using SciPy's vector-quantization kernel and compact
linkage validation. Other backends and diagnostic cases remain with SciPy.
"""

import numpy as np
from scipy.cluster.hierarchy import _hierarchy_impl as _hierarchy
from scipy.cluster.vq import _vq_impl
from scipy._external.array_api_compat import numpy as _numpy

from . import _patch


def _vq(obs, code_book, check_finite=True):
    if (
        type(obs) is not np.ndarray
        or type(code_book) is not np.ndarray
        or type(check_finite) is not bool
        or obs.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or obs.dtype != code_book.dtype
        or obs.ndim not in (1, 2)
        or obs.ndim != code_book.ndim
        or obs.size == 0
        or code_book.size == 0
        or obs.ndim == 2
        and obs.shape[1] != code_book.shape[1]
    ):
        return NotImplemented
    if check_finite and (not np.isfinite(obs).all() or not np.isfinite(code_book).all()):
        return NotImplemented
    return _vq_impl._vq.vq(obs, code_book)


def _whiten(obs, check_finite=None):
    if (
        type(obs) is not np.ndarray
        or obs.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or obs.ndim != 2
        or obs.shape[0] < 2
        or obs.shape[1] == 0
        or check_finite is not None
        and type(check_finite) is not bool
    ):
        return NotImplemented
    check_finite = check_finite is not False
    if check_finite and not np.isfinite(obs).all():
        return NotImplemented
    # A failed speculative computation must leave warnings and floating-point
    # exceptions to the original function, which runs exactly once afterward.
    try:
        with np.errstate(all="raise"):
            std = np.std(obs, axis=0, ddof=0.0)
            zero = std == 0
            if check_finite and zero.any():
                return NotImplemented
            std[zero] = 1.0
            return obs / std
    except FloatingPointError:
        return NotImplemented


def _is_valid_linkage(Z, warning=False, throw=False, name=None, materialize=False, *, xp):
    if (
        type(Z) is not np.ndarray
        or xp is not np
        and xp is not _numpy
        or Z.dtype != np.float64
        or Z.ndim != 2
        or Z.shape[1] != 4
        or Z.shape[0] == 0
        or type(warning) is not bool
        or type(throw) is not bool
        or type(materialize) is not bool
        or name is not None
        and type(name) is not str
    ):
        return NotImplemented
    n = Z.shape[0]
    if n == 1:
        return True  # SciPy does not inspect the contents of a single merge.
    if not np.isfinite(Z).all() or (Z < 0).any() or (Z[:, 3] > n + 1).any():
        return NotImplemented
    children = Z[:, :2]
    if (children.max(axis=1) >= np.arange(n + 1, 2 * n + 1, dtype=np.float64)).any():
        return NotImplemented
    indices = children.astype(np.intp)
    if not np.array_equal(indices, children):
        return NotImplemented
    # Integral cluster IDs are bounded above by 2*n. Counting avoids sorting
    # all IDs, and accepts only the same no-repeated-clusters case as SciPy.
    if np.bincount(indices.ravel(), minlength=2 * n).max() != 1:
        return NotImplemented
    return True


def patch():
    _patch.require_validated("1.18.")
    for owner, name, fast in (
        (_vq_impl, "vq", _vq),
        (_vq_impl, "whiten", _whiten),
        (_hierarchy, "_is_valid_linkage", _is_valid_linkage),
    ):
        old = getattr(owner, name)
        _patch.rebind(old, _patch.wrap(old, fast, name))


def smoke():
    from scipy.cluster import hierarchy, vq

    x = np.arange(24, dtype=np.float64).reshape(8, 3)
    vq.whiten(x)
    vq.vq(x, x[::2])
    hierarchy.leaves_list(np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 3.0]]))
