"""scipy.sparse fast paths. Tier 1 (faithful ports and algorithmic fixes), no change in results.

- constructor helpers (``get_index_dtype``, ``check_shape``, ``prune``, ``check_format``):
  pure Python, about 1.5x on small constructions
- ``argmax``/``argmin`` along an axis, COO ``sum_duplicates`` and dense -> CSR/CSC/COO: Rust
- ``find``: canonicalise in CSR (C++) instead of ``coo.sum_duplicates``

The memory-bound ``csr_*`` kernels are not taken over: on one thread SciPy's C++ is faster.
"""
import numpy as np

from . import _native, _patch
from ._threads import get_num_threads

_I32MAX = 2**31 - 1
_IDX, _VALS = "il", "fdil"   # dtype.char: int32/int64 and float32/float64/int32/int64 (not np.dtype ==, which
                              # also equates longdouble with float64 here)
_PAR = {"argminmax": 1_000_000, "sumdup": 100_000, "dense": 1 << 17}   # entries before threads pay off
_DENSE_MIN = 4096                                                         # smaller dense inputs: SciPy


def _threads(work, key):
    return get_num_threads() if work >= _PAR[key] else 1


def _contig(*arrs):
    return all(a.__class__ is np.ndarray and a.ndim == 1 and a.flags.c_contiguous for a in arrs)


# ---- small calls --------------------------------------------------------------------------------

def _small_calls():
    from scipy.sparse import _base, _compressed, _sputils

    orig_gid, orig_cs = _sputils.get_index_dtype, _sputils.check_shape
    orig_prune, orig_check = _compressed._cs_matrix.prune, _compressed._cs_matrix.check_format
    orig_sgid = _base._spbase._get_index_dtype
    can_cast = {}

    def get_index_dtype(arrays=(), maxval=None, check_contents=False):
        if not _patch.enabled or maxval is not None and (maxval.__class__ is not int or maxval > _I32MAX):
            return orig_gid(arrays, maxval, check_contents)
        for arr in (arrays,) if arrays.__class__ is np.ndarray else arrays:
            dt = (arr if arr.__class__ is np.ndarray else np.asarray(arr)).dtype
            if (ok := can_cast.get(dt)) is None:
                ok = can_cast[dt] = np.can_cast(dt, np.int32)
            if not ok:
                return orig_gid(arrays, maxval, check_contents)
        return np.int32

    def spbase_get_index_dtype(self, arrays=(), maxval=None, check_contents=False):
        if not _patch.enabled:
            return orig_sgid(self, arrays, maxval, check_contents)
        # looked up at call time: SciPy's tests swap it out in _sputils
        return _sputils.get_index_dtype(arrays, maxval, check_contents and not isinstance(self, _base.sparray))

    def check_shape(args, current_shape=None, *, allow_nd=(2,)):
        if _patch.enabled and current_shape is None and args.__class__ is tuple and len(args) == 2:
            a, b = args
            if a.__class__ is int and b.__class__ is int and a >= 0 and b >= 0 and 2 in allow_nd:
                return args
        return orig_cs(args, current_shape, allow_nd=allow_nd)

    def prune(self):
        ip, ind, dat = self.indptr, self.indices, self.data
        if _patch.enabled and ip.__class__ is np.ndarray and ind.__class__ is np.ndarray and dat.__class__ is np.ndarray:
            nnz = int(ip[-1])
            if len(ip) == self._swap(self._shape_as_2d)[0] + 1 and len(ind) >= nnz and len(dat) >= nnz:
                ind, dat = ind[:nnz], dat[:nnz]
                self.indices = ind.copy() if ind.base is not None and ind.size < ind.base.size // 2 else ind
                self.data = dat.copy() if dat.base is not None and dat.size < dat.base.size // 2 else dat
                return
        orig_prune(self)

    def check_format(self, full_check=True):
        if not full_check:
            ip, ind, dat = self.indptr, self.indices, self.data
            if (ip.__class__ is np.ndarray and ind.__class__ is np.ndarray and dat.__class__ is np.ndarray
                    and ip.dtype.kind == "i" and ind.dtype.kind == "i" and ip.ndim == 1 == ind.ndim == dat.ndim
                    and len(ip) == self._swap(self._shape_as_2d)[0] + 1 and ip[0] == 0
                    and len(ind) == len(dat) and ip[-1] <= len(ind)):
                self.prune()
                return
        return NotImplemented

    _patch.rebind(orig_gid, get_index_dtype)
    _patch.rebind(orig_cs, check_shape)
    _patch.replace(_base._spbase, "_get_index_dtype", spbase_get_index_dtype)
    _patch.replace(_compressed._cs_matrix, "prune", prune)
    _patch.replace(_compressed._cs_matrix, "check_format", _patch.wrap(orig_check, check_format, "check_format"))


# ---- argmax / argmin ----------------------------------------------------------------------------

def _argminmax_axis(self, axis, argminmax, compare, explicit):
    if (self.ndim != 2 or self.dtype.char not in _VALS or axis not in (0, 1) or explicit.__class__ is not bool
            or (argminmax, compare) not in ((np.argmax, np.greater), (np.argmin, np.less))):
        return NotImplemented
    mat = self.tocsc() if axis == 0 else self.tocsr()
    mat.sum_duplicates()
    ret_size, line_size = mat._swap(mat.shape)
    ip, ij, data = mat.indptr, mat.indices, mat.data
    if ip.dtype != ij.dtype or ip.dtype.char not in _IDX or not _contig(ip, ij, data):
        return NotImplemented
    ret = _native.sparse.argminmax(ip, ij, data, line_size, argminmax is np.argmax, explicit,
                                   _threads(len(data), "argminmax"))
    if ret is None:
        return NotImplemented
    from scipy.sparse._base import sparray
    if isinstance(self, sparray):
        return ret
    return self._ascontainer(ret.reshape(-1, 1) if axis == 1 else ret)


# ---- COO sum_duplicates -------------------------------------------------------------------------

def _sum_duplicates(self, coords, data):
    if len(coords) != 2 or not _contig(data, *coords) or data.dtype.char not in _VALS or data.dtype != self.dtype:
        return NotImplemented
    row, col = coords
    n, m = len(data), self.shape[0]
    if n == 0 or row.dtype.char not in _IDX or row.dtype != col.dtype or len(row) != n or len(col) != n \
            or m > 8 * n + 1_000_000:
        return NotImplemented
    r = _native.sparse.sum_duplicates(row, col, data, m, _threads(n, "sumdup"))
    return NotImplemented if r is None else ((r[0], r[1]), r[2])


# ---- dense -> sparse ----------------------------------------------------------------------------

def _dense(arg1, shape, dtype, csc):
    """(indptr, indices, data) of a dense 2-D array when it is a plain case, else None."""
    if (arg1.__class__ is not np.ndarray or arg1.ndim != 2 or shape is not None or dtype is not None
            or arg1.size < _DENSE_MIN or arg1.dtype.char not in _VALS or max(arg1.shape) > _I32MAX):
        return None
    return _native.sparse.dense_to_compressed(arg1, csc, _threads(arg1.size, "dense"))


def _dense_inits(cs_cls, coo_cls):
    orig_cs, orig_coo = cs_cls.__init__, coo_cls.__init__

    def cs_init(self, arg1, shape=None, dtype=None, copy=False, *extra, maxprint=None):
        if extra or self.format not in ("csr", "csc") or (r := _dense(arg1, shape, dtype, self.format == "csc")) is None:
            return NotImplemented
        indptr, indices, data = r
        orig_cs(self, (data, indices, indptr), shape=arg1.shape, copy=False, maxprint=maxprint)

    def coo_init(self, arg1, shape=None, dtype=None, copy=False, *extra, maxprint=None):
        if extra or (r := _dense(arg1, shape, dtype, False)) is None:
            return NotImplemented
        indptr, indices, data = r
        row = np.repeat(np.arange(arg1.shape[0], dtype=np.int32), np.diff(indptr))
        orig_coo(self, (data, (row, indices)), shape=arg1.shape, copy=False, maxprint=maxprint)
        self.has_canonical_format = True

    for cls, fast, name in ((cs_cls, cs_init, "cs.__init__"), (coo_cls, coo_init, "coo.__init__")):
        _patch.replace(cls, "__init__", _patch.wrap(cls.__init__, fast, name))   # fast returns None when it built the object


# ---- find ---------------------------------------------------------------------------------------

def _find(A):
    from scipy.sparse import csr_array
    from scipy.sparse._base import sparray
    if not (isinstance(A, sparray) and A.ndim == 2 and A.format in ("csr", "csc", "coo")):
        return NotImplemented
    if A.format == "coo" and A.has_canonical_format:   # SciPy's own path has nothing to sort
        return NotImplemented
    dtypes = (A.row.dtype, A.col.dtype) if A.format == "coo" else (A.indices.dtype,) * 2
    A = csr_array(A, copy=True)
    A.sum_duplicates()
    A = A.tocoo()
    row, col = A.row.astype(dtypes[0], copy=False), A.col.astype(dtypes[1], copy=False)
    keep = A.data != 0
    return row[keep], col[keep], A.data[keep]


def patch():
    _patch.require_scipy("1.18.")
    from scipy.sparse import _coo, _compressed, _data, _extract
    _native.sparse  # noqa: B018  (fail early if the extension lacks the module)
    _small_calls()
    for owner, name, fast in ((_data._minmax_mixin, "_argminmax_axis", _argminmax_axis),
                              (_coo._coo_base, "_sum_duplicates", _sum_duplicates)):
        _patch.replace(owner, name, _patch.wrap(getattr(owner, name), fast, name))
    _dense_inits(_compressed._cs_matrix, _coo._coo_base)
    old = _extract.find
    _patch.rebind(old, _patch.wrap(old, _find, "find"))


def smoke():
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    d = rng.random((80, 70)) * (rng.random((80, 70)) < 0.2)
    for f in (sp.csr_array, sp.csc_array, sp.coo_array, sp.csr_matrix):
        a = f(d)
        a.argmax(axis=0), a.argmin(axis=1)
    r = rng.integers(0, 50, 500)
    c = sp.coo_array((rng.random(500), (r, rng.integers(0, 50, 500))), shape=(50, 50))
    c.sum_duplicates()
    sp.find(sp.csr_array(d))
