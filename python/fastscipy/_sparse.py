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
# Windows uses 'q' for int64, while LP64 platforms commonly use 'l'. Keep
# character checks: dtype equality can also equate longdouble with float64.
_IDX, _VALS = "ilq", "fdilq"
_PAR = {"argminmax": 1_000_000, "sumdup": 100_000, "dense": 1 << 17}  # entries before threads pay off
_DENSE_MIN = 4096  # smaller dense inputs: SciPy


def _threads(work, key):
    return get_num_threads() if work >= _PAR[key] else 1


def _contig(*arrs):
    # Native kernels construct typed slices rather than NumPy buffer iterators.
    return all(
        a.__class__ is np.ndarray and a.ndim == 1 and a.flags.c_contiguous and a.flags.aligned and a.dtype.isnative
        for a in arrs
    )


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
        if (
            _patch.enabled
            and ip.__class__ is np.ndarray
            and ind.__class__ is np.ndarray
            and dat.__class__ is np.ndarray
        ):
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
            if (
                ip.__class__ is np.ndarray
                and ind.__class__ is np.ndarray
                and dat.__class__ is np.ndarray
                and ip.dtype.kind == "i"
                and ind.dtype.kind == "i"
                and ip.ndim == 1 == ind.ndim == dat.ndim
                and len(ip) == self._swap(self._shape_as_2d)[0] + 1
                and ip[0] == 0
                and len(ind) == len(dat)
                and ip[-1] <= len(ind)
            ):
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
    if (
        self.ndim != 2
        or self.dtype.char not in _VALS
        or axis not in (0, 1)
        or explicit.__class__ is not bool
        or (argminmax, compare) not in ((np.argmax, np.greater), (np.argmin, np.less))
    ):
        return NotImplemented
    mat = self.tocsc() if axis == 0 else self.tocsr()
    mat.sum_duplicates()
    ret_size, line_size = mat._swap(mat.shape)
    ip, ij, data = mat.indptr, mat.indices, mat.data
    if ip.dtype != ij.dtype or ip.dtype.char not in _IDX or not _contig(ip, ij, data):
        return NotImplemented
    ret = _native.sparse.argminmax(
        ip, ij, data, line_size, argminmax is np.argmax, explicit, _threads(len(data), "argminmax")
    )
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
    if (
        n == 0
        or row.dtype.char not in _IDX
        or row.dtype != col.dtype
        or len(row) != n
        or len(col) != n
        or m > 8 * n + 1_000_000
    ):
        return NotImplemented
    r = _native.sparse.sum_duplicates(row, col, data, m, _threads(n, "sumdup"))
    return NotImplemented if r is None else ((r[0], r[1]), r[2])


# ---- dense -> sparse ----------------------------------------------------------------------------


def _dense(arg1, shape, dtype, csc):
    """(indptr, indices, data) of a dense 2-D array when it is a plain case, else None."""
    if (
        arg1.__class__ is not np.ndarray
        or arg1.ndim != 2
        or shape is not None
        or dtype is not None
        or arg1.size < _DENSE_MIN
        or arg1.dtype.char not in _VALS
        or not arg1.dtype.isnative
        or not arg1.flags.aligned
        # Native offsets are expressed in elements, not arbitrary byte strides.
        or any(s % arg1.itemsize for s in arg1.strides)
        or max(arg1.shape) > _I32MAX
    ):
        return None
    return _native.sparse.dense_to_compressed(arg1, csc, _threads(arg1.size, "dense"))


def _dense_inits(cs_cls, coo_cls):
    orig_cs, orig_coo = cs_cls.__init__, coo_cls.__init__

    def cs_init(self, arg1, shape=None, dtype=None, copy=False, *extra, maxprint=None):
        if (
            extra
            or self.format not in ("csr", "csc")
            or (r := _dense(arg1, shape, dtype, self.format == "csc")) is None
        ):
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
        _patch.replace(
            cls, "__init__", _patch.wrap(cls.__init__, fast, name)
        )  # fast returns None when it built the object


# ---- find ---------------------------------------------------------------------------------------


def _find(A):
    from scipy.sparse import csr_array
    from scipy.sparse._base import sparray

    if not (isinstance(A, sparray) and A.ndim == 2 and A.format in ("csr", "csc", "coo")):
        return NotImplemented
    if A.format == "coo" and A.has_canonical_format:  # SciPy's own path has nothing to sort
        return NotImplemented
    dtypes = (A.row.dtype, A.col.dtype) if A.format == "coo" else (A.indices.dtype,) * 2
    if A.format != "csr" or not A.has_canonical_format:
        A = csr_array(A, copy=True)
        A.sum_duplicates()
    # The final masks already copy the outputs. Canonical CSR needs no
    # intermediate copies, and noncanonical inputs were detached above.
    A = A.tocoo(copy=False)
    row, col = A.row.astype(dtypes[0], copy=False), A.col.astype(dtypes[1], copy=False)
    keep = A.data != 0
    return row[keep], col[keep], A.data[keep]


# ---- structured construction -----------------------------------------------------------------


def _compressed_buffers_ok(block, major):
    """Bounds needed before handing plain compressed buffers to SciPy kernels."""
    minor = 1 - major
    return (
        _contig(block.indptr, block.indices, block.data)
        and block.dtype.char in "?bBhHiIlLqQfdFD"
        and block.dtype.isnative
        and block.indices.dtype.char in _IDX
        and block.indices.dtype.isnative
        and block.indptr.dtype == block.indices.dtype
        and len(block.data) == len(block.indices)
        and len(block.indptr) == block.shape[major] + 1
        and len(block.indptr) > 0
        and block.indptr[0] == 0
        and len(block.data) == block.indptr[-1]
        and not (block.indptr[1:] < block.indptr[:-1]).any()
        and block.indices.min(initial=0) >= 0
        and block.indices.max(initial=-1) < block.shape[minor]
    )


def _stack(blocks, axis, format, dtype):
    from scipy.sparse import _construct, csc_array, csc_matrix, csr_array, csr_matrix

    if type(blocks) not in (list, tuple) or not blocks or dtype is not None:
        return NotImplemented
    first = blocks[0]
    if type(first) not in (csr_array, csr_matrix, csc_array, csc_matrix):
        return NotImplemented
    fmt = first.format
    if format is not None and (type(format) is not str or format != fmt):
        return NotImplemented
    types = (csr_array, csr_matrix) if fmt == "csr" else (csc_array, csc_matrix)
    major = 0 if fmt == "csr" else 1
    if axis == major or len(blocks) != 2:
        return NotImplemented
    for block in blocks:
        if (
            type(block) not in types
            or block.ndim != 2
            or not _compressed_buffers_ok(block, major)
            or block.shape[1 - axis] != first.shape[1 - axis]
        ):
            return NotImplemented
    shape = list(first.shape)
    shape[axis] = sum(b.shape[axis] for b in blocks)
    if max(*shape, sum(len(b.data) for b in blocks)) > _I32MAX:
        return NotImplemented
    matrix = all(type(b) is types[1] for b in blocks)
    result = _construct._stack_along_minor_axis(blocks, axis)
    if matrix or type(result) is types[0]:
        return result
    return types[0]((result.data, result.indices, result.indptr), shape=result.shape)


def _hstack(blocks, format=None, dtype=None):
    return _stack(blocks, 1, format, dtype)


def _vstack(blocks, format=None, dtype=None):
    return _stack(blocks, 0, format, dtype)


def _block_diag(mats, format=None, dtype=None):
    from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix
    from scipy.sparse._sparsetools import csr_has_canonical_format

    if (
        type(mats) not in (list, tuple)
        or not mats
        or type(format) is not str
        or format not in ("csr", "csc")
        or dtype is not None
    ):
        return NotImplemented
    types = (csr_array, csr_matrix) if format == "csr" else (csc_array, csc_matrix)
    all_types = (csr_array, csr_matrix, csc_array, csc_matrix)
    major, minor = (0, 1) if format == "csr" else (1, 0)
    for block in mats:
        bmajor = 0 if type(block) in (csr_array, csr_matrix) else 1
        if (
            type(block) not in all_types
            or block.ndim != 2
            or not _compressed_buffers_ok(block, bmajor)
            or type(block) in (csr_matrix, csc_matrix)
            and block.indices.dtype != np.int32
            or not csr_has_canonical_format(block.shape[bmajor], block.indptr, block.indices)
        ):
            return NotImplemented
    shape = tuple(sum(b.shape[d] for b in mats) for d in (0, 1))
    nnz = sum(len(b.data) for b in mats)
    if max(*shape, nnz) > _I32MAX:
        return NotImplemented
    if any(b.format != format for b in mats):
        mats = [b.asformat(format) for b in mats]
    is_array = any(type(b) is types[0] for b in mats)
    constructor = types[0] if is_array else types[1]
    if nnz == 0:
        return constructor(shape, dtype=np.result_type(*(b.dtype for b in mats)))
    idx_dtype = np.int64 if any(b.indices.dtype == np.int64 for b in mats) else np.int32
    # Offset directly into the final buffers instead of allocating one
    # temporary per block and copying those again during concatenation.
    indices = np.empty(nnz, dtype=idx_dtype)
    pointers = np.empty(shape[major] + 1, dtype=idx_dtype)
    offset = used = position = 0
    for block in mats:
        end = used + len(block.data)
        stop = position + block.shape[major]
        np.add(block.indices, offset, out=indices[used:end])
        np.add(block.indptr[:-1], used, out=pointers[position:stop])
        offset += block.shape[minor]
        used, position = end, stop
    pointers[-1] = used
    result = constructor(
        (
            np.concatenate([b.data for b in mats]),
            indices,
            pointers,
        ),
        shape=shape,
    )
    result.has_canonical_format = True
    return result


def patch():
    _patch.require_validated("1.18.")
    from scipy.sparse import _coo, _compressed, _construct, _data, _extract

    _native.sparse  # noqa: B018  (fail early if the extension lacks the module)
    _small_calls()
    for owner, name, fast in (
        (_data._minmax_mixin, "_argminmax_axis", _argminmax_axis),
        (_coo._coo_base, "_sum_duplicates", _sum_duplicates),
    ):
        _patch.replace(owner, name, _patch.wrap(getattr(owner, name), fast, name))
    _dense_inits(_compressed._cs_matrix, _coo._coo_base)
    old = _extract.find
    _patch.rebind(old, _patch.wrap(old, _find, "find"))
    old = _construct.block_diag
    _patch.rebind(old, _patch.wrap(old, _block_diag, "block_diag"))
    for name, fast in (("hstack", _hstack), ("vstack", _vstack)):
        old = getattr(_construct, name)
        _patch.rebind(old, _patch.wrap(old, fast, name))


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
    for fmt in ("csr", "csc"):
        block = sp.csr_array(d).asformat(fmt)
        sp.block_diag([block, block], format=fmt)
        sp.hstack([block, block])
        sp.vstack([block, block])
