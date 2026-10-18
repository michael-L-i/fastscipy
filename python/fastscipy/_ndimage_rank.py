"""Fast paths for _nd_image.{rank_filter, min_or_max_filter, min_or_max_filter1d}.

Each function receives exactly the arguments of SciPy's C function. It
normalizes them cheaply (C-contiguous native input, bool viewed as uint8, a
temporary output when the given one is non-contiguous, byte-swapped or
overlaps the input) and calls the Rust kernel, which returns False without
writing anything when the call is outside what it reproduces exactly; SciPy's
C code then runs.
"""
import numpy as np

from ._ndimage_seam import accelerated, core


def _supported_dtype(dt):
    if dt.kind in "iu":
        return dt.itemsize in (1, 2, 4, 8)
    if dt.kind == "f":
        return dt.itemsize in (4, 8)
    return False


def _as_kernel_dtype(a):
    """View/convert ``a`` to what the kernels take, or None.

    SciPy's C code treats bool as ``unsigned char`` (values copied as bytes,
    constants cast with C semantics), exactly like uint8, so bool arrays are
    viewed as uint8. Non-native byte order is converted to native first, as
    SciPy does.
    """
    dt = a.dtype
    if dt == np.bool_:
        return a.view(np.uint8)
    if not _supported_dtype(dt):
        return None
    if not dt.isnative:
        return a.astype(dt.newbyteorder("="))
    return a


def _prepare(input, output, inplace_ok=False):
    """Return (inp, out, target) or None to decline.

    ``inp`` is C-contiguous. ``out`` is the array the kernel writes: the
    output itself when that is C-contiguous and native, else a temporary that
    the caller copies into ``target`` on success (``target`` is None when no
    copy is needed). ``inp`` and ``out`` never overlap, except when
    ``inplace_ok`` and they are the same buffer with the same layout (the 1-D
    kernel copies each line before writing it).
    """
    if not isinstance(input, np.ndarray) or not isinstance(output, np.ndarray):
        return None
    if input.ndim == 0 or input.size == 0 or input.shape != output.shape:
        return None
    if not output.flags.writeable:
        return None
    inp = _as_kernel_dtype(input)
    if inp is None:
        return None
    target = output.view(np.uint8) if output.dtype == np.bool_ else output
    if not _supported_dtype(target.dtype):
        return None
    if not inp.flags.c_contiguous:
        inp = np.ascontiguousarray(inp)
    if not target.dtype.isnative or not target.flags.c_contiguous:
        # results go to a fresh array; input is fully read before copy-back
        return inp, np.empty(target.shape, target.dtype.newbyteorder("=")), target
    if np.may_share_memory(inp, target):
        same_buffer = (
            inp.dtype == target.dtype
            and inp.__array_interface__["data"][0] == target.__array_interface__["data"][0]
        )
        if not (inplace_ok and same_buffer):
            inp = inp.copy()
    return inp, target, None


_LIMIT = 2**31


def _small(*values):
    """All integers fit comfortably in the kernels' integer arguments."""
    return all(-_LIMIT < v < _LIMIT for v in values)


def _origins(origins, ndim):
    try:
        o = [int(v) for v in np.atleast_1d(np.asarray(origins)).tolist()]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(o) != ndim or not _small(*o):
        return None
    return o


def _footprint(footprint):
    if not isinstance(footprint, np.ndarray) or footprint.dtype != np.bool_:
        return None
    return np.ascontiguousarray(footprint)


@accelerated("rank_filter")
def rank_filter(input, rank, footprint, output, mode, cval, origins):
    try:
        rank, mode, cval = int(rank), int(mode), float(cval)
    except (TypeError, ValueError, OverflowError):
        return False
    if not _small(rank, mode):
        return False
    prep = _prepare(input, output)
    fp = _footprint(footprint)
    if prep is None or fp is None or fp.ndim != input.ndim:
        return False
    org = _origins(origins, input.ndim)
    if org is None:
        return False
    inp, out, target = prep
    if not core.rank_filter(inp, rank, fp, out, mode, cval, org):
        return False
    if target is not None:
        target[...] = out
    return True


@accelerated("min_or_max_filter")
def min_or_max_filter(input, footprint, structure, output, mode, cval, origins, minimum):
    if structure is not None:
        return False
    try:
        mode, cval, minimum = int(mode), float(cval), int(minimum)
    except (TypeError, ValueError, OverflowError):
        return False
    if not _small(mode, minimum):
        return False
    prep = _prepare(input, output)
    fp = _footprint(footprint)
    if prep is None or fp is None or fp.ndim != input.ndim:
        return False
    org = _origins(origins, input.ndim)
    if org is None:
        return False
    inp, out, target = prep
    if not core.min_or_max_filter(inp, fp, None, out, mode, cval, org, minimum):
        return False
    if target is not None:
        target[...] = out
    return True


@accelerated("min_or_max_filter1d")
def min_or_max_filter1d(input, size, axis, output, mode, cval, origin, minimum):
    try:
        size, axis, mode = int(size), int(axis), int(mode)
        cval, origin, minimum = float(cval), int(origin), int(minimum)
    except (TypeError, ValueError, OverflowError):
        return False
    if not _small(size, axis, mode, origin, minimum):
        return False
    prep = _prepare(input, output, inplace_ok=True)
    if prep is None:
        return False
    if axis < 0:
        axis += input.ndim
    inp, out, target = prep
    if not core.min_or_max_filter1d(inp, size, axis, out, mode, cval, origin, minimum):
        return False
    if target is not None:
        target[...] = out
    return True
