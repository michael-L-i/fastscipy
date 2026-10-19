"""Fast paths for _nd_image.{spline_filter1d, zoom_shift, geometric_transform}.

Each function receives exactly the arguments of SciPy's C function. It
normalizes them cheaply (contiguous input, a temporary when the output is not
C-contiguous), calls the Rust kernel, and returns True; or returns False
without touching the output when the call is outside what the kernels cover
(Python mappings, unsupported dtypes, overlapping arrays, odd argument types),
so SciPy's C code runs instead.
"""

import operator

import numpy as np

from ._ndimage_seam import accelerated, core

# Native-byte-order real types the kernels read and write.
_TYPES = frozenset(
    np.dtype(t)
    for t in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64)
)

# Coordinate dtypes SciPy's C code accepts (converted to float64 here, which
# is the same conversion the C code performs per element).
_COORD_TYPES = _TYPES | {np.dtype(np.bool_)}


def _int(v):
    """The C int argument, or None when SciPy's parser would not accept it."""
    try:
        return operator.index(v)
    except TypeError:
        return None


def _float(v):
    """The C double argument ("d" format), or None when not convertible."""
    if isinstance(v, (str, bytes)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _usable_pair(input, output):
    return (
        isinstance(input, np.ndarray)
        and isinstance(output, np.ndarray)
        and input.dtype in _TYPES
        and output.dtype in _TYPES
        and output.flags.writeable
    )


def _contig(a):
    if a.flags.c_contiguous and a.flags.aligned:
        return a
    return np.array(a, order="C", copy=True)


def _target(output):
    """Where the kernel writes: the output itself, or a C-contiguous temporary."""
    if output.flags.c_contiguous and output.flags.aligned:
        return output
    return np.empty(output.shape, output.dtype)


def _finish(ok, dst, output):
    if ok and dst is not output:
        output[...] = dst
    return ok


def _f64_vector(a, n):
    return isinstance(a, np.ndarray) and a.dtype == np.float64 and a.ndim == 1 and a.shape[0] == n


@accelerated("spline_filter1d")
def spline_filter1d(input, order, axis, output, mode):
    order, axis, mode = _int(order), _int(axis), _int(mode)
    if None in (order, axis, mode) or not _usable_pair(input, output):
        return False
    if not (2 <= order <= 5 and 0 <= mode <= 6) or input.ndim == 0:
        return False
    if input.shape != output.shape or not 0 <= axis < input.ndim:
        return False
    if np.may_share_memory(input, output):
        # In place (spline_filter does this for every axis after the first):
        # each line is read completely before it is written, so this equals
        # filtering a copy. Partial overlaps are left to SciPy.
        same = (
            input.__array_interface__["data"][0] == output.__array_interface__["data"][0]
            and input.strides == output.strides
            and input.dtype == output.dtype
        )
        if not same:
            return False
        if output.flags.c_contiguous and output.flags.aligned:
            return core.interp_spline_filter1d(output, order, axis, output, mode)
        input = np.array(input, order="C", copy=True)
    src = _contig(input)
    dst = _target(output)
    return _finish(core.interp_spline_filter1d(src, order, axis, dst, mode), dst, output)


@accelerated("zoom_shift")
def zoom_shift(input, zoom, shift, output, order, mode, cval, npad, grid_mode):
    order, mode, npad, grid_mode = _int(order), _int(mode), _int(npad), _int(grid_mode)
    cval = _float(cval)
    if None in (order, mode, npad, grid_mode, cval) or not _usable_pair(input, output):
        return False
    if input.ndim == 0 or output.ndim != input.ndim or output.size == 0:
        return False
    if not (0 <= order <= 5 and 0 <= mode <= 6 and npad >= 0):
        return False
    for v in (zoom, shift):
        if v is not None and not _f64_vector(v, input.ndim):
            return False
    if np.may_share_memory(input, output):
        return False
    zoom = None if zoom is None else _contig(zoom)
    shift = None if shift is None else _contig(shift)
    src = _contig(input)
    dst = _target(output)
    ok = core.interp_zoom_shift(src, zoom, shift, dst, order, mode, cval, npad, bool(grid_mode))
    return _finish(ok, dst, output)


@accelerated("geometric_transform")
def geometric_transform(
    input, mapping, coordinates, matrix, offset, output, order, mode, cval, npad, extra_arguments, extra_keywords
):
    if mapping is not None:
        return False  # Python callables and LowLevelCallables stay in SciPy
    order, mode, npad = _int(order), _int(mode), _int(npad)
    cval = _float(cval)
    if None in (order, mode, npad, cval) or not _usable_pair(input, output):
        return False
    if input.ndim == 0 or output.ndim == 0 or output.size == 0:
        return False
    if not (0 <= order <= 5 and 0 <= mode <= 6 and npad >= 0):
        return False
    if np.may_share_memory(input, output):
        return False
    if matrix is not None:
        if not (
            isinstance(matrix, np.ndarray)
            and matrix.dtype == np.float64
            and matrix.shape == (input.ndim, output.ndim)
            and _f64_vector(offset, input.ndim)
        ):
            return False
        src = _contig(input)
        dst = _target(output)
        ok = core.interp_affine(src, _contig(matrix), _contig(offset), dst, order, mode, cval, npad)
        return _finish(ok, dst, output)
    if coordinates is not None:
        if not (
            isinstance(coordinates, np.ndarray)
            and coordinates.dtype in _COORD_TYPES
            and coordinates.shape == (input.ndim,) + output.shape
        ):
            return False
        if np.may_share_memory(coordinates, output):
            return False
        coords = np.ascontiguousarray(coordinates, dtype=np.float64)
        src = _contig(input)
        dst = _target(output)
        ok = core.interp_map_coordinates(src, coords, dst, order, mode, cval, npad)
        return _finish(ok, dst, output)
    return False
