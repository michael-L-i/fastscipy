"""Binary morphology, binary dilation to a fixed point (binary_fill_holes, binary_propagation)
and the Euclidean distance transform."""
import functools
import numbers
import operator

import numpy as np
from scipy.ndimage import _morphology, _ni_support
import scipy.ndimage as ndi

from . import _patch

from ._ndimage_seam import accelerated, core

_BYTE = (np.dtype(bool), np.dtype(np.uint8), np.dtype(np.int8))


def _as_bytes(a):
    """C-contiguous uint8 array holding (a != 0), or None."""
    if a.dtype.kind == "c":
        return None
    if a.dtype in _BYTE:
        a = np.ascontiguousarray(a)
        return a.view(np.uint8)
    return np.ascontiguousarray(a != 0).view(np.uint8)


@accelerated("binary_erosion")
def binary_erosion(input, structure, mask, output, border_value, origin, invert, cit, return_coordinates):
    if not isinstance(input, np.ndarray) or not isinstance(output, np.ndarray):
        return NotImplemented
    if input.ndim == 0 or input.size == 0 or input.shape != output.shape:
        return NotImplemented
    if not output.flags.writeable or output.dtype.kind == "c":
        return NotImplemented
    if not isinstance(border_value, (numbers.Integral, bool)) or not isinstance(invert, (numbers.Integral, bool)):
        return NotImplemented
    origin = [int(o) for o in origin]
    if len(origin) != input.ndim or structure.ndim != input.ndim:
        return NotImplemented
    inp = _as_bytes(input)
    st = _as_bytes(np.asarray(structure))
    if inp is None or st is None:
        return NotImplemented
    m = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != input.shape:
            return NotImplemented
        m = _as_bytes(mask)
        if m is None:
            return NotImplemented
    direct = output.dtype in _BYTE and output.flags.c_contiguous
    out = output.view(np.uint8) if direct else np.empty(output.shape, np.uint8)
    changed, lst = core.binary_erosion_pass(
        inp, st, m, out, bool(border_value), origin, bool(invert), bool(cit), bool(return_coordinates)
    )
    if not direct:
        output[...] = out
    if return_coordinates:
        return int(changed), lst
    return int(changed)


@accelerated("binary_erosion2")
def binary_erosion2(array, structure, mask, niter, origin, invert, cobj):
    if not isinstance(cobj, np.ndarray):
        return NotImplemented  # a C capsule from SciPy's own pass 1
    origin = [int(o) for o in origin]
    st = _as_bytes(np.asarray(structure))
    direct = array.dtype in _BYTE and array.flags.c_contiguous and array.flags.writeable
    arr = array.view(np.uint8) if direct else np.ascontiguousarray(array != 0).view(np.uint8)
    m = None
    mdirect = False
    if mask is not None:
        mdirect = mask.dtype == np.int8 and mask.flags.c_contiguous
        m = mask if mdirect else np.ascontiguousarray(mask, dtype=np.int8).copy()
    core.binary_erosion2(arr, st, m, int(niter), origin, bool(invert), cobj)
    if not direct:
        array[...] = arr
    if mask is not None and not mdirect:
        mask[...] = m  # SciPy leaves its working values in the (int8) mask array
    return None


@accelerated("euclidean_feature_transform")
def euclidean_feature_transform(input, sampling, ft):
    if not (isinstance(input, np.ndarray) and isinstance(ft, np.ndarray)):
        return False
    if input.dtype != np.int8 or not input.flags.c_contiguous or input.ndim == 0 or input.size == 0:
        return False
    if ft.dtype != np.int32 or not ft.flags.c_contiguous or not ft.flags.writeable \
            or ft.shape != (input.ndim,) + input.shape:
        return False
    if sampling is not None and not (sampling.dtype == np.float64 and sampling.flags.c_contiguous
                                     and sampling.size == input.ndim):
        return False
    core.euclidean_feature_transform(input, sampling, ft)
    return True


def _distance_transform_edt(input, sampling=None, return_distances=True, return_indices=False,
                            distances=None, indices=None):
    """distance_transform_edt with the numpy post-processing fused into one Rust pass."""
    if not (isinstance(input, np.ndarray) and distances is None and indices is None
            and input.dtype.kind in "biuf" and input.size > 0):
        return NotImplemented
    _morphology._distance_transform_arg_check(False, False, return_distances, return_indices)
    inp = np.atleast_1d(np.ascontiguousarray(input != 0).view(np.int8))
    if sampling is not None:
        sampling = np.ascontiguousarray(np.asarray(_ni_support._normalize_sequence(sampling, inp.ndim),
                                                   dtype=np.float64))
    ft = np.zeros((inp.ndim,) + inp.shape, dtype=np.int32)
    core.euclidean_feature_transform(inp, sampling, ft)
    result = []
    if return_distances:
        dt = np.empty(inp.shape, np.float64)
        core.edt_distances(ft, sampling, dt)
        result.append(dt)
    if return_indices:
        result.append(ft)
    return tuple(result) if len(result) == 2 else result[0]


@functools.lru_cache(maxsize=256)
def _offsets(sbytes, sshape, origin, erode):
    """Set of q - p such that input pixel p sets output pixel q in one dilation step."""
    structure = np.frombuffer(sbytes, dtype=np.bool_).reshape(sshape)
    shp = tuple(4 * s + 2 * abs(o) + 3 for s, o in zip(sshape, origin))
    c = tuple(s // 2 for s in shp)
    delta = np.zeros(shp, dtype=np.bool_)
    delta[c] = True
    out = erode(delta, structure, 1, None, None, 0, list(origin), 1, False, None)
    return tuple(map(tuple, (np.argwhere(out) - np.array(c)).tolist()))


def _fixed_point_dilation(erode):
    """Fast path for ``_binary_erosion(..., iterations < 1, invert=True)``: dilation until nothing changes."""
    def fast(input, structure, iterations, mask, output, border_value, origin, invert, brute_force, axes):
        if iterations > 0 or not invert:
            return NotImplemented
        if axes is not None and tuple(axes) != tuple(range(np.ndim(input))):
            return NotImplemented
        if not isinstance(input, np.ndarray) or input.ndim == 0 or input.size == 0:
            return NotImplemented
        if input.dtype.kind not in "biuf" or not input.dtype.isnative:
            return NotImplemented
        ndim = input.ndim
        if structure is None:
            structure = _morphology.generate_binary_structure(ndim, 1)
        structure = np.asarray(structure, dtype=bool)
        if structure.ndim != ndim or structure.size < 1 or not np.all(np.array(structure.shape) <= 9):
            return NotImplemented
        try:
            org = tuple(operator.index(o) for o in _ni_support._normalize_sequence(origin, ndim))
        except TypeError:
            return NotImplemented
        if any(o < -(s // 2) or o > (s - 1) // 2 for o, s in zip(org, structure.shape)):
            return NotImplemented     # SciPy raises for these
        m = None
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != input.shape or mask.dtype not in _BYTE:
                return NotImplemented
            m = np.ascontiguousarray(mask).view(np.uint8)
        if not _morphology._center_is_true(structure, list(org)):
            return NotImplemented     # SciPy may loop forever
        offs = _offsets(np.ascontiguousarray(structure).tobytes(), structure.shape, org, erode)
        if (0,) * ndim not in offs:
            return NotImplemented
        if isinstance(output, np.ndarray):
            if output.shape != input.shape or output.dtype.kind == "c":
                return NotImplemented
            out = output
        else:
            out = np.zeros(input.shape, dtype=bool)
        inp = np.ascontiguousarray(input if input.dtype == np.bool_ else input != 0).view(np.uint8)
        direct = (out.dtype in _BYTE and out.flags.c_contiguous and out.flags.writeable
                  and not np.may_share_memory(out, input)
                  and not (mask is not None and np.may_share_memory(out, mask)))
        res = out.view(np.uint8) if direct else np.empty(input.shape, np.uint8)
        core.fill_core(inp, m, [list(o) for o in offs], bool(border_value), res, True)
        if not direct:
            out[...] = res.view(np.bool_)
        return out
    return fast


def patch():
    erode = _morphology._binary_erosion
    _patch.replace(_morphology, "_binary_erosion",
                   _patch.wrap(erode, _fixed_point_dilation(erode), "fixed_point_dilation"))
    old = ndi.distance_transform_edt
    new = _patch.wrap(old, _distance_transform_edt, "distance_transform_edt")
    _patch.rebind(old, new)
