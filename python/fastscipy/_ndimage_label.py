"""Labelling and labelled measurements: label, find_objects, watershed_ift, and the min/max/position
and centre-of-mass reductions behind scipy.ndimage.measurements."""
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import _measurements, _ni_label

from . import _patch
from ._ndimage_seam import accelerated, core

_INT = "iu"


def _label(input, structure, output):
    if type(input) is not np.ndarray or type(output) is not np.ndarray:
        return NotImplemented
    if input.ndim == 0 or input.size == 0 or input.shape != output.shape:
        return NotImplemented
    if output.dtype.kind not in _INT or not output.dtype.isnative or not output.flags.writeable:
        return NotImplemented
    if not output.flags.c_contiguous:
        return NotImplemented   # SciPy's scan axis (and so its numbering) depends on the memory layout
    k, sz = input.dtype.kind, input.dtype.itemsize
    if not input.dtype.isnative or k not in "biuf" or (k == "f" and sz not in (4, 8)):
        return NotImplemented
    st = np.asarray(structure)
    if st.ndim != input.ndim or set(st.shape) > {3}:
        return NotImplemented
    st = st.astype(np.bool_)
    if not np.array_equal(st, st[(np.s_[::-1],) * st.ndim]):
        return NotImplemented
    if input.size >= 2**32 - 8 or input.shape[-1] >= 2**31:
        return NotImplemented
    inp = np.ascontiguousarray(input)
    if k == "b":
        inp = inp.view(np.uint8)
    if np.may_share_memory(inp, output):
        inp = inp.copy()
    n = core.label_core(inp, st.astype(np.uint8).ravel().tolist(), output)
    return NotImplemented if n is None else int(n)


@accelerated("find_objects")
def find_objects(input, max_label):
    if not (type(input) is np.ndarray and input.dtype.kind in _INT and input.dtype.isnative
            and input.ndim > 0 and input.size > 0 and 0 < max_label < 2**40):
        return False
    r = core.find_objects_core(np.ascontiguousarray(input), int(max_label))
    return False if r is None else r


def _nstrides(structure, shape):
    nd = len(shape)
    strides = [1] * nd
    for i in range(nd - 2, -1, -1):
        strides[i] = shape[i + 1] * strides[i + 1]
    coords = np.indices(structure.shape).reshape(nd, -1) - 1
    offs = (coords * np.array(strides)[:, None]).sum(axis=0)
    return [int(v) for v in offs[structure.ravel() & (offs != 0)]]   # C order, like the C odometer


@accelerated("watershed_ift")
def watershed_ift(input, markers, structure, output):
    if not all(type(a) is np.ndarray for a in (input, markers, structure, output)):
        return False
    if input.ndim == 0 or input.size == 0 or input.ndim > 7 or not input.shape == markers.shape == output.shape:
        return False
    if input.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        return False
    if markers.dtype.kind != "i" or output.dtype.kind != "i" or not output.dtype.isnative or not markers.dtype.isnative:
        return False
    if not output.flags.writeable or np.may_share_memory(output, input):
        return False
    if structure.shape != (3,) * input.ndim or structure.dtype != np.bool_ or input.size >= 2**32 - 2:
        return False
    mk = np.ascontiguousarray(np.ascontiguousarray(markers).astype(np.int32, copy=False))
    direct = output.dtype == np.int32 and output.flags.c_contiguous
    lab = output if direct else np.empty(input.shape, np.int32)
    if not core.watershed_core(np.ascontiguousarray(input), mk, _nstrides(structure, input.shape),
                               8 * output.dtype.itemsize, lab):
        return False
    if not direct:
        output[...] = lab
    return True


# --- labelled reductions ----------------------------------------------------------------------

def _prep(input, labels, index):
    """(values, labels64, index64, label min, label max, index shape, dtype), or None to decline."""
    if type(input) is not np.ndarray and not isinstance(input, (list, tuple)):
        return None
    input, labels, index = np.asarray(input), np.asarray(labels), np.asarray(index)
    if input.shape != labels.shape or input.size == 0 or input.ndim == 0:
        return None
    if labels.dtype.kind not in _INT or index.dtype.kind not in _INT or np.uint64 in (labels.dtype, index.dtype):
        return None
    dt = input.dtype
    if dt == np.bool_:
        values = np.ascontiguousarray(input).view(np.uint8)
    elif dt.kind in "iuf" and dt.isnative and dt.itemsize in (1, 2, 4, 8) and dt != np.float16:
        values = np.ascontiguousarray(input)
    else:
        return None
    lab = np.ascontiguousarray(labels, dtype=np.int64)
    idx = np.ascontiguousarray(index.ravel(), dtype=np.int64)
    return values, lab, idx, int(lab.min()), int(lab.max()), index.shape, dt


def _select(input, labels=None, index=None, find_min=False, find_max=False,
            find_min_positions=False, find_max_positions=False, find_median=False):
    if (find_median or labels is None or index is None or np.isscalar(index)
            or not (find_min or find_max or find_min_positions or find_max_positions)):
        return NotImplemented
    p = _prep(input, labels, index)
    r = p and core.label_select(*p[:5])
    if not r:
        return NotImplemented
    minv, minp, maxv, maxp, mintie, maxtie, absent = r
    # positions are only reproducible when the extreme is unique (SciPy's tie result is an unstable sort)
    if absent or (find_min_positions and mintie) or (find_max_positions and maxtie):
        return NotImplemented
    shape, boolean = p[5], p[6] == np.bool_
    out = []
    if find_min:
        out.append((minv.view(np.bool_) if boolean else minv).reshape(shape))
    if find_min_positions:
        out.append(minp.reshape(shape))
    if find_max:
        out.append((maxv.view(np.bool_) if boolean else maxv).reshape(shape))
    if find_max_positions:
        out.append(maxp.reshape(shape))
    return out


def _center_of_mass(input, labels=None, index=None):
    if labels is None or index is None or np.isscalar(index) or np.ndim(index) != 1:
        return NotImplemented
    inp = np.asarray(input)
    if not (inp.dtype.kind in "iuf" and inp.dtype.itemsize != 2 or inp.dtype == np.bool_):
        return NotImplemented
    p = _prep(input, labels, index)
    if p is None or len(p[2]) == 0:
        return NotImplemented
    r = core.label_com_sums(*p[:5])
    if r is None:
        return NotImplemented
    return [tuple(v) for v in np.array([r[d + 1] / r[0] for d in range(inp.ndim)]).T]


def patch():
    _patch.replace(_ni_label, "_label", _patch.wrap(_ni_label._label, _label, "label"))
    _patch.replace(_measurements, "_select", _patch.wrap(_measurements._select, _select, "_select"))
    old = ndi.center_of_mass
    _patch.rebind(old, _patch.wrap(old, _center_of_mass, "center_of_mass"))
