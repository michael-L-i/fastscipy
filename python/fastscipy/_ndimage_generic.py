"""generic_filter with a numpy reducer (np.mean, np.sum, np.max, np.min, np.median, np.std, np.var, np.ptp)."""
import numpy as np

from ._ndimage_seam import accelerated, core

# callable -> kernel code
_REDUCERS = {np.mean: 0, np.sum: 1, np.max: 2, np.min: 3, np.median: 4, np.std: 5, np.var: 6, np.ptp: 7}

# scipy.ndimage._ni_support._extend_mode_to_code -> np.pad mode
_PAD = {0: "edge", 1: "wrap", 2: "symmetric", 3: "reflect", 4: "constant", 5: "wrap", 6: "constant"}


@accelerated("generic_filter")
def generic_filter(input, function, footprint, output, mode, cval, origins, extra_arguments, extra_keywords):
    try:
        which = _REDUCERS.get(function)
    except TypeError:
        return NotImplemented
    if which is None or extra_arguments or extra_keywords:
        return NotImplemented
    if not isinstance(input, np.ndarray) or not isinstance(output, np.ndarray):
        return NotImplemented
    if input.ndim == 0 or input.size == 0 or input.shape != output.shape or not output.flags.writeable:
        return NotImplemented
    if mode not in _PAD or output.dtype.kind not in "iuf" or output.dtype.itemsize == 2 and output.dtype.kind == "f":
        return NotImplemented
    dt = input.dtype
    if dt == np.bool_:
        src = input.view(np.uint8) if input.flags.c_contiguous else np.ascontiguousarray(input).view(np.uint8)
    elif dt.kind in "iuf" and dt.isnative and dt.itemsize in (1, 2, 4, 8) and not (dt.kind == "f" and dt.itemsize == 2):
        src = input
    else:
        return NotImplemented
    fp = np.ascontiguousarray(footprint).view(np.uint8) if footprint.dtype == np.bool_ else \
        np.ascontiguousarray(footprint != 0).view(np.uint8)
    if fp.ndim != input.ndim or fp.sum() == 0:
        return NotImplemented
    cval = float(cval)
    if mode in (4, 6):
        # the C buffer keeps cval as a double: pad in float64 unless cval survives the input dtype
        if np.array(cval).astype(src.dtype).astype(np.float64) != cval and not np.isnan(cval):
            src = src.astype(np.float64)
        elif np.isnan(cval) and src.dtype.kind != "f":
            src = src.astype(np.float64)
    widths = []
    for fs, o in zip(fp.shape, origins):
        orgn = fs // 2 + int(o)
        if orgn < 0 or orgn >= fs:
            return NotImplemented
        widths.append((orgn, fs - 1 - orgn))
    if mode in (4, 6):
        padded = np.pad(src, widths, mode="constant", constant_values=cval)
    else:
        padded = np.pad(src, widths, mode=_PAD[mode])
    padded = np.ascontiguousarray(padded)
    direct = output.dtype == np.float64 and output.flags.c_contiguous
    out = output if direct else np.empty(output.shape, np.float64)
    if not core.generic_reduce(padded, fp, out, which):
        return NotImplemented
    if not direct:
        if output.dtype.kind in "iu":
            # out-of-range double->int casts are undefined in C and differ between C and numpy; only
            # the in-range (truncating) cast is guaranteed identical, so decline otherwise
            info = np.iinfo(output.dtype)
            if not np.all((out >= float(info.min)) & (out < float(info.max) + 1.0)):
                return NotImplemented
        output[...] = out  # numpy double -> T cast, like the C `(T)` cast
    return None
