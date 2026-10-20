"""Fast paths for _nd_image.{correlate1d, correlate, uniform_filter1d}.

The Rust kernels reproduce SciPy's arithmetic exactly, but two details depend
on how the installed SciPy binary was compiled: whether its multiply-adds are
fused (and where), and what its out-of-range double -> integer casts return.
On first use, the kernels are checked against SciPy's own C functions on
small probes; the matching variant is used. Unmatched output conversions
use SciPy's C casts after the accelerated arithmetic; unmatched arithmetic
schemes leave the full call to SciPy.
"""

import os
import platform
import sys
import threading

import numpy as np

from ._ndimage_seam import accelerated, core, orig

# Native-endian real types the kernels handle (bool, float16, longdouble and
# complex are left to SciPy; complex inputs never reach _nd_image).
_CHARS = frozenset("bBhHiIlLqQfd")
_ONE = np.ones(1)


def _dtype_ok(dt):
    return dt.char in _CHARS and dt.isnative


# ---------------------------------------------------------------------------
# Calibration against the installed SciPy

# Correlate1D FMA schemes (see src/correlate/kern.rs).
_UNFUSED, _FUSED, _CLANG_ARM64 = 0, 1, 2

_cal = None
_cal_lock = threading.Lock()


def _after_fork():
    global _cal_lock
    _cal_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _probe_values():
    v = [
        0.0,
        -0.0,
        0.4,
        -0.4,
        0.9999,
        -0.9999,
        1.0,
        -1.0,
        -1.5,
        2.5,
        127.5,
        128.0,
        -128.5,
        -129.0,
        255.0,
        255.9,
        256.0,
        -255.9,
        300.7,
        -300.7,
        32767.9,
        32768.0,
        -32769.0,
        65535.5,
        65536.0,
        -65536.5,
        2.0**31 - 0.5,
        2.0**31,
        -(2.0**31) - 1.0,
        -(2.0**31),
        2.0**32 - 1.0,
        2.0**32,
        -(2.0**32),
        3e9,
        -3e9,
        1e10,
        -1e10,
        2.0**53,
        2.0**63,
        -(2.0**63),
        2.0**63 * 1.5,
        2.0**64,
        -(2.0**64),
        1e30,
        -1e30,
        np.inf,
        -np.inf,
        np.nan,
    ]
    return np.array(v, dtype=np.float64)


def _calibrate():
    rng = np.random.default_rng(20261015)
    x = rng.standard_normal(61) * 100.0
    reflect = 2

    def c1d_ok(scheme, w, data=x, out_dtype=np.float64):
        ref = np.zeros(data.shape, out_dtype)
        got = np.zeros(data.shape, out_dtype)
        orig.correlate1d(data, w, 0, ref, reflect, 0.0, 0)
        if not core.corr_correlate1d(data, w, 0, got, reflect, 0.0, 0, scheme):
            return False
        return _same(ref, got)

    probes = []
    for s1 in list(range(26)) + [31, 32, 33, 40]:
        half = rng.standard_normal(s1)
        probes.append(np.concatenate([half, rng.standard_normal(1), half[::-1]]))
        if s1:
            probes.append(np.concatenate([half, rng.standard_normal(1), -half[::-1]]))
    for fs in list(range(2, 26)) + [40, 41]:
        probes.append(rng.standard_normal(fs))

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        order = [_CLANG_ARM64, _FUSED, _UNFUSED] if sys.platform == "darwin" else [_FUSED, _CLANG_ARM64, _UNFUSED]
    else:
        order = [_UNFUSED, _FUSED, _CLANG_ARM64]
    scheme = next((s for s in order if all(c1d_ok(s, w) for w in probes)), None)

    def uniform_ok(size, data=x, out_dtype=np.float64):
        ref = np.zeros(data.shape, out_dtype)
        got = np.zeros(data.shape, out_dtype)
        orig.uniform_filter1d(data, size, 0, ref, reflect, 0.0, 0)
        if not core.corr_uniform_filter1d(data, size, 0, got, reflect, 0.0, 0):
            return False
        return _same(ref, got)

    uniform = all(uniform_ok(size) for size in list(range(1, 12)) + [31, 80])

    x2 = rng.standard_normal((13, 11)) * 100.0
    nd_probes = [rng.standard_normal(s) for s in [(3, 3), (5, 5), (2, 4), (1, 7), (6, 1)]]

    def nd_ok(fused, w, data=x2, out_dtype=np.float64):
        ref = np.zeros(data.shape, out_dtype)
        got = np.zeros(data.shape, out_dtype)
        origins = [0] * data.ndim
        orig.correlate(data, w, ref, reflect, 0.0, origins)
        if not core.corr_correlate(data, w, got, reflect, 0.0, origins, fused):
            return False
        return _same(ref, got)

    nd_order = [False, True] if machine in ("x86_64", "amd64", "i386", "x86") else [True, False]
    nd_fused = next((f for f in nd_order if all(nd_ok(f, w) for w in nd_probes)), None)

    # Output casts, including out-of-range values (undefined behaviour in C):
    # NI_LineBufferToArray's through a size-1 filter (an exact copy), and
    # NI_Correlate's through a 1-element kernel of weight 1.
    v = _probe_values()
    one = np.ones(1)
    line_types, safe_types = set(), set()
    for ch in _CHARS:
        dt = np.dtype(ch)
        if uniform_ok(1, v, dt) if uniform else scheme is not None and c1d_ok(scheme, one, v, dt):
            line_types.add(ch)
        if nd_fused is not None and nd_ok(nd_fused, one, v, dt):
            safe_types.add(ch)
    return {
        "scheme": scheme,
        "uniform": uniform,
        "nd_fused": nd_fused,
        "line_types": frozenset(line_types),
        "safe_types": frozenset(safe_types),
    }


_DISABLED = {"scheme": None, "uniform": False, "nd_fused": None, "line_types": frozenset(), "safe_types": frozenset()}


def calibration():
    """The kernel variants matching the installed SciPy (computed once)."""
    global _cal
    if _cal is None:
        with _cal_lock:
            if _cal is None:
                try:
                    with np.errstate(all="ignore"):
                        _cal = _calibrate()
                except Exception:  # noqa: BLE001 - never break a filter call; just use SciPy
                    _cal = _DISABLED
    return _cal


# ---------------------------------------------------------------------------
# Argument normalization


def _prepare(input, output, allow_inplace):
    """Return (input, target) for the kernels, or None to decline.

    ``target`` is ``output`` when it can be written directly, else a
    temporary that the caller copies into ``output`` afterwards.
    """
    if not (isinstance(input, np.ndarray) and isinstance(output, np.ndarray)):
        return None
    if not (_dtype_ok(input.dtype) and _dtype_ok(output.dtype)):
        return None
    if input.ndim == 0 or input.size == 0 or input.shape != output.shape:
        return None
    if not output.flags.writeable:
        return None
    if np.may_share_memory(input, output):
        # The 1-D filters read each line completely before writing it, so an
        # output that is exactly the input (same buffer and layout) gives the
        # same result as a separate one. Other overlaps depend on SciPy's
        # buffering order.
        identical = (
            allow_inplace
            and input.__array_interface__["data"][0] == output.__array_interface__["data"][0]
            and input.strides == output.strides
            and input.dtype == output.dtype
        )
        if not identical:
            return None
    flags = input.flags
    if not (flags.c_contiguous and flags.aligned):
        input = np.require(input, requirements=["C_CONTIGUOUS", "ALIGNED"])
    flags = output.flags
    if flags.c_contiguous and flags.aligned:
        target = output
    else:
        target = np.empty(output.shape, output.dtype)
    return input, target


def _out_ok(output, chars):
    dt = getattr(output, "dtype", None)
    return dt is not None and dt.char in chars and dt.isnative


def _finish(ok, output, target):
    if ok and target is not output:
        np.copyto(output, target, casting="no")
    return ok


def _finish_line(ok, output, target, axis, native_cast):
    if ok and not native_cast:
        # GCC can convert vector prefixes through int64 but scalar tails
        # through int32. Let its original line loop perform only the cast;
        # the expensive filter arithmetic still runs in the native kernel.
        orig.correlate1d(target, _ONE, axis, output, 2, 0.0, 0)
        return True
    return _finish(ok, output, target)


@accelerated("correlate1d")
def correlate1d(input, weights, axis, output, mode, cval, origin):
    cal = calibration()
    if cal["scheme"] is None:
        return False
    prep = _prepare(input, output, True)
    if prep is None:
        return False
    src, target = prep
    native_cast = _out_ok(output, cal["line_types"])
    if not native_cast:
        target = np.empty(output.shape, np.float64)
    try:
        ok = core.corr_correlate1d(src, weights, int(axis), target, int(mode), float(cval), int(origin), cal["scheme"])
    except (TypeError, ValueError, OverflowError):
        return False
    return _finish_line(ok, output, target, axis, native_cast)


@accelerated("uniform_filter1d")
def uniform_filter1d(input, size, axis, output, mode, cval, origin):
    cal = calibration()
    if not cal["uniform"]:
        return False
    prep = _prepare(input, output, True)
    if prep is None:
        return False
    src, target = prep
    native_cast = _out_ok(output, cal["line_types"])
    if not native_cast:
        target = np.empty(output.shape, np.float64)
    try:
        ok = core.corr_uniform_filter1d(src, int(size), int(axis), target, int(mode), float(cval), int(origin))
    except (TypeError, ValueError, OverflowError):
        return False
    return _finish_line(ok, output, target, axis, native_cast)


@accelerated("correlate")
def correlate(input, weights, output, mode, cval, origins):
    cal = calibration()
    if cal["nd_fused"] is None:
        return False
    if not isinstance(input, np.ndarray):
        return False
    # With all strides zero, SciPy's border flag value (max_size * max_stride + 1)
    # collides with valid coordinates; leave that quirk to SciPy.
    if input.size > 1 and not any(input.strides):
        return False
    prep = _prepare(input, output, False)
    if prep is None:
        return False
    src, target = prep
    native_cast = _out_ok(output, cal["safe_types"])
    if not native_cast:
        target = np.empty(output.shape, np.float64)
    try:
        origins = [int(o) for o in origins]
        ok = core.corr_correlate(src, weights, target, int(mode), float(cval), origins, cal["nd_fused"])
    except (TypeError, ValueError, OverflowError):
        return False
    if ok and not native_cast:
        # NI_Correlate's unsigned conversion differs from its line-buffer
        # conversion. A singleton footprint keeps that exact cast policy.
        orig.correlate(target, np.ones((1,) * input.ndim), output, 2, 0.0, [0] * input.ndim)
        return True
    return _finish(ok, output, target)
