"""Rust kernels for ``scipy.signal``: peak prominences and widths, ``dlsim``, direct 2-D correlation.

Each wraps a seam below the public function (SciPy keeps its argument handling, warnings
and errors) or, for ``dlsim``, replays SciPy's setup and replaces its Python state loop.
"""

import warnings

import numpy as np

from . import _native, _threads

NI = NotImplemented
F8 = np.dtype(np.float64)
# smallest calls worth a Rust call; tests lower them to reach tiny shapes
MIN_PEAKS, MIN_SAMPLES, MIN_WIDTHS, MIN_STEPS, MIN_CORR = 16, 4096, 1 << 15, 2000, 1 << 14
_sig = _native.signal
_fused_width = _fused_corr = _fused_conv = None
_blas_available = False


def configure(widths, correlate, convolve):
    """Read contraction rules from the installed SciPy kernels, not the host CPU."""
    global _fused_width, _fused_corr, _fused_conv, _blas_available
    e = 2.0**-27
    pk, lo, hi = np.array([1], np.intp), np.array([0], np.intp), np.array([2], np.intp)
    height = widths(np.array([0.0, 1.0, 0.0]), pk, 1 - e, np.array([1 + e]), lo, hi)[1][0]
    _fused_width = {0.0: False, 2.0**-54: True}.get(height)
    a, k = np.array([[-1.0, 1 + e]]), np.array([[1.0, 1 - e]])
    out = np.empty((1, 1))
    correlate(a, k, out, 0)
    _fused_corr = {0.0: False, -(2.0**-54): True}.get(out[0, 0])
    tail = convolve(a, k, 0, 0, 0, 0.0)[0, 0]
    group = convolve(np.array([[1 + e, -1.0, 0.0, 0.0]]), np.array([[1 - e, 1.0, 0.0, 0.0]]), 0, 0, 0, 0.0)[0, 0]
    _fused_conv = {0.0: False, -(2.0**-54): True}.get(tail) if tail == group else None
    _blas_available = _sig.init_blas()


def _arr(a, dtype, ndim=1):
    return type(a) is np.ndarray and a.dtype == dtype and a.ndim == ndim and a.flags.c_contiguous


def _warn(message):
    from scipy.signal._peak_finding_utils import PeakPropertyWarning

    warnings.warn(message, PeakPropertyWarning, stacklevel=4)  # SciPy's stacklevel=2 is the caller of find_peaks etc.


def peak_prominences(x, peaks, wlen):
    if not (
        _arr(x, F8)
        and _arr(peaks, np.intp)
        and wlen.__class__ is int
        and peaks.size >= MIN_PEAKS
        and x.size >= MIN_SAMPLES
    ):
        return NI
    if wlen >= 2 and (_threads.get_num_threads() < 2 or peaks.size * wlen < 1 << 20):
        return NI  # a windowed scan is SciPy's own scan: only the threaded one is faster
    res = _sig.peak_prominences(x, peaks, wlen)
    if res is None:
        return NI
    if (res[0] == 0).any():
        _warn("some peaks have a prominence of 0")
    return res


def peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases):
    if not (
        _arr(x, F8)
        and _arr(peaks, np.intp)
        and _arr(prominences, F8)
        and _arr(left_bases, np.intp)
        and _arr(right_bases, np.intp)
        and rel_height.__class__ in (float, int)
        and peaks.size >= MIN_WIDTHS
        and _threads.get_num_threads() > 1
        and _fused_width is not None
    ):
        return NI  # serial, it is SciPy's own scan
    res = _sig.peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases, _fused_width)
    if res is None or rel_height < 0:
        return NI
    if (res[0] == 0).any():
        _warn("some peaks have a width of 0")
    return tuple(res)


def correlateND(in1, in2, out, val):
    if not (
        _arr(in1, F8, 2)
        and _arr(out, F8, 2)
        and type(in2) is np.ndarray
        and in2.dtype == F8
        and in2.ndim == 2
        and val in (0, 1, 2)
        and in1.size * in2.size >= MIN_CORR
        and _fused_corr is not None
    ):
        return NI
    (h, w), (kh, kw) = in1.shape, in2.shape
    if out.shape != ((h, w) if val == 2 else (h - kh + 1, w - kw + 1)) or not out.size or not in2.size:
        return NI
    if not np.isfinite(in2).all():
        return NI  # zeros outside the input times inf or NaN would not be skipped
    # first input row/column read by output (0, 0), relative to the kernel's last tap
    off = (0, 0) if val == 0 else (kh // 2, kw // 2) if val == 1 else (kh - 1, kw - 1)
    _sig.correlate2d(in1, np.ascontiguousarray(in2), out, *off, _fused_corr)
    return out


def convolve2d(in1, in2, flip=1, mode=2, boundary=0, fillvalue=None):
    """``_sigtools._convolve2d``, the kernel of ``convolve2d`` and ``correlate2d`` (flip=0)."""
    if not (
        type(in1) is np.ndarray
        and type(in2) is np.ndarray
        and in1.dtype == F8 == in2.dtype
        and in1.ndim == in2.ndim == 2
        and flip.__class__ is int
        and mode in (0, 1, 2)
        and boundary in (0, 4, 8)
        and in1.size * in2.size >= MIN_CORR
        and _fused_conv is not None
    ):
        return NI
    if fillvalue is not None and boundary == 0 and fillvalue.__class__ not in (int, float, np.float64):
        return NI
    shape = [(a - b + 1, a, a + b - 1)[mode] for a, b in zip(in1.shape, in2.shape)]
    if min(shape) <= 0 or not in2.size:
        return NI
    out = np.empty(shape)
    fill = float(fillvalue) if fillvalue is not None and boundary == 0 else 0.0
    _sig.convolve2d(
        np.ascontiguousarray(in1), np.ascontiguousarray(in2), out, flip != 0, mode, boundary, fill, _fused_conv
    )
    return out


_gemv = {}
_GEMV_CALIBRATION_BYTES = 8 << 20
_GEMV_TRIALS = 600


def _gemv_matches(shape, matrix=None):
    """Select a matching Accelerate model or installed BLAS kernel for this shape."""
    if min(shape) < 1 or max(shape) > np.iinfo(np.int32).max:
        return None
    if shape not in _gemv:
        rows, cols = shape
        # References temporarily coexist as a list of rows and a stacked array.
        vector_bytes = 8 * (cols + 2 * rows)
        if vector_bytes > _GEMV_CALIBRATION_BYTES:
            return None
        trial_bytes = 8 * rows * cols + vector_bytes
        use_caller = trial_bytes > _GEMV_CALIBRATION_BYTES
        if use_caller and matrix is None:
            return None
        batch = 1 if use_caller else max(1, _GEMV_CALIBRATION_BYTES // trial_bytes)
        candidates = [0, 1, 2] if _blas_available else [0]
        rng = np.random.default_rng(12345)
        for start in range(0, _GEMV_TRIALS, batch):
            count = min(batch, _GEMV_TRIALS - start)
            ms = matrix[None] if use_caller else rng.standard_normal((count, *shape))
            vs = rng.standard_normal((count, cols))
            with np.errstate(all="raise"):
                refs = np.stack([np.dot(m, v) for m, v in zip(ms, vs)])
            candidates = [mode for mode in candidates if _sig.gemv_matches(ms, vs, refs, mode)]
            if not candidates:
                break
            # Free each batch before allocating the next one.
            del ms, vs, refs
        _gemv[shape] = candidates[0] if candidates else None
    return _gemv[shape]


def dlsim(system, u, t=None, x0=None):
    """SciPy's ``dlsim`` with its per-step ``np.dot`` loop in Rust."""
    from scipy.signal import _ltisys as lti

    try:
        with warnings.catch_warnings(record=True) as caught:  # SciPy re-runs it and warns itself
            warnings.simplefilter("always")
            if isinstance(system, lti.lti):
                return NI
            if not isinstance(system, lti.dlti):
                system = lti.dlti(*system[:-1], dt=system[-1])
            is_ss = isinstance(system, lti.StateSpace)
            ss = system._as_ss()
            u = np.atleast_1d(u)
            if u.ndim == 1:
                u = np.atleast_2d(u).T
            if t is None:
                steps, stop = len(u), (len(u) - 1) * ss.dt
            else:
                stop = t[-1]
                steps = int(np.floor(stop / ss.dt)) + 1
            if steps < MIN_STEPS or not all(_arr(m, F8, 2) for m in (ss.A, ss.B, ss.C, ss.D)):
                return NI
            xout, yout = np.zeros((steps, ss.A.shape[0])), np.zeros((steps, ss.C.shape[0]))
            tout = np.linspace(0.0, stop, num=steps)
            xout[0, :] = np.zeros((ss.A.shape[1],)) if x0 is None else np.asarray(x0)
            if t is None:
                u_dt = u
            else:
                u_dt = lti.make_interp_spline(t, u[:, np.newaxis] if u.ndim == 1 else u, k=1)(tout)
            u_dt = np.asarray(u_dt)
            if not (
                _arr(u_dt, F8, 2)
                and u_dt.shape == (steps, ss.B.shape[1])
                and all(_gemv_matches(m.shape, m) is not None for m in (ss.A, ss.B, ss.C, ss.D))
            ):
                return NI
            if caught:
                return NI
    except Exception:
        return NI  # SciPy raises its own error
    _sig.dlsim_loop(ss.A, ss.B, ss.C, ss.D, u_dt, xout, yout, [_gemv[m.shape] for m in (ss.A, ss.B, ss.C, ss.D)])
    return (tout, yout, xout) if is_ss else (tout, yout)
