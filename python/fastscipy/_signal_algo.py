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


def _arr(a, dtype, ndim=1):
    return type(a) is np.ndarray and a.dtype == dtype and a.ndim == ndim and a.flags.c_contiguous


def _warn(message):
    from scipy.signal._peak_finding_utils import PeakPropertyWarning
    warnings.warn(message, PeakPropertyWarning, stacklevel=4)   # SciPy's stacklevel=2 is the caller of find_peaks etc.


def peak_prominences(x, peaks, wlen):
    if not (_arr(x, F8) and _arr(peaks, np.intp) and wlen.__class__ is int
            and peaks.size >= MIN_PEAKS and x.size >= MIN_SAMPLES):
        return NI
    if wlen >= 2 and (_threads.get_num_threads() < 2 or peaks.size * wlen < 1 << 20):
        return NI    # a windowed scan is SciPy's own scan: only the threaded one is faster
    res = _sig.peak_prominences(x, peaks, wlen)
    if res is None:
        return NI
    if (res[0] == 0).any():
        _warn("some peaks have a prominence of 0")
    return res


def peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases):
    if not (_arr(x, F8) and _arr(peaks, np.intp) and _arr(prominences, F8) and _arr(left_bases, np.intp)
            and _arr(right_bases, np.intp) and rel_height.__class__ in (float, int)
            and peaks.size >= MIN_WIDTHS and _threads.get_num_threads() > 1):
        return NI    # serial, it is SciPy's own scan
    res = _sig.peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases)
    if res is None or rel_height < 0:
        return NI
    if (res[0] == 0).any():
        _warn("some peaks have a width of 0")
    return tuple(res)


def correlateND(in1, in2, out, val):
    if not (_arr(in1, F8, 2) and _arr(out, F8, 2) and type(in2) is np.ndarray and in2.dtype == F8
            and in2.ndim == 2 and val in (0, 1, 2) and in1.size * in2.size >= MIN_CORR):
        return NI
    (h, w), (kh, kw) = in1.shape, in2.shape
    if out.shape != ((h, w) if val == 2 else (h - kh + 1, w - kw + 1)) or not out.size or not in2.size:
        return NI
    if not np.isfinite(in2).all():
        return NI    # zeros outside the input times inf or NaN would not be skipped
    # first input row/column read by output (0, 0), relative to the kernel's last tap
    off = (0, 0) if val == 0 else (kh // 2, kw // 2) if val == 1 else (kh - 1, kw - 1)
    _sig.correlate2d(in1, np.ascontiguousarray(in2), out, *off)
    return out


def convolve2d(in1, in2, flip=1, mode=2, boundary=0, fillvalue=None):
    """``_sigtools._convolve2d``, the kernel of ``convolve2d`` and ``correlate2d`` (flip=0)."""
    if not (type(in1) is np.ndarray and type(in2) is np.ndarray and in1.dtype == F8 == in2.dtype
            and in1.ndim == in2.ndim == 2 and flip.__class__ is int and mode in (0, 1, 2)
            and boundary in (0, 4, 8) and in1.size * in2.size >= MIN_CORR):
        return NI
    if fillvalue is not None and boundary == 0 and fillvalue.__class__ not in (int, float, np.float64):
        return NI
    shape = [(a - b + 1, a, a + b - 1)[mode] for a, b in zip(in1.shape, in2.shape)]
    if min(shape) <= 0 or not in2.size:
        return NI
    out = np.empty(shape)
    fill = float(fillvalue) if fillvalue is not None and boundary == 0 else 0.0
    _sig.convolve2d(np.ascontiguousarray(in1), np.ascontiguousarray(in2), out, flip != 0, mode, boundary, fill)
    return out


_gemv = {}


def _gemv_matches(shape):
    """Whether the Rust ``dgemv`` model equals ``np.dot`` for matrices of this shape (random trials)."""
    if shape not in _gemv:
        rng = np.random.default_rng(12345)
        ms, vs = rng.standard_normal((600, *shape)), rng.standard_normal((600, shape[1]))
        _gemv[shape] = _sig.gemv_matches(ms, vs, np.stack([np.dot(m, v) for m, v in zip(ms, vs)]))
    return _gemv[shape]


def dlsim(system, u, t=None, x0=None):
    """SciPy's ``dlsim`` with its per-step ``np.dot`` loop in Rust."""
    from scipy.signal import _ltisys as lti
    try:
        with warnings.catch_warnings(record=True) as caught:    # SciPy re-runs it and warns itself
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
            if not (_arr(u_dt, F8, 2) and u_dt.shape == (steps, ss.B.shape[1])
                    and all(_gemv_matches(m.shape) for m in (ss.A, ss.B, ss.C, ss.D))):
                return NI
            if caught:
                return NI
    except Exception:
        return NI    # SciPy raises its own error
    _sig.dlsim_loop(ss.A, ss.B, ss.C, ss.D, u_dt, xout, yout)
    return (tout, yout, xout) if is_ss else (tout, yout)
