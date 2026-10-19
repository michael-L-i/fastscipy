"""Fast paths for small ``scipy.signal`` calls (per-voxel and per-epoch loops).

They run the same compiled kernels with the same arithmetic as SciPy, minus
the Python layers (array-API dispatch, moveaxis/reshape copies, a design or
``lfilter_zi`` recomputed on every call). Each returns ``NotImplemented``
for anything else, and SciPy then decides every warning and error.

Caches are keyed on the exact bytes of their inputs and filled by SciPy's own
code, only when it emitted no warning, so a hit returns SciPy's own value.
"""

import warnings

import numpy as np
import scipy.fft
from scipy.linalg import _batched_linalg
from scipy.signal import ShortTimeFFT, _filter_design, _sigtools, get_window
from scipy.signal._sosfilt import _sosfilt

NI = NotImplemented
F8, F4 = np.dtype(np.float64), np.dtype(np.float32)
ndarray = np.ndarray
_linear_filter = _sigtools._linear_filter
orig = {}  # name -> SciPy's function, set by _signal.patch()
_RAISE = dict(divide="raise", over="raise", invalid="raise")  # overflow: let SciPy warn from its own frame


class _Uncacheable(Exception):
    pass


class _Cache(dict):
    def put(self, key, val):
        if len(self) >= 256:
            del self[next(iter(self))]
        self[key] = val
        return val


_zi_cache, _sos_cache, _iir_cache, _sft_cache, _A_cache = (_Cache() for _ in range(5))


def _quiet(fn, *a):
    """``fn(*a)`` and whether it emitted no warning (only then is the value cacheable)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = fn(*a)
    return not w, r


def _ro(a):
    a.setflags(write=False)
    return a


def _lf_zi(b, a):
    """Read-only ``lfilter_zi(b, a)`` for float64 1-D b, a; None if not cacheable."""
    key = b.tobytes(), a.tobytes()
    z = _zi_cache.get(key)
    if z is None:
        try:
            ok, z = _quiet(orig["lfilter_zi"], b, a)
        except Exception:
            return None
        if not ok:
            return None
        z = _zi_cache.put(key, _ro(z))
    return z


def _sos_info(sos):
    """(usable, ntaps, read-only sosfilt_zi) for a C-contiguous float64 (n, 6) sos."""
    key = sos.shape, sos.tobytes()
    info = _sos_cache.get(key)
    if info is None:
        info = False, 0, None
        try:
            _filter_design._validate_sos(sos)
            ns = sos.shape[0]
            ntaps = 2 * ns + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
            ok, zi = _quiet(orig["sosfilt_zi"], sos)
            if ok and zi.dtype == F8 and zi.shape == (ns, 2):
                info = True, int(ntaps), _ro(zi)
        except Exception:
            pass
        _sos_cache.put(key, info)
    return info


def _is(v, s):
    return type(v) is str and v == s


def _f8(x):
    return type(x) is ndarray and x.dtype == F8


def _last_axis(x, axis):
    return x.ndim in (1, 2) and type(axis) is int and axis in (-1, x.ndim - 1)


def _odd_ext(x, edge):
    """``odd_ext(x, edge, axis=-1)`` for float32/float64 1-D/2-D x of any strides."""
    n = x.shape[-1]
    ext = np.empty(x.shape[:-1] + (n + 2 * edge,), x.dtype)
    np.subtract(2 * x[..., :1], x[..., edge:0:-1], out=ext[..., :edge])
    ext[..., edge : edge + n] = x
    np.subtract(2 * x[..., -1:], x[..., -2 : -(edge + 2) : -1], out=ext[..., edge + n :])
    return ext


# --------------------------------------------------------------------- filters
def sosfilt(sos, x, axis=-1, zi=None):
    if not (
        type(x) is ndarray
        and x.dtype in (F8, F4)
        and _f8(sos)
        and x.size
        and _last_axis(x, axis)
        and sos.ndim == 2
        and sos.shape[1] == 6
        and sos.flags.c_contiguous
        and _sos_info(sos)[0]
    ):
        return NI
    ns = sos.shape[0]
    y = np.array(x, F8, order="C")  # the copy SciPy makes, filtered in place
    if zi is None:
        _sosfilt(sos, y.reshape(-1, x.shape[-1]), np.zeros((y.size // x.shape[-1], ns, 2)))
        return y
    if not (_f8(zi) and zi.shape == (ns,) + x.shape[:-1] + (2,)):
        return NI
    if x.ndim == 1:
        z = zi.copy().reshape(1, ns, 2)
        _sosfilt(sos, y.reshape(1, -1), z)
        return y, z.reshape(ns, 2)
    z = np.ascontiguousarray(np.moveaxis(zi, (0, 2), (-2, -1))).reshape(-1, ns, 2)
    _sosfilt(sos, y, z)
    return y, np.moveaxis(z.reshape(x.shape[0], ns, 2), (-2, -1), (0, 2))


def sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=None):
    if not (
        type(x) is ndarray
        and x.dtype in (F8, F4)
        and _f8(sos)
        and _last_axis(x, axis)
        and padlen is None
        and _is(padtype, "odd")
        and sos.ndim == 2
        and sos.shape[1] == 6
        and sos.flags.c_contiguous
    ):
        return NI
    ok, ntaps, zi = _sos_info(sos)
    edge = 3 * ntaps
    if not ok or x.shape[-1] <= edge or not x.size:
        return NI
    ns, one_d = sos.shape[0], x.ndim == 1
    try:
        with np.errstate(**_RAISE):
            y = _odd_ext(x, edge)
            z = zi * y[:1] if one_d else zi[None] * y[:, :1, None]
            if y.dtype != F8:
                y = y.astype(F8)
            _sosfilt(sos, y.reshape(1, -1) if one_d else y, z.reshape(-1, ns, 2))
            z = zi * y[..., -1:] if one_d else zi[None] * y[:, -1:, None]
            yr = y[..., ::-1].copy()
            _sosfilt(sos, yr.reshape(1, -1) if one_d else yr, z.reshape(-1, ns, 2))
    except FloatingPointError:
        return NI
    return yr[..., ::-1][..., edge:-edge]


def lfilter(b, a, x, axis=-1, zi=None):
    if not (_f8(b) and _f8(a) and _f8(x) and b.ndim == 1 and a.ndim == 1 and b.size and a.size > 1):
        return NI
    if zi is not None and not _f8(zi):
        return NI
    try:
        return _linear_filter(b, a, x, axis, *(() if zi is None else (zi,)))
    except Exception:  # bad axis, shapes or coefficients: SciPy raises its own error
        return NI


def filtfilt(b, a, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None):
    if not (
        _f8(b)
        and _f8(a)
        and type(x) is ndarray
        and x.dtype in (F8, F4)
        and _last_axis(x, axis)
        and padlen is None
        and _is(padtype, "odd")
        and _is(method, "pad")
        and b.ndim == 1
        and a.ndim == 1
        and b.size
        and a.size > 1
    ):
        return NI
    edge = 3 * max(a.size, b.size)
    if x.shape[-1] <= edge or not x.size:
        return NI
    zi = _lf_zi(b, a)
    if zi is None:
        return NI
    try:
        with np.errstate(**_RAISE):
            ext = _odd_ext(x, edge)
            y, _ = _linear_filter(b, a, ext, axis, zi * ext[..., :1])
            y, _ = _linear_filter(b, a, y[..., ::-1], axis, zi * y[..., -1:])
    except FloatingPointError:
        return NI
    return y[..., ::-1][..., edge:-edge]


# ------------------------------------------------------------------- caches
def lfilter_zi(b, a):
    if not (_f8(b) and _f8(a)):
        return NI
    z = _lf_zi(b, a)
    if z is None:
        return NI
    return z[::-1].copy()[::-1] if z.strides[0] < 0 else z.copy()  # SciPy returns a reversed view


def sosfilt_zi(sos):
    if not (_f8(sos) and sos.ndim == 2 and sos.shape[1] == 6):
        return NI
    zi = _sos_info(sos)[2]
    return NI if zi is None else zi.copy()


def _key(v):
    t = type(v)
    if v is None or t in (int, float, str, bool):
        return t, v
    if t is ndarray and v.dtype == F8 and v.ndim <= 1:
        return t, v.shape, v.tobytes()
    if t is np.float64:
        return t, float(v)
    if t in (list, tuple) and len(v) <= 4 and all(type(i) in (int, float, np.float64) for i in v):
        return t, tuple((type(i), float(i)) for i in v)
    raise _Uncacheable


def _copy(o):
    if isinstance(o, ndarray):
        return o.copy(order="K")
    if isinstance(o, tuple):
        return tuple(map(_copy, o))
    if type(o) in (float, np.float64, int):
        return o
    raise _Uncacheable


def iirfilter(N, Wn, rp=None, rs=None, btype="band", analog=False, ftype="butter", output="ba", fs=None):
    args = N, Wn, rp, rs, btype, analog, ftype, output, fs
    try:
        key = tuple(map(_key, args))
    except _Uncacheable:
        return NI
    r = _iir_cache.get(key)
    if r is not None:
        return _copy(r)
    try:
        ok, r = _quiet(lambda: orig["iirfilter"](*args))
        if ok:
            _iir_cache.put(key, _copy(r))
            return r
    except Exception:
        pass
    return NI  # an error or a warning: SciPy re-runs it and raises or warns itself


# ---------------------------------------------------------- detrend / hilbert
def detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):
    if not (
        _f8(data)
        and data.size
        and data.ndim
        and (_is(type, "linear") or _is(type, "l"))
        and bp.__class__ is int
        and bp == 0
        and axis.__class__ is int
        and -data.ndim <= axis < data.ndim
    ):
        return NI
    N = data.shape[axis]
    if N < 2:
        return NI
    A = _A_cache.get(N)
    if A is None:
        A = np.ones((N, 2))
        A[:, 0] = np.arange(1, N + 1, dtype="d") / N
        A = _A_cache.put(N, _ro(A))
    axis %= data.ndim
    moved = np.moveaxis(data, axis, 0)
    shape = moved.shape
    b = moved.reshape(N, -1)
    if not overwrite_data:
        b = b.copy()
    if not np.abs(b).max() < 1e100:  # NaN/inf: lstsq raises; huge: its residual check can warn
        return NI
    x, _, _, err = _batched_linalg._lstsq(A, b, 2.220446049250313e-16, "gelsd", False, False)
    if err:
        return NI
    b[...] = b - A @ x[..., :2, :]
    return np.moveaxis(b.reshape(shape), 0, axis)


def hilbert(x, N=None, axis=-1):
    if not (_f8(x) and x.ndim and N is None and axis.__class__ is int and -x.ndim <= axis < x.ndim and x.size):
        return NI
    N = x.shape[axis]
    Xf = np.moveaxis(scipy.fft.fft(x, N, axis=axis), axis, -1)
    h = (N + 1) // 2
    try:
        with np.errstate(**_RAISE):
            Xf[..., 1:h] *= 2.0
            Xf[..., h + (N % 2 == 0) :] *= 0.0  # SciPy multiplies by h, so signed zeros and NaN match
    except FloatingPointError:
        return NI
    return scipy.fft.ifft(np.moveaxis(Xf, -1, axis), axis=axis)


# ---------------------------------------------------------------- welch / csd
def _window_key(w):
    if type(w) is str:
        return w
    if type(w) is tuple and all(type(i) in (str, int, float) for i in w):
        return tuple((type(i), i) for i in w)
    if _f8(w) and w.ndim == 1:
        return w.shape, w.tobytes()
    raise _Uncacheable


def _sft(window, nperseg, hop, fs, nfft, scaling):
    """(ShortTimeFFT, conjugated scaled window, read-only f), built by SciPy's own code."""
    key = _window_key(window), nperseg, hop, _key(fs), nfft, scaling
    e = _sft_cache.get(key)
    if e is None:
        win = get_window(window, nperseg) if isinstance(window, (str, tuple)) else np.asarray(window)
        scale = {"spectrum": "magnitude", "density": "psd"}[scaling]
        ok, sft = _quiet(
            lambda: ShortTimeFFT(win, hop, fs, fft_mode="onesided", mfft=nfft, scale_to=scale, phase_shift=None)
        )
        if not ok or sft.win.dtype != F8:
            return None
        e = _sft_cache.put(key, (sft, sft.win.conj(), _ro(sft.f.copy()), _Cache()))
    return e


def _frames(e, x, nperseg, p1, detr):
    """``ShortTimeFFT.stft_detrend(x, detr, 0, p1, k_offset=nperseg // 2)`` for C-contiguous
    float64 x. One FFT per frame: a batched rfft is not bit-identical."""
    sft, wc, f, p_ranges = e
    n, m, hop = x.shape[-1], sft.m_num, sft.hop
    pr = p_ranges.get((n, p1))
    if pr is None:
        pr = p_ranges.put((n, p1), sft.p_range(n, 0, p1))
    p0, p1 = pr
    n1 = (p1 - p0) * hop
    k0 = p0 * hop - sft.m_num_mid + nperseg // 2
    k1 = k0 + n1 + m
    lo, hi = -min(k0, 0), max(k1 - n, 0)
    if lo or hi:
        x1 = np.zeros(x.shape[:-1] + (lo + min(k1, n) - max(k0, 0) + hi,))
        x1[..., lo : x1.shape[-1] - hi] = x[..., max(k0, 0) : min(k1, n)]
    else:
        x1 = x[..., k0:k1]
    S = np.zeros(x.shape[:-1] + (len(f), p1 - p0), dtype=complex)
    for p, k in enumerate(range(0, n1, hop)):
        xs = x1[..., k : k + m]
        if detr:
            r = np.add.reduce(xs, axis=-1, keepdims=True)
            xs = xs - np.true_divide(r, m, out=r)  # what np.mean(..., keepdims=True) does
        S[..., :, p] = scipy.fft.rfft(xs * wc, n=sft.mfft, axis=-1)
    return S


def csd(
    x,
    y,
    fs=1.0,
    window="hann_periodic",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
):
    return _csd(x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, average)


def welch(
    x,
    fs=1.0,
    window="hann_periodic",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
):
    return _csd(x, x, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, average)


def _csd(x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, average):
    same = y is x
    if not (
        _f8(x)
        and (same or (_f8(y) and y.shape == x.shape))
        and x.size
        and _last_axis(x, axis)
        and return_onesided is True
        and _is(average, "mean")
        and (_is(scaling, "density") or _is(scaling, "spectrum"))
        and (detrend is False or _is(detrend, "constant"))
        and type(window) in (str, tuple, ndarray)
    ):
        return NI
    n = x.shape[-1]
    if nperseg is None:
        nperseg = len(window) if type(window) is ndarray else 256
    nfft = nperseg if nfft is None else nfft
    noverlap = nperseg // 2 if noverlap is None else noverlap
    if not (
        type(nperseg) is int
        and 1 <= nperseg <= n
        and type(nfft) is int
        and nfft >= nperseg
        and type(noverlap) is int
        and 0 <= noverlap < nperseg
        and (type(window) is not ndarray or len(window) == nperseg)
    ):
        return NI
    try:
        e = _sft(window, nperseg, nperseg - noverlap, fs, nfft, scaling)
    except _Uncacheable:
        return NI
    if e is None:
        return NI
    sft, p1 = e[0], (n - noverlap) // (nperseg - noverlap)
    # SciPy's np.pad copies the data into a fresh C-contiguous array
    xc = np.ascontiguousarray(x)
    yc = xc if same else np.ascontiguousarray(y)
    detr = detrend is not False
    try:
        with np.errstate(**_RAISE):
            Sy = _frames(e, yc, nperseg, p1, detr)
            if same:
                P = Sy.real**2 + Sy.imag**2
            else:
                P = Sy * _frames(e, xc, nperseg, p1, detr).conj()
            P = np.moveaxis(P, P.ndim - 2, -1)
            P[..., 1 : -1 if sft.mfft % 2 == 0 else None] *= 2
            P = np.moveaxis(P, -1, P.ndim - 2)
            P = P.mean(axis=-1) if P.shape[-1] > 1 else P.reshape(P.shape[:-1])
    except FloatingPointError:
        return NI
    P = P.astype(np.complex128)
    return e[2].copy(), P.real if same else P
