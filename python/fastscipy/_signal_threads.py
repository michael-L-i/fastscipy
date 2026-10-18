"""Multichannel filters on several threads.

A 2-D+ input is split along its largest non-filter dimension and each slab goes
through SciPy's own function, whose compiled kernel releases the GIL. Every
output row depends on one input row only, so the result is identical by
construction. The slabs are copied into an array laid out like SciPy's result.

Not threaded: ``filtfilt(method='gust')`` and ``resample_poly(padtype='mean'|'line')``
(their result depends on the slab shape); ``welch``, ``csd``, ``stft`` (many small
NumPy calls: 2.5x slower on a GIL build).
"""
import inspect

import numpy as np
from numpy.lib.stride_tricks import as_strided

from . import _signal_small, _threads

NI = NotImplemented
# smallest input and estimated multiplications worth threads (about 1 ms of SciPy time); tests lower them
MIN_SIZE, MIN_WORK, MIN_IMAGE = 4096, 1 << 20, 1 << 16
_size = np.size


def _slab(a, d, s):
    return a[(slice(None),) * d + (s,)]


def _like(r, d, parts):
    """Join ``parts`` along ``d`` into an array with the strides of ``r`` (SciPy returns
    reversed or sliced views of padded arrays, not always C-ordered ones). Strides of
    dimensions outside ``d`` contain its extent, so they scale with it."""
    n, full = r.shape[d], sum(p.shape[d] for p in parts)
    shape = r.shape[:d] + (full,) + r.shape[d + 1:]
    strides = [s if abs(s) <= abs(r.strides[d]) else s // n * full for s in r.strides]
    if any(s % r.itemsize for s in strides):
        raise ValueError("unexpected result layout")
    reach = sum((m - 1) * abs(s) // r.itemsize for m, s in zip(shape, strides))
    back = sum((m - 1) * -s // r.itemsize for m, s in zip(shape, strides) if s < 0)
    out = as_strided(np.empty(reach + 1, r.dtype)[back:], shape, strides)
    start = 0
    for p in parts:
        _slab(out, d, slice(start, start + p.shape[d]))[...] = p
        start += p.shape[d]
    return out


def _join(results, axes):
    def cat(parts, axis):
        return _like(max(parts, key=lambda p: p.shape[axis]), axis, parts)
    if isinstance(results[0], tuple):
        return tuple(cat([r[j] for r in results], axes[j]) for j in range(len(results[0])))
    return cat(results, axes[0])


def _threaded(orig, xname, axis_default, work, zi_off=None, veto=None):
    """Fast path for ``orig``: ``work(bound)`` estimates multiplications per sample."""
    params = list(inspect.signature(orig).parameters)
    xpos = params.index(xname)

    def fast(*args, **kwargs):
        x = args[xpos] if len(args) > xpos else kwargs.get(xname)
        if not (type(x) is np.ndarray and x.ndim >= 2 and x.size >= MIN_SIZE and x.dtype.kind in "iufc"
                and 1 not in x.shape and len(args) <= len(params)):
            return NI
        threads = _threads.get_num_threads()
        if threads < 2 or _threads._busy.locked():
            return NI
        bound = dict(zip(params, args), **kwargs)
        axis = bound.get("axis", axis_default)
        if len(bound) != len(args) + len(kwargs) or axis.__class__ is not int or not -x.ndim <= axis < x.ndim or (veto and veto(bound)):
            return NI
        if x.size * work(bound) < MIN_WORK:
            return NI
        axis %= x.ndim
        d = max((i for i in range(x.ndim) if i != axis), key=x.shape.__getitem__)
        zi = bound.get("zi")
        if zi is not None and not (type(zi) is np.ndarray and zi.ndim == x.ndim + zi_off
                                   and zi.shape[d + zi_off] == x.shape[d]):
            return NI
        slices = _threads.chunks(x.shape[d], 2, threads)

        def call(s):
            b = dict(bound, **{xname: _slab(x, d, s)})
            if zi is not None:
                b["zi"] = _slab(zi, d + zi_off, s)
            return orig(**b)
        try:
            results = _threads.parallel_map(call, slices)
        except Exception:
            return NI    # SciPy raises its own error on the whole input
        return _join(results, (d, d + (zi_off or 0)))
    return fast


def _medfilt2d(orig):
    """Row bands with a half-kernel halo of real rows; zeros only at the true image edges."""
    def fast(input, kernel_size=3):
        x = input
        if not (type(x) is np.ndarray and x.ndim == 2 and x.dtype in (np.uint8, np.float32, np.float64)
                and x.size >= MIN_IMAGE) or _threads.get_num_threads() < 2 or _threads._busy.locked():
            return NI
        ks = np.asarray(kernel_size)
        if ks.dtype.kind != "i" or ks.ndim > 1 or ks.size not in (1, 2) or (ks % 2 == 0).any() or (ks < 1).any():
            return NI
        h = int(ks.flat[0]) // 2
        n = x.shape[0]
        bounds = [s.start for s in _threads.chunks(n, 8)] + [n]
        taps = int(ks.prod()) ** (2 // ks.size)     # a scalar size means a square kernel
        if x.size * taps < MIN_WORK or len(bounds) < 3:
            return NI

        def band(i):
            r0, r1 = bounds[i], bounds[i + 1]
            a0, a1 = max(r0 - h, 0), min(r1 + h, n)
            return orig(np.ascontiguousarray(x[a0:a1]), kernel_size)[r0 - a0:r1 - a0]
        try:
            bands = _threads.parallel_map(band, range(len(bounds) - 1))
        except Exception:
            return NI
        return np.concatenate(bands, axis=0)
    return fast


def _coefs(*names):
    return lambda b: max(_size(b[n]) for n in names)


def make(name, orig):
    """The threaded fast path for ``name`` (a ``scipy.signal`` function), or None."""
    if name == "medfilt2d":
        return _medfilt2d(orig)
    spec = {
        "lfilter": ("x", -1, _coefs("b", "a"), 0, None),
        "sosfilt": ("x", -1, lambda b: _size(b["sos"]) // 6, 1, None),
        "filtfilt": ("x", -1, _coefs("b", "a"), None, lambda b: b.get("method", "pad") != "pad"),
        "sosfiltfilt": ("x", -1, lambda b: _size(b["sos"]) // 6, None, None),
        "decimate": ("x", -1, lambda b: 16, None, None),
        "resample_poly": ("x", 0, lambda b: 64, None, lambda b: b.get("padtype", "constant") in ("mean", "line")),
        "upfirdn": ("x", -1, _coefs("h"), None, None),
    }.get(name)
    return _threaded(orig, spec[0], spec[1], spec[2], spec[3], spec[4]) if spec else None
