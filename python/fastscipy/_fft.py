"""Skip redundant Python preparation for unresized NumPy ND transforms.

The same SciPy DUCC kernels do the arithmetic. Only the SciPy backend is
wrapped, leaving public uarray dispatch and its alternate backends intact.
SciPy resolves workers itself; this component creates no pools or workers.
"""

import builtins
import functools

import numpy as np

from . import _patch


def _wrap(original, fast, name, kinds):
    """Reject conversion-heavy inputs before entering the ND preparation path."""

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if _patch.enabled:
            x = args[0] if args else kwargs.get("x")
            if type(x) is np.ndarray and x.dtype.char in kinds:
                try:
                    result = fast(*args, **kwargs)
                except Exception:
                    if _patch.STRICT:
                        raise
                    _patch.counts[name, "error"] += 1
                else:
                    if result is not NotImplemented:
                        _patch.counts[name, "fast"] += 1
                        return result
        _patch.counts[name, "scipy"] += 1
        return original(*args, **kwargs)

    wrapper.fast = fast
    return wrapper


def _axes(x, s, axes, norm, overwrite_x, workers):
    if (
        type(x) is not np.ndarray
        or s is not None
        or overwrite_x is not False
        or x.dtype.char not in "fdFD"
        or not x.dtype.isnative
        or not x.flags.aligned
        or not x.ndim
        or (norm is not None and (type(norm) is not str or norm not in ("backward", "ortho", "forward")))
        or (workers is not None and (type(workers) is not int or workers <= 0))
    ):
        return None
    if axes is None:
        if not all(x.shape):
            return None
        return tuple(range(x.ndim))
    if type(axes) is not tuple or not axes:
        return None
    if any(type(a) is not int or not -x.ndim <= a < x.ndim for a in axes):
        return None
    normalized = tuple(a % x.ndim for a in axes)
    if len(set(normalized)) != len(normalized) or any(x.shape[a] == 0 for a in normalized):
        return None
    return normalized


def _complex_nd(forward):
    from scipy.fft._duccfft import helper, pyduccfft

    def fast(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None):
        if plan is not None:
            return NotImplemented
        selected = _axes(x, s, axes, norm, overwrite_x, workers)
        if selected is None:
            return NotImplemented
        normalization = helper._normalization(norm, forward)
        return pyduccfft.c2c(x, selected, forward, normalization, None, helper._workers(workers))

    return fast


def _real_nd(forward, sine=False, legacy=False):
    from scipy.fft._duccfft import helper, pyduccfft

    transform = pyduccfft.dst if sine else pyduccfft.dct

    def fast(x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, orthogonalize=None):
        selected = _axes(x, s, axes, norm, overwrite_x, workers)
        if selected is None or x.dtype.kind != "f":
            return NotImplemented
        if builtins.type(type) is not int or type not in (1, 2, 3, 4):
            return NotImplemented
        if orthogonalize is not None and builtins.type(orthogonalize) is not bool:
            return NotImplemented
        if type == 1 and not sine and any(x.shape[a] == 1 for a in selected):
            return NotImplemented
        if not forward and type in (2, 3):
            type = 5 - type
        # fftpack's inverse changes the transform type but keeps forward
        # normalization; scipy.fft's inverse also reverses normalization.
        normalization = helper._normalization(norm, forward or legacy)
        return transform(x, type, selected, normalization, None, helper._workers(workers), orthogonalize)

    return fast


def _legacy_complex_nd(forward):
    candidate = _complex_nd(forward)

    def fast(x, shape=None, axes=None, overwrite_x=False):
        return candidate(x, shape, axes, overwrite_x=overwrite_x)

    return fast


def _legacy_real_nd(forward, sine):
    candidate = _real_nd(forward, sine, legacy=True)

    def fast(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
        return candidate(x, type, shape, axes, norm, overwrite_x)

    return fast


def patch():
    _patch.require_validated("1.18.")
    from scipy.fft import _basic_backend, _realtransforms_backend
    from scipy.fftpack import _basic, _realtransforms

    for name, forward in (("fftn", True), ("ifftn", False)):
        original = getattr(_basic_backend, name)
        _patch.replace(_basic_backend, name, _wrap(original, _complex_nd(forward), f"fft.{name}", "fdFD"))
        original = getattr(_basic, name)
        wrapped = _wrap(original, _legacy_complex_nd(forward), f"fftpack.{name}", "fdFD")
        _patch.replace(_basic, name, wrapped)
        _patch.rebind(original, wrapped)
    for name, forward, sine in (
        ("dctn", True, False),
        ("idctn", False, False),
        ("dstn", True, True),
        ("idstn", False, True),
    ):
        original = getattr(_realtransforms_backend, name)
        _patch.replace(_realtransforms_backend, name, _wrap(original, _real_nd(forward, sine), f"fft.{name}", "fd"))
        original = getattr(_realtransforms, name)
        wrapped = _wrap(original, _legacy_real_nd(forward, sine), f"fftpack.{name}", "fd")
        _patch.replace(_realtransforms, name, wrapped)
        _patch.rebind(original, wrapped)


def smoke():
    from scipy import fft, fftpack

    x = np.arange(64.0).reshape(8, 8)
    for name in ("fftn", "ifftn", "dctn", "idctn", "dstn", "idstn"):
        getattr(fft, name)(x)
        getattr(fftpack, name)(x)
