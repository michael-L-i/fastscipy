"""``scipy.signal``: small-call fast paths, multichannel threading and Rust kernels.

* ``_signal_small``: same kernels and arithmetic as SciPy minus the Python layers,
  plus caches of filter designs and initial conditions (identical by construction).
* ``_signal_threads``: independent channels on several threads (identical by construction).
"""
import importlib

import numpy as np

from . import _patch

_FUNCS = {"_signaltools": ("sosfilt", "sosfiltfilt", "lfilter", "filtfilt", "lfilter_zi", "sosfilt_zi",
                           "detrend", "hilbert", "decimate", "resample_poly", "medfilt2d"),
          "_filter_design": ("iirfilter",),
          "_spectral_py": ("welch", "csd"),
          "_upfirdn": ("upfirdn",)}


def _first(*paths):
    """One fast path that tries each of ``paths`` (same signature) until one handles the call."""
    paths = [p for p in paths if p]

    def fast(*args, **kwargs):
        for path in paths:
            result = path(*args, **kwargs)
            if result is not NotImplemented:
                return result
        return NotImplemented
    return fast


def patch():
    _patch.require_scipy("1.18.")
    from . import _signal_small as small
    from . import _signal_threads as threads
    small.orig.clear()
    for mod, names in _FUNCS.items():
        owner = importlib.import_module(f"scipy.signal.{mod}")
        for name in names:
            old = small.orig[name] = getattr(owner, name)
            fast = _first(threads.make(name, old), getattr(small, name, None))
            _patch.rebind(old, _patch.wrap(old, fast))


def smoke():
    import scipy.signal as sg
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 300))
    big = rng.standard_normal((8, 1 << 14))
    sos = sg.butter(3, 0.2, output="sos")
    b, a = sg.butter(3, 0.2)
    for data in (x, big):
        sg.sosfilt(sos, data)
        sg.sosfiltfilt(sos, data)
        sg.lfilter(b, a, data)
        sg.filtfilt(b, a, data)
    sg.detrend(x)
    sg.hilbert(x)
    sg.welch(x, nperseg=64)
    sg.csd(x, x[::-1].copy(), nperseg=64)
    sg.decimate(big, 4)
    sg.resample_poly(big, 3, 2, axis=-1)
    sg.upfirdn(np.ones(16), big, 2)
    sg.medfilt2d(rng.random((512, 512)), 5)
