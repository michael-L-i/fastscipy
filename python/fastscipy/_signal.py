"""``scipy.signal``: small-call fast paths, multichannel threading and Rust kernels.

* ``_signal_small``: same kernels and arithmetic as SciPy minus the Python layers,
  plus caches of filter designs and initial conditions (identical by construction).
"""
import numpy as np

from . import _patch

_SMALL = {"_signaltools": ("sosfilt", "sosfiltfilt", "lfilter", "filtfilt", "lfilter_zi", "sosfilt_zi",
                           "detrend", "hilbert"),
          "_filter_design": ("iirfilter",),
          "_spectral_py": ("welch", "csd")}


def patch():
    _patch.require_scipy("1.18.")
    import importlib

    from . import _signal_small as small
    small.orig.clear()
    for mod, names in _SMALL.items():
        owner = importlib.import_module(f"scipy.signal.{mod}")
        for name in names:
            old = small.orig[name] = getattr(owner, name)
            _patch.rebind(old, _patch.wrap(old, getattr(small, name)))


def smoke():
    import scipy.signal as sg
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 300))
    sos = sg.butter(3, 0.2, output="sos")
    b, a = sg.butter(3, 0.2)
    sg.sosfilt(sos, x)
    sg.sosfiltfilt(sos, x)
    sg.lfilter(b, a, x)
    sg.filtfilt(b, a, x)
    sg.detrend(x)
    sg.hilbert(x)
    sg.welch(x, nperseg=64)
    sg.csd(x, x[::-1].copy(), nperseg=64)
