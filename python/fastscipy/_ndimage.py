"""scipy.ndimage: Rust kernels behind SciPy's own Python wrappers.

The wrappers keep their identity, argument checking and output types; only the
calls into the private C module `_nd_image` are routed through one composed
stand-in (see `_ndimage_seam`).
"""
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import _filters, _interpolation

from . import _patch
from . import _ndimage_correlate, _ndimage_interp, _ndimage_rank  # noqa: F401  (register overrides)
from ._ndimage_seam import standin

_WRAPPERS = (_filters, _interpolation)


def patch():
    _patch.require_scipy("1.18.")
    mod = standin()
    for wrapper in _WRAPPERS:
        _patch.replace(wrapper, "_nd_image", mod)


def smoke():
    x = np.random.default_rng(0).random((24, 24))
    ndi.gaussian_filter(x, 1.0)
    ndi.uniform_filter(x, 3)
    ndi.convolve(x, np.ones((3, 3)))
    ndi.median_filter(x, 3)
    ndi.maximum_filter(x, 3)
    ndi.zoom(x, 1.5)
    ndi.rotate(x, 10)
    ndi.map_coordinates(x, np.mgrid[:5, :5] + 0.5)
