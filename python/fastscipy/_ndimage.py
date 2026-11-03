"""scipy.ndimage: Rust kernels behind SciPy's own Python wrappers.

The wrappers keep their identity, argument checking and output types; only the
calls into the private C module `_nd_image` are routed through one composed
stand-in (see `_ndimage_seam`).
"""

import numpy as np
import scipy
import scipy.ndimage as ndi
from scipy.ndimage import _filters, _interpolation, _measurements, _morphology

from . import _patch
from . import _ndimage_label, _ndimage_morph
from . import _ndimage_correlate, _ndimage_generic, _ndimage_interp, _ndimage_rank  # noqa: F401  (register overrides)
from ._ndimage_seam import standin

_WRAPPERS = (_filters, _interpolation, _measurements, _morphology)


def patch():
    _patch.require_validated(*(f"1.{minor}." for minor in range(10, 19)))
    # Interpolation and EDT kernels reproduce the current SciPy arithmetic, including
    # its compiler contraction. Older wheels differ even on integer outputs.
    interpolation = (
        ()
        if scipy.__version__.startswith("1.18.")
        else ("spline_filter1d", "zoom_shift", "geometric_transform", "euclidean_feature_transform")
    )
    mod = standin(exclude=interpolation)
    for wrapper in _WRAPPERS:
        _patch.replace(wrapper, "_nd_image", mod)
    _ndimage_morph.patch(edt=not interpolation)
    _ndimage_label.patch()


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
    b = x > 0.5
    lab, n = ndi.label(b)
    idx = np.arange(1, n + 1)
    ndi.binary_erosion(b)
    ndi.binary_dilation(b, iterations=2)
    ndi.binary_fill_holes(b)
    ndi.binary_propagation(b & (x > 0.9), mask=b)
    ndi.distance_transform_edt(b)
    ndi.generic_filter(x, np.mean, size=3)
    ndi.find_objects(lab)
    ndi.minimum(x, lab, idx)
    ndi.maximum_position(x, lab, idx)
    ndi.center_of_mass(x, lab, idx)
    ndi.center_of_mass((x * 100).astype(np.uint16) + 1, lab, idx)
    ndi.watershed_ift((x * 255).astype(np.uint8), lab.astype(np.int32))
