"""Linear extrema reductions while SciPy owns binning and result metadata."""

import functools

import numpy as np
from scipy.stats import _binned_statistic as sp

from . import _patch


def _minimum(values):
    return np.min(values)


def _maximum(values):
    return np.max(values)


def _reduce(vdim, bin_numbers, result, values, statistic):
    if statistic is _minimum:
        operation = np.fmin
    elif statistic is _maximum:
        operation = np.fmax
    else:
        return NotImplemented
    # SciPy has initialized empty bins to NaN. fmin/fmax replace those on the
    # first finite observation, retaining NaN only in unoccupied bins.
    for row in range(vdim):
        operation.at(result[row], bin_numbers, values[row])


def _dispatch(original):
    @functools.wraps(original)
    def wrapped(
        sample, values, statistic="mean", bins=10, range=None, expand_binnumbers=False, binned_statistic_result=None
    ):
        marker = None
        if statistic is np.min or (type(statistic) is str and statistic == "min"):
            marker = _minimum
        elif statistic is np.max or (type(statistic) is str and statistic == "max"):
            marker = _maximum
        if (
            _patch.enabled
            and marker is not None
            and binned_statistic_result is None
            and type(values) is np.ndarray
            and values.dtype == np.dtype("float64")
            and values.ndim in (1, 2)
            and values.size >= 4096
            and np.isfinite(values).all()
            and (values != 0).all()
        ):
            # Zero signs depend on argsort's tie order, so leave zeros to SciPy.
            # This marker reaches only the existing callable reduction seam;
            # no user callback is evaluated speculatively or repeated.
            statistic = marker
        return original(sample, values, statistic, bins, range, expand_binnumbers, binned_statistic_result)

    return wrapped


def patch():
    _patch.replace(
        sp, "_calc_binned_statistic", _patch.wrap(sp._calc_binned_statistic, _reduce, "binned_statistic.extrema")
    )
    original = sp.binned_statistic_dd
    replacement = _dispatch(original)
    _patch.replace(sp, "binned_statistic_dd", replacement)
    _patch.rebind(original, replacement)
