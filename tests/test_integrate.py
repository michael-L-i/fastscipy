import numpy as np
import pytest
from scipy import integrate

from fastscipy.testing import check

rng = np.random.default_rng(1)


@pytest.mark.parametrize("n", [2, 3, 100, 5000])
def test_trapezoid(n):
    y = rng.standard_normal(n)
    x = np.sort(rng.random(n))
    check(lambda: integrate.trapezoid(y))
    check(lambda: integrate.trapezoid(y, x))
    check(lambda: integrate.trapezoid(y, dx=0.25))
    check(lambda: integrate.trapezoid(y, dx=2))
    y2 = rng.standard_normal((n, 4, 3))
    for axis in (0, 1, 2, -1, -3):
        check(lambda: integrate.trapezoid(y2, axis=axis))
        if axis in (0, -3):
            check(lambda: integrate.trapezoid(y2, x, axis=axis))


def test_trapezoid_declined():
    y = rng.standard_normal(20)
    for args, kw in (
        ((y.astype(np.float32),), {}),
        ((list(y),), {}),
        ((y[:0],), {}),
        ((y,), {"x": y.astype(int)}),
        ((y,), {"dx": np.float64(0.5)}),
    ):
        check(lambda: integrate.trapezoid(*args, **kw), fast=False)
    check(lambda: integrate.trapezoid(y, axis=np.int64(0)), fast=False)
    check(lambda: integrate.trapezoid(np.ones(1)))
    with pytest.raises(Exception) as e1:
        integrate.trapezoid(y, axis=3)
    assert e1.type.__name__ in ("AxisError", "IndexError")
