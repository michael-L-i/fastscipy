import numpy as np
import pytest
from scipy import optimize

from fastscipy.testing import check

rng = np.random.default_rng(0)


@pytest.mark.parametrize("n", [1, 5, 300])
def test_approx_fprime(n):
    x = rng.standard_normal(n)
    check(lambda: optimize.approx_fprime(x, optimize.rosen, 1e-8))
    check(lambda: optimize.approx_fprime(x, lambda z, a: a * np.sum(np.sin(z)), np.full(n, 1e-7), 3.0))
    check(lambda: optimize.approx_fprime(x, lambda z: np.array([z.sum(), (z ** 2).sum()]), 1e-8))
    check(lambda: optimize.approx_fprime(x, lambda z: float(z[0]), 1e-8))


def test_other_numdiff_users():
    x = rng.standard_normal(6)
    from scipy.optimize._numdiff import approx_derivative
    for method in ("2-point", "3-point", "cs"):
        check(lambda: approx_derivative(optimize.rosen, x, method=method, bounds=(-3, 3)), fast=method != "cs")
    check(lambda: approx_derivative(lambda z: np.array([z.sum(), z[0] * z[1]]), x, method="3-point"))
    check(lambda: optimize.minimize(optimize.rosen, x, method="BFGS").x)
    check(lambda: optimize.least_squares(lambda z: z - 1.5, x).x)


def test_callback_mutating_argument_and_errors():
    x = np.ones(4)

    def f(z):
        z[0] += 1.0     # the gradient must not depend on the callback's mutation of its argument
        return float(z.sum())

    check(lambda: optimize.approx_fprime(x.copy(), f, 1e-8))
    with pytest.raises(RuntimeError, match="more than 1 dimension"):
        optimize.approx_fprime(x, lambda z: np.ones((2, 2)) if z[0] != 1 else 1.0, 1e-8)
