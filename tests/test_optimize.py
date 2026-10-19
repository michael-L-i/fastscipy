import numpy as np
import pytest
from scipy import optimize

import fastscipy
from fastscipy.testing import check

rng = np.random.default_rng(0)


@pytest.mark.parametrize("n", [1, 5, 300])
def test_approx_fprime(n):
    x = rng.standard_normal(n)
    check(lambda: optimize.approx_fprime(x, optimize.rosen, 1e-8))
    check(lambda: optimize.approx_fprime(x, lambda z, a: a * np.sum(np.sin(z)), np.full(n, 1e-7), 3.0))
    check(lambda: optimize.approx_fprime(x, lambda z: np.array([z.sum(), (z**2).sum()]), 1e-8))
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
        z[0] += 1.0  # the gradient must not depend on the callback's mutation of its argument
        return float(z.sum())

    check(lambda: optimize.approx_fprime(x.copy(), f, 1e-8))
    with pytest.raises(RuntimeError, match="more than 1 dimension"):
        optimize.approx_fprime(x, lambda z: np.ones((2, 2)) if z[0] != 1 else 1.0, 1e-8)


@pytest.fixture(params=[1, 2])
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield
    fastscipy.set_num_threads(old)


@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (3, 8), (12, 4), (40, 10)])
def test_lsq_linear_dense_unbounded(shape, threads):
    local = np.random.default_rng(812 + shape[0])
    for _ in range(8):
        A, b = local.standard_normal(shape), local.standard_normal(shape[0])
        for matrix in (A, np.asfortranarray(A), A[::-1]):
            rhs = b if matrix is A else b[::-1]
            before = matrix.tobytes(), rhs.tobytes()
            for method in ("trf", "bvls"):
                check(lambda: optimize.lsq_linear(matrix, rhs, method=method))
            assert (matrix.tobytes(), rhs.tobytes()) == before
    A = np.ones(shape)
    check(lambda: optimize.lsq_linear(A, np.ones(shape[0]), lsq_solver="exact"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": (0, np.inf)},
        {"bounds": (0, 0)},
        {"bounds": (0,)},
        {"bounds": (np.zeros(2), np.ones(2))},
        {"method": "bad"},
        {"lsq_solver": "lsmr"},
        {"lsq_solver": "bad"},
        {"lsmr_tol": 0.0},
        {"max_iter": 0},
        {"max_iter": 1},
        {"verbose": 3},
        {"lsmr_maxiter": 0},
    ],
)
def test_lsq_linear_options_decline(kwargs):
    from fastscipy._optimize import _lsq_linear_fast

    A, b = np.eye(3), np.ones(3)
    assert _lsq_linear_fast(A, b, **kwargs) is NotImplemented
    check(lambda: optimize.lsq_linear(A, b, **kwargs), fast=False)


@pytest.mark.parametrize(
    "A,b",
    [
        (np.eye(3, dtype=np.float32), np.ones(3)),
        (np.eye(3), np.ones(3, dtype=np.float32)),
        (np.ones((2, 3, 4)), np.ones(2)),
        (np.empty((3, 0)), np.ones(3)),
        (np.eye(3), np.ones((3, 1))),
        (np.eye(3), np.ones(2)),
        ([[1.0, 0.0], [0.0, 1.0]], np.ones(2)),
    ],
)
def test_lsq_linear_inputs_decline(A, b):
    from fastscipy._optimize import _lsq_linear_fast

    assert _lsq_linear_fast(A, b) is NotImplemented
    check(lambda: optimize.lsq_linear(A, b), fast=False)


@pytest.mark.parametrize("n", [2, 3, 17, 64, 257])
def test_rosen_hess_diagonals(n, threads):
    local = np.random.default_rng(813 + n)
    for _ in range(15):
        x = local.standard_normal(n * 2)
        x[::7] = -0.0
        for point in (x[:n], x[::2], x[:n][::-1], np.zeros(n)):
            before = point.tobytes()
            check(lambda: optimize.rosen_hess(point))
            assert point.tobytes() == before
    point = local.standard_normal(n)
    point.flags.writeable = False
    check(lambda: optimize.rosen_hess(point))


@pytest.mark.parametrize(
    "x",
    [
        np.empty(0),
        np.ones(1),
        np.ones((2, 3)),
        np.ones(3, dtype=np.float32),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1e200, 1.0]),
    ],
)
def test_rosen_hess_declines(x):
    from fastscipy._optimize import _rosen_hess_fast

    assert _rosen_hess_fast(x) is NotImplemented
    with np.errstate(all="ignore"):
        check(lambda: optimize.rosen_hess(x), fast=False)


@pytest.mark.parametrize("method", ["Newton-CG", "trust-exact"])
def test_minimize_with_accelerated_hessian(method, threads):
    x = np.array([-1.2, 1.0, 0.8])

    def run():
        history = []
        result = optimize.minimize(
            optimize.rosen,
            x,
            method=method,
            jac=optimize.rosen_der,
            hess=optimize.rosen_hess,
            callback=lambda point: history.append(point.copy()),
        )
        return result, history

    check(run)
