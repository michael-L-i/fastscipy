import warnings

import numpy as np
import pytest
from scipy import optimize

import fastscipy
from fastscipy.testing import assert_identical, check

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
                check(
                    lambda: optimize.lsq_linear(matrix, rhs, method=method),
                    reference_validation=_validate_lsq_cost,
                )
            assert (matrix.tobytes(), rhs.tobytes()) == before
    A = np.ones(shape)
    check(lambda: optimize.lsq_linear(A, np.ones(shape[0]), lsq_solver="exact"))


def _validate_lsq_cost(expected, actual):
    # Only the observed alignment-sensitive dot-product cost may vary. The
    # solution, residual, singular values, status and every other field stay exact.
    assert type(expected) is type(actual) is optimize.OptimizeResult
    assert expected.keys() == actual.keys()
    assert_identical(
        {k: v for k, v in expected.items() if k != "cost"}, {k: v for k, v in actual.items() if k != "cost"}
    )
    for result in (expected, actual):
        assert type(result.cost) is np.float64
        oracle = np.sum(result.fun.astype(np.longdouble) ** 2) / 2
        bound = np.finfo(np.float64).eps * max(1, result.fun.size) * abs(oracle)
        assert np.isfinite(result.cost) and not np.signbit(result.cost)
        assert abs(np.longdouble(result.cost) - oracle) <= bound


def test_lsq_reference_validator_rejects_other_changes():
    result = optimize.OptimizeResult(fun=np.array([1.0, 2.0]), cost=np.float64(2.5), x=np.ones(1))
    near = optimize.OptimizeResult(result, cost=np.nextafter(result.cost, np.inf))
    _validate_lsq_cost(result, near)
    for bad in (
        optimize.OptimizeResult(result, x=np.zeros(1)),
        optimize.OptimizeResult(result, cost=np.float64(2.51)),
        optimize.OptimizeResult(result, cost=np.float64(np.nan)),
    ):
        with pytest.raises(AssertionError):
            _validate_lsq_cost(result, bad)


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


@pytest.mark.parametrize("n", [2, 3, 10, 17, 1000])
def test_rosen_derivative_family(n, threads):
    local = np.random.default_rng(982 + n)
    for _ in range(12):
        x, p = local.standard_normal(2 * n), local.standard_normal(2 * n)
        x[::5], p[::3] = -0.0, 0.0
        for point, vector in ((x[:n], p[:n]), (x[::2], p[::-2]), (np.zeros(n), np.zeros(n))):
            point.flags.writeable = False
            vector.flags.writeable = False
            before = point.tobytes(), vector.tobytes()
            check(lambda: optimize.rosen_der(point))
            check(lambda: optimize.rosen_hess_prod(point, vector))
            assert (point.tobytes(), vector.tobytes()) == before


@pytest.mark.parametrize(
    "x,p",
    [
        (np.empty(0), np.empty(0)),
        (np.ones(1), np.ones(1)),
        (np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
        (np.ones((2, 3)), np.ones((2, 3))),
        ([1.0, 2.0], [1.0, 2.0]),
    ],
)
def test_rosen_derivative_family_declines(x, p):
    check(lambda: optimize.rosen_der(x), fast=False)
    check(lambda: optimize.rosen_hess_prod(x, p), fast=False)
    check(lambda: optimize.rosen_hess_prod(np.ones(3), np.ones(2)), fast=False)


@pytest.mark.parametrize("name", ["rosen_der", "rosen_hess_prod"])
@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, 1e200, 1e-310])
def test_rosen_derivative_family_warnings(name, special, monkeypatch):
    from fastscipy import _patch

    monkeypatch.setattr(_patch, "STRICT", False)
    x = np.array([special, -special, 1.0])

    def outcome():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                out = (
                    getattr(optimize, name)(x, np.ones(3)) if name == "rosen_hess_prod" else getattr(optimize, name)(x)
                )
            except Exception as error:
                out = type(error), str(error)
        return out, [(w.category, str(w.message)) for w in caught]

    for settings in ({"all": "warn"}, {"all": "raise"}, {"over": "warn", "invalid": "raise"}):
        with np.errstate(**settings):
            with fastscipy.disabled():
                expected = outcome()
            assert_identical(expected, outcome())


@pytest.mark.parametrize("name", ["rosen_der", "rosen_hess_prod"])
def test_rosen_derivative_family_error_callback_once(name, monkeypatch):
    from fastscipy import _patch

    monkeypatch.setattr(_patch, "STRICT", False)
    calls = []
    error = RuntimeError("numeric derivative callback")

    def callback(*args):
        calls.append(args)
        raise error

    with np.errstate(over="call", call=callback), pytest.raises(RuntimeError) as caught:
        args = (
            (np.array([1e200, 1.0, 0.0]), np.ones(3)) if name == "rosen_hess_prod" else (np.array([1e200, 1.0, 0.0]),)
        )
        getattr(optimize, name)(*args)
    assert caught.value is error and len(calls) == 1


@pytest.mark.parametrize("name", ["rosen_der", "rosen_hess_prod"])
def test_rosen_derivative_family_mutating_numeric_callback(name):
    outcomes = []
    for disabled in (True, False):
        x = np.array([1e200, 1.0, 0.0])
        calls = []

        def callback(*args):
            calls.append(args)
            x[0] = 1.0

        with np.errstate(all="call", call=callback):
            context = fastscipy.disabled() if disabled else warnings.catch_warnings()
            with context:
                out = (
                    getattr(optimize, name)(x, np.ones(3)) if name == "rosen_hess_prod" else getattr(optimize, name)(x)
                )
        outcomes.append((out, calls, x))
    assert outcomes[0][1]
    assert_identical(outcomes[0], outcomes[1])


@pytest.mark.parametrize("method", ["Newton-CG", "trust-ncg"])
def test_minimize_with_accelerated_hessian_product(method, threads):
    x = np.array([-1.2, 1.0, 0.8])

    def run():
        history = []
        result = optimize.minimize(
            optimize.rosen,
            x,
            method=method,
            jac=optimize.rosen_der,
            hessp=optimize.rosen_hess_prod,
            callback=lambda point: history.append(point.copy()),
        )
        return result, history

    check(run)
