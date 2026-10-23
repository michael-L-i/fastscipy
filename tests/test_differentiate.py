"""Compatibility controls for the currently unaccelerated differentiate area.

These intentionally do not require a fast path. Other patched SciPy components
must not alter numerical derivatives, callback events, or public metadata.
"""

import warnings

import numpy as np
import pytest
from scipy import differentiate

import fastscipy
from fastscipy.testing import assert_identical, check


@pytest.fixture(params=[1, 2])
def differentiation_threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield
    fastscipy.set_num_threads(old)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", [2, 5, 8])
def test_random_derivative(dtype, order, differentiation_threads):
    rng = np.random.default_rng(606 + order)
    x = rng.uniform(0.2, 1.0, (4, 6)).astype(dtype)[:, ::2]
    x.flags.writeable = False
    before = x.copy()
    direction = rng.integers(-1, 2, x.shape)
    out = check(
        lambda: differentiate.derivative(np.sin, x, order=order, maxiter=16, step_direction=direction), fast=False
    )
    assert_identical(x.copy(), before)
    # Independent analytic check, in addition to exact SciPy result comparison.
    np.testing.assert_allclose(out.df, np.cos(x), atol=2e-4 if dtype == np.float32 else 2e-8)


def test_broadcast_args_kwargs_and_preserve_shape(differentiation_threads):
    x = np.linspace(0.2, 0.8, 4)
    a = np.arange(1.0, 4.0)[:, None]

    def call(preserve):
        return differentiate.derivative(
            lambda x, a, *, b: a * x**2 + b,
            x,
            args=(a,),
            kwargs={"b": np.arange(4.0)},
            preserve_shape=preserve,
            initial_step=np.linspace(0.2, 0.4, 4),
        )

    out = check(lambda: call(False), fast=False)
    np.testing.assert_allclose(out.df, 2 * a * x, rtol=1e-10, atol=1e-12)
    broadcast_x = np.broadcast_to(x, (3, 4))
    out = check(lambda: differentiate.derivative(lambda x: x**2, broadcast_x, preserve_shape=True), fast=False)
    np.testing.assert_allclose(out.df, 2 * broadcast_x, rtol=1e-10, atol=1e-12)
    # SciPy's preserved-shape loop rejects these flattened multidimensional
    # arguments. Preserve its exception instead of assuming they are accepted.
    with fastscipy.disabled():
        expected = _outcome(lambda: call(True))
    assert isinstance(expected[0], tuple) and issubclass(expected[0][0], Exception)
    assert_identical(expected, _outcome(lambda: call(True)))


def test_jacobian_hessian_analytic(differentiation_threads):
    x = np.random.default_rng(608).uniform(0.1, 0.7, (3, 2))
    jac = check(lambda: differentiate.jacobian(lambda x: np.sin(x) + np.sum(x, axis=0), x), fast=False)
    expected = np.ones((3, 3, 2))
    expected[np.arange(3), np.arange(3)] += np.cos(x)
    np.testing.assert_allclose(jac.df, expected, rtol=1e-8, atol=1e-10)
    hes = check(lambda: differentiate.hessian(lambda x: np.sum(x**2, axis=0), x), fast=False)
    expected = np.repeat((2 * np.eye(3))[..., None], 2, axis=-1)
    np.testing.assert_allclose(hes.ddf, expected, rtol=1e-7, atol=1e-7)


def _outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
        except Exception as error:
            result = type(error), str(error)
    return result, [(w.category, str(w.message)) for w in caught]


@pytest.mark.parametrize("x", [np.array([]), np.array([np.nan, np.inf, -np.inf]), np.array([0.0, -0.0, 1e300])])
@pytest.mark.parametrize("errors", ["warn", "raise"])
def test_nonfinite_empty_warnings_and_errors(x, errors):
    call = lambda: differentiate.derivative(np.sin, x)
    with np.errstate(all=errors):
        with fastscipy.disabled():
            expected = _outcome(call)
        assert_identical(expected, _outcome(call))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": 0},
        {"maxiter": 0},
        {"initial_step": -1},
        {"step_factor": 1},
        {"step_factor": 0},
        {"step_direction": [0, 1, -1]},
        {"preserve_shape": "invalid"},
    ],
)
def test_validation_errors(kwargs):
    call = lambda: differentiate.derivative(np.sin, np.array([0.2, 0.4]), **kwargs)
    with fastscipy.disabled():
        expected = _outcome(call)
    assert_identical(expected, _outcome(call))


@pytest.mark.parametrize("stop", [None, 0, 2])
def test_callback_trace_termination_and_mutation(stop):
    def run():
        calls, callbacks = [], []
        x = np.array([0.2, 0.5])

        def f(value):
            calls.append(value.copy())
            return np.exp(value)

        def callback(res):
            callbacks.append({k: np.copy(v) for k, v in res.items()})
            if stop is not None and np.max(res.nit) >= stop:
                raise StopIteration

        result = differentiate.derivative(f, x, callback=callback)
        return result, calls, callbacks, x

    check(run, fast=False)


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_user_function_failure_is_not_retried(fail_at):
    def run():
        calls = []

        def f(x):
            calls.append(x.copy())
            if len(calls) == fail_at:
                warnings.warn("callback warning", UserWarning, stacklevel=2)
                raise RuntimeError("callback failure")
            return np.exp(x)

        return _outcome(lambda: differentiate.derivative(f, np.array([0.2, 0.5]))), calls

    check(run, fast=False)
