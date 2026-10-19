import warnings

import numpy as np
import pytest
from scipy import integrate

import fastscipy
from fastscipy.testing import assert_identical, check

rng = np.random.default_rng(1)


@pytest.mark.parametrize("n", [1, 5, 16])
@pytest.mark.parametrize("bounds", [(0, 1), (1.0, -2.0), (0.25, 0.25)])
def test_fixed_quad_callback_arguments_and_outputs(n, bounds):
    a, b = bounds
    for vector in (False, True):

        def run():
            calls = []

            def func(y, scale):
                calls.append(y.copy())
                return np.stack((np.sin(y) * scale, y**2)) if vector else np.cos(y) * scale

            return integrate.fixed_quad(func, a, b, args=(2.0,), n=n), calls

        check(run)


def test_fixed_quad_exceptions_do_not_repeat_callback():
    calls = []
    error = ValueError("callback error")

    def func(y):
        calls.append(y)
        raise error

    with pytest.raises(ValueError) as caught:
        integrate.fixed_quad(func, 0, 1)
    assert caught.value is error and len(calls) == 1
    calls.clear()

    def bad_result(y):
        calls.append(y)
        return np.ones((2, 3))

    with pytest.raises(ValueError):
        integrate.fixed_quad(bad_result, 0, 1)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "bounds,kw",
    [
        ((np.float32(0), np.float32(1)), {}),
        ((np.array(0.0), np.array(1.0)), {}),
        ((0.0, np.inf), {}),
        ((0.0, np.nan), {}),
        ((0, 10**400), {}),
        ((0, 1), {"n": 0}),
        ((0, 1), {"n": 1.5}),
        ((0, 1), {"args": [2.0]}),
    ],
)
def test_fixed_quad_fallbacks(bounds, kw):
    check(lambda: integrate.fixed_quad(lambda y, *args: np.cos(y), *bounds, **kw), fast=False)


@pytest.mark.parametrize("n", [3, 5, 17, 101, 1001])
@pytest.mark.parametrize("dx", [1.0, 0.25, -2, 0.0, -0.0])
def test_simpson_uniform(n, dx):
    r = np.random.default_rng(n)
    y = r.standard_normal((2, n, 3))
    for a, axis in ((y, 1), (y[:, ::-1], -2), (y.swapaxes(0, 1), 0), (y[0, :, 0], -1)):
        a.flags.writeable = False
        before = a.tobytes()
        check(lambda: integrate.simpson(a, dx=dx, axis=axis))
        assert a.tobytes() == before


@pytest.mark.parametrize(
    "y,kw",
    [
        (np.arange(10.0), {}),
        (np.ones(1), {}),
        (np.empty(0), {}),
        (np.arange(9, dtype=np.float32), {}),
        (np.arange(9), {}),
        (list(range(9)), {}),
        (np.arange(9.0), {"x": np.linspace(0, 1, 9)}),
        (np.arange(9.0), {"axis": 4}),
        (np.arange(9.0), {"dx": 10**400}),
        (np.array([np.inf, 0.0, -np.inf]), {}),
        (np.array([1e308, 1e308, -1e308]), {}),
    ],
)
def test_simpson_fallback_warnings_and_errors(y, kw):
    def outcome():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = integrate.simpson(y, **kw)
            except Exception as error:
                result = type(error), str(error)
        return result, [(w.category, str(w.message)) for w in caught]

    with fastscipy.disabled():
        expected = outcome()
    assert_identical(expected, outcome())


@pytest.mark.parametrize("n", [1, 2, 17, 1000])
@pytest.mark.parametrize("initial", [None, 0, -0.0])
@pytest.mark.parametrize("axis", [0, -1])
def test_cumulative_trapezoid(n, initial, axis):
    r = np.random.default_rng(n)
    for y in (r.standard_normal(n), r.standard_normal(2 * n)[::-2], np.zeros(n)):
        y.flags.writeable = False
        before = y.tobytes()
        x = np.sort(r.random(n))[::-1]
        for coords, dx in ((None, 1.0), (None, 0.25), (None, -2), (x, 1.0)):
            check(lambda: integrate.cumulative_trapezoid(y, coords, dx=dx, axis=axis, initial=initial))
        assert y.tobytes() == before


def test_cumulative_trapezoid_fallbacks():
    y = np.arange(10.0)
    for a, kw in (
        (y.astype(np.float32), {}),
        (list(y), {}),
        (y[:0], {}),
        (y, {"x": y[:2]}),
        (y, {"x": y.astype(int)}),
        (y, {"axis": 2}),
        (y, {"initial": 1}),
        (y, {"initial": np.array(0)}),
        (y, {"dx": np.float64(0.5)}),
        (y.reshape(2, 5), {}),
    ):
        check(lambda: integrate.cumulative_trapezoid(a, **kw), fast=False)
    for special in (np.nan, np.inf, -np.inf):
        a = y.copy()
        a[2] = special
        check(lambda: integrate.cumulative_trapezoid(a))


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
