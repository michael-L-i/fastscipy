import warnings

import numpy as np
import pytest
from scipy import integrate

import fastscipy
from fastscipy.testing import assert_identical, check

rng = np.random.default_rng(1)


@pytest.fixture(params=[1, 2])
def integration_threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield
    fastscipy.set_num_threads(old)


def _sampled_outcome(name, y, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = getattr(integrate, name)(y, **kwargs)
        except Exception as error:
            out = type(error), str(error)
    return out, [(w.category, str(w.message)) for w in caught]


@pytest.mark.parametrize("n", [3, 4, 5, 16, 17, 1000])
def test_cumulative_simpson_selected_intervals(n, integration_threads):
    local = np.random.default_rng(930 + n)
    y = local.standard_normal((2, n, 3))
    arrays = [
        (y, 1),
        (y[:, ::-1], -2),
        (y.swapaxes(0, 1), 0),
        (y[0, :, 0], -1),
        (np.zeros((0, n)), -1),
        (np.empty((2, n, 0)), 1),
    ]
    zero = np.zeros(n)
    zero[::2] = -0.0
    arrays.append((zero, -1))
    for data, axis in arrays:
        data.flags.writeable = False
        before = data.tobytes()
        for dx in (1.0, 0.25, -2, 0.0, -0.0):
            for initial in (None, 0, -0.0, 1.25):
                check(lambda: integrate.cumulative_simpson(data, dx=dx, axis=axis, initial=initial))
        assert data.tobytes() == before


@pytest.mark.parametrize("name", ["cumulative_simpson", "romb"])
@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, 1e308, 1e-310])
def test_new_sampled_nonfinite_and_warning_events(name, special):
    y = np.array([1.0, special, -special, 0.0, -0.0])
    for settings in ({}, {"all": "warn"}, {"all": "raise"}):
        with np.errstate(**settings):
            with fastscipy.disabled():
                expected = _sampled_outcome(name, y)
            assert_identical(expected, _sampled_outcome(name, y))


@pytest.mark.parametrize(
    "y,kwargs",
    [
        (np.arange(5.0), {"x": np.arange(5.0)}),
        (np.arange(5.0), {"dx": np.ones(2)}),
        (np.arange(5.0), {"dx": 10**400}),
        (np.arange(5.0), {"initial": 10**400}),
        (np.arange(5.0), {"initial": np.ones(2)}),
        (np.arange(5.0), {"axis": 2}),
        (np.arange(5, dtype=np.float32), {}),
        (np.arange(5), {}),
        (np.empty(0), {}),
        (np.ones(1), {}),
        (np.ones(2), {"initial": 2.0}),
    ],
)
def test_cumulative_simpson_fallbacks(y, kwargs):
    with fastscipy.disabled():
        expected = _sampled_outcome("cumulative_simpson", y, **kwargs)
    assert_identical(expected, _sampled_outcome("cumulative_simpson", y, **kwargs))


def test_cumulative_simpson_discarded_overflow_warning():
    # Only an estimate discarded during interleaving overflows; the retained
    # interval estimates and cumulative sum are finite.
    y = np.array([0.0, 4e307, 0.0, 4e307, 0.0])
    with fastscipy.disabled():
        expected = _sampled_outcome("cumulative_simpson", y)
    assert expected[1]
    assert_identical(expected, _sampled_outcome("cumulative_simpson", y))


@pytest.mark.parametrize("dx", [2**64, -(2**63) - 1, 10**30])
def test_cumulative_simpson_object_spacing_declines(dx):
    from fastscipy._integrate import _cumulative_simpson

    y = np.arange(5.0)
    assert _cumulative_simpson(y, dx=dx) is NotImplemented
    with fastscipy.disabled():
        expected = integrate.cumulative_simpson(y, dx=dx)
    got = integrate.cumulative_simpson(y, dx=dx)
    assert got.dtype == expected.dtype and got.shape == expected.shape and got.strides == expected.strides
    # Object-array bytes contain addresses; compare the numeric scalars.
    assert_identical(expected.tolist(), got.tolist())


@pytest.mark.parametrize("n", [2, 3, 9, 33, 257])
def test_romb_rows(n, integration_threads):
    local = np.random.default_rng(940 + n)
    y = local.standard_normal((2, n, 3))
    for data, axis in (
        (y, 1),
        (y[:, ::-1], -2),
        (y.swapaxes(0, 1), 0),
        (y[0, :, 0], -1),
        (np.empty((0, n)), -1),
        (np.empty((2, n, 0)), 1),
        (np.zeros(n), 0),
    ):
        data.flags.writeable = False
        before = data.tobytes()
        for dx in (1.0, 0.25, -2, 0.0, -0.0):
            check(lambda: integrate.romb(data, dx=dx, axis=axis))
        assert data.tobytes() == before


@pytest.mark.parametrize(
    "y,kwargs",
    [
        (np.empty(0), {}),
        (np.ones(1), {}),
        (np.ones(4), {}),
        (np.ones(5, dtype=np.float32), {}),
        (np.arange(5), {}),
        (np.ones(5), {"dx": 10**400}),
        (np.ones(5), {"axis": 2}),
    ],
)
def test_romb_fallbacks(y, kwargs):
    with fastscipy.disabled():
        expected = _sampled_outcome("romb", y, **kwargs)
    assert_identical(expected, _sampled_outcome("romb", y, **kwargs))


@pytest.mark.parametrize("show", [True, (3, 10)])
def test_romb_printed_table_fallback(show, capsys):
    y = np.arange(5.0)
    with fastscipy.disabled():
        expected = integrate.romb(y, show=show)
    text = capsys.readouterr().out
    assert_identical(expected, integrate.romb(y, show=show))
    assert capsys.readouterr().out == text


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_cumulative_trapezoid_nd_dtypes(dtype, integration_threads):
    local = np.random.default_rng(950)
    y = local.standard_normal((2, 17, 3)).astype(dtype)
    if dtype == np.complex128:
        y += 1j * local.standard_normal(y.shape)
    for data in (y, y[:, ::-1], np.asfortranarray(y), np.empty((0, 17), dtype=dtype)):
        data.flags.writeable = False
        before = data.tobytes()
        for axis in range(data.ndim):
            if not data.shape[axis]:
                continue
            for x in (
                None,
                local.standard_normal(data.shape[axis]).astype(np.float32),
                local.standard_normal(data.shape),
            ):
                for initial in (None, 0, -0.0):
                    check(
                        lambda: integrate.cumulative_trapezoid(data, x, axis=axis, initial=initial),
                        fast=bool(data.size),
                    )
        assert data.tobytes() == before


@pytest.mark.parametrize("dtype", [np.float32, np.complex128])
def test_cumulative_trapezoid_nd_warnings(dtype):
    y = np.array([[1e38, -1e38, np.inf, -np.inf], [1.0, np.nan, -0.0, 0.0]], dtype=dtype)
    for axis in (0, -1):
        for settings in ({}, {"all": "warn"}, {"all": "raise"}):
            with np.errstate(**settings):
                with fastscipy.disabled():
                    expected = _sampled_outcome("cumulative_trapezoid", y, dx=1e30, axis=axis, initial=0)
                assert_identical(expected, _sampled_outcome("cumulative_trapezoid", y, dx=1e30, axis=axis, initial=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_cumulative_trapezoid_warning_then_error_is_not_retried(dtype, monkeypatch):
    from fastscipy import _patch

    monkeypatch.setattr(_patch, "STRICT", False)
    large = np.finfo(np.float32 if dtype == np.float32 else np.float64).max
    y = np.array([large, large, np.inf, -np.inf], dtype=dtype)
    with np.errstate(over="warn", invalid="raise"):
        with fastscipy.disabled():
            expected = _sampled_outcome("cumulative_trapezoid", y)
        assert expected[1] and expected[0][0] is FloatingPointError
        assert_identical(expected, _sampled_outcome("cumulative_trapezoid", y))


def test_cumulative_trapezoid_error_callback_is_not_retried(monkeypatch):
    from fastscipy import _patch

    monkeypatch.setattr(_patch, "STRICT", False)
    calls = []
    error = RuntimeError("floating-point callback")

    def callback(*args):
        calls.append(args)
        raise error

    with np.errstate(over="call", call=callback):
        with pytest.raises(RuntimeError) as caught:
            integrate.cumulative_trapezoid(np.array([1e308, 1e308]))
    assert caught.value is error and len(calls) == 1


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
        (list(y), {}),
        (y[:0], {}),
        (y, {"x": y[:2]}),
        (y, {"x": y.astype(int)}),
        (y, {"axis": 2}),
        (y, {"initial": 1}),
        (y, {"initial": np.array(0)}),
        (y, {"dx": np.float64(0.5)}),
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
