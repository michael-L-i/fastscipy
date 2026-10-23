"""Exact ND transform coverage without changing FFT backend/worker semantics."""

import warnings

import numpy as np
import pytest
from scipy import fft, fftpack

import fastscipy
from fastscipy.testing import assert_identical, check, fast_calls

NAMES = ("fftn", "ifftn", "dctn", "idctn", "dstn", "idstn")


@pytest.fixture(params=[1, 2])
def fft_threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield
    fastscipy.set_num_threads(old)


def _outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
        except Exception as error:
            value = type(error), str(error)
    return value, [(w.category, str(w.message)) for w in caught]


@pytest.mark.parametrize("namespace", [fft, fftpack], ids=["fft", "fftpack"])
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_nd_dtype_layout_norm(namespace, name, dtype, fft_threads):
    rng = np.random.default_rng(618)
    x = rng.standard_normal((3, 4, 5)).astype(dtype)
    if x.dtype.kind == "c":
        x += 1j * rng.standard_normal(x.shape).astype(dtype)
    for data in (x, np.asfortranarray(x), x[:, ::-1, ::2], x.transpose(2, 0, 1)):
        data.flags.writeable = False
        before = data.tobytes()
        for axes in (None, (0, 2), (-1, -2), (1,)):
            for norm in (None, "backward", "ortho", "forward"):
                check(
                    lambda: (
                        getattr(namespace, name)(data, axes=axes, norm=norm)
                        if name not in ("fftn", "ifftn") or namespace is fft
                        else getattr(namespace, name)(data, axes=axes)
                    ),
                    fast=name in ("fftn", "ifftn") or data.dtype.kind == "f",
                )
        assert data.tobytes() == before


@pytest.mark.parametrize("namespace", [fft, fftpack], ids=["fft", "fftpack"])
@pytest.mark.parametrize("name", ["dctn", "idctn", "dstn", "idstn"])
@pytest.mark.parametrize("kind", [1, 2, 3, 4])
def test_real_transform_types(namespace, name, kind):
    x = np.random.default_rng(619).standard_normal((4, 5))
    for norm in (None, "ortho", "forward", "backward"):
        check(lambda: getattr(namespace, name)(x, type=kind, norm=norm))
    if namespace is fft:
        for orthogonalize in (False, True):
            check(lambda: getattr(namespace, name)(x, type=kind, orthogonalize=orthogonalize))


@pytest.mark.parametrize("namespace", [fft, fftpack], ids=["fft", "fftpack"])
@pytest.mark.parametrize("name", NAMES)
def test_fallbacks_errors_and_mutation(namespace, name):
    class ArraySubclass(np.ndarray):
        pass

    x = np.arange(20.0).reshape(4, 5)
    unaligned = np.ndarray(x.shape, buffer=bytearray(x.nbytes + 1), offset=1, dtype=np.float64)
    unaligned[:] = x
    inputs = [
        x.astype(np.int64),
        x.astype(np.float16),
        x.astype(np.longdouble),
        x.astype(x.dtype.newbyteorder("S")),
        unaligned,
        x.view(ArraySubclass),
        x.tolist(),
        np.empty((0, 5)),
        np.array(3.0),
    ]
    for data in inputs:
        with fastscipy.disabled():
            expected = _outcome(lambda: getattr(namespace, name)(data))
        before = fast_calls()
        assert_identical(expected, _outcome(lambda: getattr(namespace, name)(data)))
        assert fast_calls() == before
    options = [
        {"axes": []},
        {"axes": ()},
        {"axes": (0, 0)},
        {"axes": (8,)},
        {"axes": [0, 1]},
        {"axes": (np.int64(0),)},
        {"axes": 0},
        {"axes": (0.0,)},
        {"overwrite_x": True},
        {"s" if namespace is fft else "shape": (3, 7)},
        {"s" if namespace is fft else "shape": (-1, -1)},
        {"s" if namespace is fft else "shape": (0, 5)},
    ]
    if namespace is fft or name not in ("fftn", "ifftn"):
        options.extend([{"norm": "bad"}, {"norm": []}])
    if namespace is fft:
        options.extend([{"workers": 0}, {"workers": 1.5}, {"workers": -(10**9)}])
        if name in ("fftn", "ifftn"):
            options.append({"plan": object()})
    if name not in ("fftn", "ifftn"):
        options.extend([{"type": 0}, {"type": 5}, {"type": 2.0}])
    for kwargs in options:
        # Fresh inputs let overwrite_x exercise its exact mutation contract.
        reference, actual = x.copy(), x.copy()
        with fastscipy.disabled():
            expected = _outcome(lambda: getattr(namespace, name)(reference, **kwargs))
        before = fast_calls()
        assert_identical(expected, _outcome(lambda: getattr(namespace, name)(actual, **kwargs)))
        assert_identical(reference, actual)
        assert fast_calls() == before


@pytest.mark.parametrize("namespace", [fft, fftpack], ids=["fft", "fftpack"])
@pytest.mark.parametrize("name", NAMES)
def test_nonfinite_empty_batches_and_singletons(namespace, name):
    arrays = [
        np.array([[0.0, -0.0, np.nan], [np.inf, -np.inf, 1e-310]]),
        np.ones((1, 3)),
        np.empty((0, 3)),
    ]
    for x in arrays:
        for axes in (None, (1,)):
            for settings in ({"all": "warn"}, {"all": "raise"}):
                with np.errstate(**settings):
                    with fastscipy.disabled():
                        expected = _outcome(lambda: getattr(namespace, name)(x, axes=axes))
                    assert_identical(expected, _outcome(lambda: getattr(namespace, name)(x, axes=axes)))
    if name in ("dctn", "idctn"):
        check(lambda: getattr(namespace, name)(np.ones((1, 3)), type=1), fast=False)


def test_workers_preserve_scipy_resolution(monkeypatch, fft_threads):
    from scipy.fft._duccfft import helper, pyduccfft

    compiled = pyduccfft.c2c
    seen = []

    def record(*args):
        seen.append(args[-1])
        return compiled(*args)

    monkeypatch.setattr(pyduccfft, "c2c", record)
    monkeypatch.setattr(helper, "_cpu_count", 2)
    x = np.arange(64.0).reshape(8, 8)
    with fft.set_workers(2):
        check(lambda: fft.fftn(x))
        assert seen[-2:] == [2, 2]
        check(lambda: fft.fftn(x, workers=1))
        assert seen[-2:] == [1, 1]
        check(lambda: fft.fftn(x, workers=-1), fast=False)
        assert seen[-2:] == [2, 2]
        check(lambda: fft.fftn(x, workers=np.int64(2)), fast=False)
        assert seen[-2:] == [2, 2]
        check(lambda: fftpack.fftn(x))
        assert seen[-2:] == [2, 2]
        with fft.set_workers(1):
            check(lambda: fft.fftn(x))
            assert seen[-2:] == [1, 1]
        assert fft.get_workers() == 2
    assert fft.get_workers() == 1


def test_alternate_backends_and_skip():
    token = object()
    calls = []

    class Backend:
        __ua_domain__ = "numpy.scipy.fft"

        @staticmethod
        def __ua_function__(method, args, kwargs):
            calls.append((method.__name__, args, kwargs))
            return token

    x = np.ones((8, 8))
    before = fast_calls()
    with fft.set_backend(Backend, only=True):
        for name in NAMES:
            assert getattr(fft, name)(x, workers=2) is token
    assert fast_calls() == before
    assert [entry[0] for entry in calls] == list(NAMES)
    assert all(entry[1][0] is x and entry[2] == {"workers": 2} for entry in calls)
    with fft.skip_backend("scipy"):
        with pytest.raises(Exception) as caught:
            fft.fftn(x)
        assert type(caught.value).__name__ == "BackendNotImplementedError"


def test_two_dimensional_aliases_and_no_result_cache():
    x = np.arange(64.0).reshape(8, 8)
    for namespace in (fft, fftpack):
        for name in ("fft2", "ifft2"):
            check(lambda: getattr(namespace, name)(x))
    first = fft.fftn(x)
    x[0, 0] += 1
    second = check(lambda: fft.fftn(x))
    np.testing.assert_allclose(second - first, 1, rtol=0, atol=1e-13)
    assert not np.shares_memory(first, second)


def test_independent_fourier_definition_and_orthonormal_properties():
    x = np.random.default_rng(620).standard_normal((4, 5))
    matrices = [np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n) for n in x.shape]
    expected = matrices[0] @ x @ matrices[1].T
    np.testing.assert_allclose(fft.fftn(x), expected, rtol=2e-14, atol=2e-14)
    for namespace in (fft, fftpack):
        for name, inverse in (("dctn", "idctn"), ("dstn", "idstn")):
            y = getattr(namespace, name)(x, norm="ortho")
            np.testing.assert_allclose(getattr(namespace, inverse)(y, norm="ortho"), x, rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(np.sum(y**2), np.sum(x**2), rtol=2e-14)
    # DCT-II definition, independent of SciPy's inverse implementation.
    cosines = [2 * np.cos(np.pi * np.outer(np.arange(n), 2 * np.arange(n) + 1) / (2 * n)) for n in x.shape]
    np.testing.assert_allclose(fft.dctn(x), cosines[0] @ x @ cosines[1].T, rtol=2e-14, atol=2e-14)


def test_guarded_wrapper_disabled_and_error_contract(monkeypatch):
    from fastscipy import _fft, _patch

    x = np.ones((2, 3))
    expected = object()

    def fail(data):
        raise RuntimeError("candidate failure")

    wrapped = _fft._wrap(lambda data: expected, fail, "fft.test_guard", "fd")
    with fastscipy.disabled():
        assert wrapped(x) is expected
    assert _patch.counts["fft.test_guard", "scipy"] == 1
    monkeypatch.setattr(_patch, "STRICT", False)
    assert wrapped(x) is expected
    assert _patch.counts["fft.test_guard", "error"] == 1
    assert _patch.counts["fft.test_guard", "scipy"] == 2
    monkeypatch.setattr(_patch, "STRICT", True)
    with pytest.raises(RuntimeError, match="candidate failure"):
        wrapped(x)
    assert _patch.counts["fft.test_guard", "error"] == 1
    assert _patch.counts["fft.test_guard", "fast"] == 0
