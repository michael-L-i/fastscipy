"""Loader and rollback checks independent of the installed extension ABI."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

import fastscipy
from fastscipy import _patch, _threads


def test_missing_native_keeps_python_thread_budget(monkeypatch):
    def unavailable(name):
        raise ImportError("test ABI unavailable")

    monkeypatch.setattr(_threads, "_native", None)
    monkeypatch.setattr(_threads, "_native_error", None)
    monkeypatch.setattr(_threads, "_fallback_threads", None)
    monkeypatch.setattr(importlib, "import_module", unavailable)
    _threads.set_num_threads(2)
    assert _threads.get_num_threads() == 2
    with pytest.raises(ImportError, match="native extension unavailable: test ABI unavailable"):
        _threads.require_native()
    with pytest.raises(ValueError, match="threads must be >= 1"):
        _threads.set_num_threads(0)


def test_native_programming_error_is_not_hidden(monkeypatch):
    def broken(name):
        raise RuntimeError("loader bug")

    monkeypatch.setattr(_threads, "_native", None)
    monkeypatch.setattr(_threads, "_native_error", None)
    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(RuntimeError, match="loader bug"):
        _threads.native()


def test_partial_component_failure_restores_original(monkeypatch):
    owner = types.SimpleNamespace(hook=object())
    original = owner.hook

    def broken():
        _patch.replace(owner, "hook", object())
        raise AttributeError("missing private hook")

    original_import = importlib.import_module

    def load(name, *args, **kwargs):
        if name == "fastscipy._compatibility_test":
            return types.SimpleNamespace(patch=broken)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", load)
    try:
        result = fastscipy.patch(["compatibility_test"])
        assert result["compatibility_test"] == "off: AttributeError: missing private hook"
        assert owner.hook is original
    finally:
        fastscipy.unpatch(["compatibility_test"])


def test_special_forwards_only_available_ufunc_attributes():
    import numpy as np

    from fastscipy._special import _make

    wrapped = _make(np.add, "numpy")
    for attr in ("nin", "nout", "types", "reduce"):
        assert getattr(wrapped, attr) == getattr(np.add, attr)


def test_legacy_morphology_adapts_probe_signature(monkeypatch):
    from fastscipy import _ndimage_morph as morph

    calls = []

    def erode(input, structure, iterations, mask, output, border_value, origin, invert, brute_force):
        calls.append((input, origin))
        return input

    # Capture the normalized callback without invoking a native kernel.
    def candidate(callback):
        callback("input", None, 1, None, None, 0, [0], 1, False, None)
        return lambda *args: NotImplemented

    monkeypatch.setattr(morph._morphology, "_binary_erosion", erode)
    monkeypatch.setattr(morph, "_fixed_point_dilation", candidate)
    monkeypatch.setattr(_patch, "replace", lambda *args: None)
    monkeypatch.setattr(_patch, "rebind", lambda *args: pytest.fail("EDT must remain original"))
    morph.patch(edt=False)
    assert calls == [("input", [0])]


@pytest.mark.parametrize("version", ["2.6.0", "2.7.0", "2.6.0rc1", "2.5.0.dev0", "3.0.0"])
def test_future_numpy_keeps_scipy_original(monkeypatch, version):
    import numpy as np
    from scipy import special

    monkeypatch.delenv("FASTSCIPY_ANY_VERSION", raising=False)
    monkeypatch.setattr(np, "__version__", version)
    before = special.erf
    with pytest.raises(RuntimeError, match="unvalidated NumPy"):
        _patch.require_numpy()
    result = fastscipy.patch(["future_numpy_test"])
    try:
        assert "unvalidated NumPy" in result["future_numpy_test"]
        assert special.erf is before
        assert special.erf(0.0) == 0.0
    finally:
        fastscipy.unpatch(["future_numpy_test"])


@pytest.mark.parametrize("version", ["1.10.1", "1.11.4", "1.12.0"])
def test_legacy_cdflib_is_not_threaded(version):
    from fastscipy._special import _CDFLIB, _thread_safe

    for name in _CDFLIB:
        assert not _thread_safe(types.SimpleNamespace(__name__=name), version)
    assert _thread_safe(types.SimpleNamespace(__name__="erf"), version)
    assert _thread_safe(types.SimpleNamespace(__name__="bdtrik"), "1.13.0")


def test_legacy_cdflib_bindings_stay_original():
    import sys

    import numpy as np
    import scipy
    import scipy.special as special
    from scipy.special import _ufuncs

    from fastscipy._special import _CDFLIB

    if not scipy.__version__.startswith(("1.10.", "1.11.", "1.12.")):
        pytest.skip("Fortran CDFLIB was replaced in SciPy 1.13")
    fastscipy.unpatch(["special"])
    originals = {name: getattr(_ufuncs, name) for name in _CDFLIB}
    bindings = [
        (module, attr, value)
        for name, module in list(sys.modules.items())
        if module is not None and (name == "scipy" or name.startswith("scipy."))
        for attr, value in list(vars(module).items())
        if any(value is original for original in originals.values())
    ]
    try:
        assert fastscipy.patch(["special"])["special"] == "on"
        for name, original in originals.items():
            assert isinstance(original, np.ufunc)
            assert getattr(_ufuncs, name) is original
            assert getattr(special, name) is original
        for module, attr, original in bindings:
            assert getattr(module, attr) is original
        x = np.linspace(-2.0, 2.0, 20)
        mask = np.arange(20) % 2 == 0
        expected = np.full(20, -1.0)
        actual = expected.copy()
        originals["stdtr"](3, x, out=expected, where=mask)
        special.stdtr(3, x, out=actual, where=mask)
        assert actual.tobytes() == expected.tobytes()
    finally:
        fastscipy.unpatch(["special"])
        fastscipy.patch(["special"])


def test_fresh_import_survives_unavailable_native():
    script = """
import importlib.abc
import sys

class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fastscipy._native":
            raise ImportError("test incompatible extension ABI")

sys.meta_path.insert(0, Unavailable())
import fastscipy
assert "fastscipy._native" not in sys.modules
from scipy import special
original = special.erf
status = fastscipy.patch(["ndimage"], threads=2)
assert "native extension unavailable: test incompatible extension ABI" in status["ndimage"]
assert special.erf is original and special.erf(0.0) == 0.0
assert fastscipy.get_num_threads() == 2
"""
    env = dict(os.environ, PYTHONPATH=str(Path(fastscipy.__file__).resolve().parent.parent))
    subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, timeout=20)
