"""Loader and rollback checks independent of the installed extension ABI."""

import importlib
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
