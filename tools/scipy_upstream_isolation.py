"""Prove upstream baseline/patch state after fixtures, without patching SciPy."""

import importlib
import json
from pathlib import Path
import os

import numpy as np
import pytest
import scipy
from scipy.ndimage import _filters
from scipy import special

import fastscipy
from fastscipy import _patch


def pytest_addoption(parser):
    parser.addoption("--upstream-isolation-report", required=True)
    parser.addoption("--upstream-expected-state", choices=("baseline", "patched"), required=True)
    parser.addoption("--upstream-required-components", default="")
    parser.addoption("--upstream-require-fast", action="store_true")


def _write(config):
    path = Path(config.getoption("upstream_isolation_report"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config._upstream_isolation, indent=2), encoding="utf-8")


def pytest_configure(config):
    config._upstream_isolation = {
        "plugin": "scipy_upstream_isolation",
        "expected_state": config.getoption("upstream_expected_state"),
        "required_components": [c for c in config.getoption("upstream_required_components").split(",") if c],
        "cwd": os.getcwd(),
        "rootdir": str(config.rootpath),
        "scipy": scipy.__file__,
        "fastscipy": fastscipy.__file__,
        "conftests": [],
        "checked_calls": 0,
        "valid": True,
    }
    _write(config)


def pytest_collection_finish(session):
    config = session.config
    proof = config._upstream_isolation
    root = Path(scipy.__file__).resolve().parent
    paths = sorted({str(Path(item.path).resolve()) for item in session.items})
    proof["test_paths"] = paths
    if not all(Path(path).is_relative_to(root) for path in paths):
        proof.update(valid=False, error="collected tests outside installed SciPy")
        _write(config)
        pytest.exit("Upstream isolation failed: collected tests outside installed SciPy", returncode=2)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    config = item.config
    proof = config._upstream_isolation
    scipy_root = Path(scipy.__file__).resolve().parent
    conftests = sorted(str(Path(p.__file__).resolve()) for p in config.pluginmanager._conftest_plugins)
    proof["conftests"] = conftests
    status = fastscipy.status()
    original_bindings = isinstance(special.expit, np.ufunc) and _filters._nd_image is importlib.import_module(
        "scipy.ndimage._nd_image"
    )
    proof["status"] = status
    proof["original_bindings"] = original_bindings
    try:
        assert all(Path(p).is_relative_to(scipy_root) for p in conftests), "non-SciPy conftest loaded"
        if proof["expected_state"] == "baseline":
            assert not status and not _patch._records and not _patch._identity_records, "baseline was patched"
            assert original_bindings, "baseline bindings are not original SciPy"
            assert not any(fastscipy.stats().values()), "baseline contains fastscipy call accounting"
        else:
            assert any(v == "on" for v in status.values()) and _patch._records, "patched child is inactive"
            assert all(status.get(c) == "on" for c in proof["required_components"]), "requested component is inactive"
    except AssertionError as error:
        proof.update(valid=False, error=str(error))
        _write(config)
        pytest.exit(f"Upstream isolation failed: {error}", returncode=2)
    proof["checked_calls"] += 1


def pytest_sessionfinish(session):
    config = session.config
    proof = config._upstream_isolation
    proof["fast_calls"] = sum(v for (_, path), v in fastscipy.stats().items() if path == "fast")
    proof["fast_errors"] = sum(v for (_, path), v in fastscipy.stats().items() if path == "error")
    if proof["expected_state"] == "patched" and config.getoption("upstream_require_fast") and not proof["fast_calls"]:
        proof.update(valid=False, error="requested partition executed no fast paths")
        session.exitstatus = 2
    _write(config)
