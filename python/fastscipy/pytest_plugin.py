"""Run SciPy's own test suite with fastscipy patched in:

    pytest -p fastscipy.pytest_plugin --pyargs scipy.ndimage -n 2

``FASTSCIPY_COMPONENTS=ndimage,special`` limits the components patched.
"""
import os

import pytest

import fastscipy
from fastscipy import _patch


def pytest_configure(config):
    names = os.environ.get("FASTSCIPY_COMPONENTS")
    fastscipy.patch(names.split(",") if names else None)


def pytest_sessionfinish(session):
    out = getattr(session.config, "workeroutput", None)   # set on pytest-xdist workers
    if out is not None:
        out["fastscipy_counts"] = [[f, p, n] for (f, p), n in _patch.counts.items()]


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    for f, p, n in getattr(node, "workeroutput", {}).get("fastscipy_counts", []):
        _patch.counts[f, p] += n


def pytest_terminal_summary(terminalreporter):
    counts = fastscipy.stats()
    fast = sum(v for (_, path), v in counts.items() if path == "fast")
    errors = sum(v for (_, path), v in counts.items() if path == "error")
    terminalreporter.write_line(f"fastscipy: {fastscipy.status()}")
    terminalreporter.write_line(f"fastscipy: {fast} fast-path calls, {errors} fast-path errors")
