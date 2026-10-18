"""Run SciPy's own test suite with fastscipy patched in:

    pytest -p fastscipy.pytest_plugin --pyargs scipy.ndimage -n 2

``FASTSCIPY_COMPONENTS=ndimage,special`` limits the components patched.
"""
import os

import fastscipy


def pytest_configure(config):
    names = os.environ.get("FASTSCIPY_COMPONENTS")
    fastscipy.patch(names.split(",") if names else None)


def pytest_terminal_summary(terminalreporter):
    counts = fastscipy.stats()
    fast = sum(v for (_, path), v in counts.items() if path == "fast")
    errors = sum(v for (_, path), v in counts.items() if path == "error")
    terminalreporter.write_line(f"fastscipy: {fastscipy.status()}")
    terminalreporter.write_line(f"fastscipy: {fast} fast-path calls, {errors} fast-path errors")
