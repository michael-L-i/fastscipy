"""Visible upstream-only skips for eight independently reproduced NumPy crashes.

Loaded by tools/scipy_suite.py in both baseline and patched interpreters.
This plugin never imports fastscipy or changes a numerical function.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig

import numpy as np
import pytest

PLUGIN = "scipy_upstream_quarantine"
REPRODUCTION = "https://github.com/michael-L-i/fastscipy/actions/runs/39376117965"
TEST_SOURCE_SHA256 = "d40ba24f428b409d9dc8787b33a14927268ebc6e94209560f0cd696484636eee"
REASON = (
    "Known upstream NumPy 2.5.4 Windows ARM64 access violation; independently reproduced "
    "without SciPy, and in both paired runs: " + REPRODUCTION
)


def scope():
    observed = {
        "platform": sys.platform,
        "machine": platform.machine().lower(),
        "implementation": sys.implementation.name,
        "python_platform": sysconfig.get_platform(),
        "python": list(sys.version_info[:3]),
    }
    for name in ("numpy", "scipy"):
        try:
            observed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed[name] = None
    try:
        spec = importlib.util.find_spec("scipy")
        source = Path(spec.origin).resolve().parent / "linalg/tests/test_basic.py"
        observed["test_source"] = str(source)
        observed["test_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    except (OSError, ValueError, TypeError, AttributeError):
        observed["test_source"] = None
        observed["test_source_sha256"] = None
    return observed


def applicable(observed):
    return (
        observed.get("platform") == "win32"
        and observed.get("machine") in ("arm64", "aarch64")
        and observed.get("implementation") == "cpython"
        and observed.get("python_platform") == "win-arm64"
        and observed.get("python") == [3, 13, 16]
        and observed.get("numpy") == "2.5.4"
        and observed.get("scipy") == "1.18.1"
        and observed.get("test_source_sha256") == TEST_SOURCE_SHA256
    )


def matches(item, observed):
    if (
        Path(item.path).resolve() != Path(observed["test_source"])
        or getattr(getattr(item, "cls", None), "__name__", None) != "TestSolve"
        or getattr(item, "originalname", None) != "test_structure_detection"
    ):
        return False
    params = getattr(getattr(item, "callspec", None), "params", {})
    return (
        params.get("assume_a") == "banded"
        and params.get("dtype") is np.complex128
        and params.get("transposed") is False
        and type(params.get("fortran")) is bool
        and type(params.get("overwrite")) is bool
        and type(params.get("nrhs")) is tuple
        and params["nrhs"] in ((), (5,))
    )


def pytest_addoption(parser):
    parser.addoption("--upstream-quarantine-report", required=True, help="JSON report for known paired upstream skips")


def write_report(config):
    path = Path(config.getoption("upstream_quarantine_report"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config._upstream_quarantine, indent=2), encoding="utf-8")


def pytest_configure(config):
    observed = scope()
    config._upstream_quarantine = {
        "plugin": PLUGIN,
        "reproduction": REPRODUCTION,
        "reason": REASON,
        "observed": observed,
        "active": applicable(observed),
        "selected": [],
    }
    write_report(config)


def pytest_collection_modifyitems(config, items):
    report = config._upstream_quarantine
    if report["active"]:
        for item in items:
            if matches(item, report["observed"]):
                item.add_marker(pytest.mark.skip(reason=REASON))
                params = item.callspec.params
                report["selected"].append(
                    {
                        "nodeid": item.nodeid,
                        "parameters": {
                            "fortran": params["fortran"],
                            "overwrite": params["overwrite"],
                            "transposed": params["transposed"],
                            "nrhs": list(params["nrhs"]),
                            "assume_a": params["assume_a"],
                            "dtype": "complex128",
                        },
                    }
                )
    write_report(config)
