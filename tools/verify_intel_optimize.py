"""Repeat focused Intel optimization correctness using an unchanged a3 wheel.

This is a partial regression check, not release validation or a benchmark.
"""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

WHEEL_RUN = 39680660563
WHEEL_SOURCE = "cf06af752abee2aebffa3b687398f1f76ef94eb8"
WHEEL_ARTIFACT = "debug-wheel-macos-x86_64-cp313"
ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_installed(wheel):
    import fastscipy
    from fastscipy import _native
    import numpy as np
    import scipy

    assert sys.platform == "darwin" and platform.machine() == "x86_64"
    assert sys.version_info[:2] == (3, 13)
    assert not os.environ.get("FASTSCIPY_ANY_PLATFORM")
    assert not os.environ.get("FASTSCIPY_ANY_VERSION")
    assert fastscipy.__version__ == importlib.metadata.version("fastscipy") == "0.1.0a3"
    assert np.__version__ == "2.5.4" and scipy.__version__ == "1.18.1"
    assert np.__config__.show(mode="dicts")["Build Dependencies"]["blas"]["name"] == "accelerate"
    package = Path(fastscipy.__file__).resolve().parent
    native = Path(_native.__file__).resolve()
    assert not package.is_relative_to(ROOT), "The package must come from the installed wheel"
    assert native.parent == package
    files = {}
    with zipfile.ZipFile(wheel) as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        assert len(records) == 1
        for name, expected_hash, expected_size in csv.reader(io.StringIO(archive.read(records[0]).decode())):
            if not expected_hash:
                continue
            data = archive.read(name)
            algorithm, encoded = expected_hash.split("=", 1)
            assert algorithm == "sha256"
            assert base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=") == encoded
            assert len(data) == int(expected_size)
        for name in archive.namelist():
            if name.startswith("fastscipy/") and name.endswith((".py", ".so")):
                data = archive.read(name)
                installed = package / Path(name).relative_to("fastscipy")
                assert installed.read_bytes() == data, f"Installed file differs from wheel: {name}"
                files[name] = sha(data)
    assert (ROOT / "python/fastscipy/testing.py").read_bytes() == (package / "testing.py").read_bytes(), (
        "Current tests require a changed helper; do not silently overlay the installed wheel"
    )
    return {
        "package": str(package),
        "native": str(native),
        "native_sha256": sha(native.read_bytes()),
        "wheel": wheel.name,
        "wheel_sha256": sha(wheel.read_bytes()),
        "installed_files_sha256": files,
        "versions": {"fastscipy": fastscipy.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "summary.json").exists(), "Use a fresh output directory"
    provenance = json.loads((output / "wheel-run.json").read_text())
    assert provenance["id"] == WHEEL_RUN and provenance["head_sha"] == WHEEL_SOURCE
    wheel = args.wheel.resolve()
    assert "-cp313-cp313-macosx_" in wheel.name and wheel.name.endswith("_x86_64.whl")
    installed = verify_installed(wheel)
    helper_path = ROOT / "tools/platform_suite.py"
    spec = importlib.util.spec_from_file_location("_focused_monitor", helper_path)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    report = {
        "purpose": "partial optimize correctness regression with unchanged a3 wheel; no build or benchmark",
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "wheel_source": WHEEL_SOURCE,
        "wheel_run": WHEEL_RUN,
        "wheel_artifact": WHEEL_ARTIFACT,
        "python": sys.version,
        "platform": platform.platform(),
        "installed": installed,
        "source_files_sha256": {
            name: sha((ROOT / name).read_bytes())
            for name in (
                "tests/test_optimize.py",
                "tests/conftest.py",
                "python/fastscipy/testing.py",
                "tools/platform_suite.py",
            )
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "FASTSCIPY_STRICT",
                "FASTSCIPY_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "FASTSCIPY_ANY_PLATFORM",
                "FASTSCIPY_ANY_VERSION",
            )
        },
        "runs": [],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    code = 0
    for repeat in range(1, 4):
        junit = output / f"optimize-{repeat}.xml"
        os.environ["FASTSCIPY_REFERENCE_REPORT"] = str(output / f"numerical-policy-{repeat}.jsonl")
        command = [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "-q",
            "--import-mode=importlib",
            str(ROOT / "tests/test_optimize.py"),
            f"--junitxml={junit}",
        ]
        resources = monitor.run(command, output / f"optimize-{repeat}.log", 512, 900)
        result = {"repeat": repeat, "command": command, "resources": resources}
        if junit.exists():
            suites = list(ET.parse(junit).getroot().iter("testsuite"))
            result["junit"] = [dict(s.attrib) for s in suites]
            result["junit_sha256"] = sha(junit.read_bytes())
            valid = bool(suites) and all(
                int(s.get("tests", 0)) > 0
                and all(int(s.get(field, 0)) == 0 for field in ("errors", "failures", "skipped"))
                for s in suites
            )
        else:
            valid = False
        report["runs"].append(result)
        code |= int(resources["returncode"] != 0 or resources["stopped"] is not None or not valid)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if code:
            break
    assert all(sha((ROOT / name).read_bytes()) == digest for name, digest in report["source_files_sha256"].items())
    assert verify_installed(wheel) == installed, "Installed files changed during the tests"
    report["complete"] = len(report["runs"]) == 3 and code == 0
    report["returncode"] = code
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
