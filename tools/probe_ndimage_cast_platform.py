"""Intel Mac selector experiment against an installed wheel, not release validation.

Only _SCIPY_INT_CAST changes. Run existing interpolation tests in a monitored
child (224 MiB, 110 seconds), retaining the installed files' hashes and JUnit.
Other platforms skip this experiment without importing numerical libraries.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "artifacts" / "cast-selector-intel"
PURPOSE = "selector_experiment_not_release_wheel_certification"


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def worker():
    # Match tests/conftest.py before importing the installed package here.
    os.environ.setdefault("FASTSCIPY_STRICT", "1")

    import numpy
    import pytest
    import scipy
    from scipy.ndimage import _nd_image

    import fastscipy
    from fastscipy import _native, _ndimage_interp

    site = Path(sysconfig.get_paths()["platlib"]).resolve()
    for module in (fastscipy, _native, _ndimage_interp):
        if not Path(module.__file__).resolve().is_relative_to(site):
            raise RuntimeError(f"Expected installed wheel, found {module.__file__}")
    original = _ndimage_interp._SCIPY_INT_CAST
    report = {
        "purpose": PURPOSE,
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": sys.version,
        "strict_environment": os.environ["FASTSCIPY_STRICT"],
        "versions": {"fastscipy": fastscipy.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__},
        "installed_interp": fingerprint(_ndimage_interp.__file__),
        "installed_native": fingerprint(_native.__file__),
        "scipy_ndimage_native": fingerprint(_nd_image.__file__),
        "existing_tests": fingerprint(ROOT / "tests" / "test_ndimage.py"),
        "initial_SCIPY_INT_CAST": bool(original),
        "experimental_SCIPY_INT_CAST": True,
        "selector_changed": not bool(original),
        "junit": "junit.xml",
    }
    print(json.dumps(report, indent=2), flush=True)
    try:
        _ndimage_interp._SCIPY_INT_CAST = True
        code = int(
            pytest.main(
                [
                    str(ROOT / "tests" / "test_ndimage.py"),
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-k",
                    "test_spline_zoom_shift or test_affine_rotate_map_coordinates",
                    f"--junitxml={OUTPUT / 'junit.xml'}",
                ]
            )
        )
        report["pytest_exit_code"] = code
        return code
    finally:
        _ndimage_interp._SCIPY_INT_CAST = original
        report["restored_SCIPY_INT_CAST"] = bool(_ndimage_interp._SCIPY_INT_CAST)
        report["selector_restored"] = _ndimage_interp._SCIPY_INT_CAST is original
        (OUTPUT / "worker.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine().lower() != "x86_64":
        print(json.dumps({"purpose": PURPOSE, "skipped": "Intel Mac only"}), flush=True)
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker()

    from platform_suite import run

    for name in ("junit.xml", "worker.json", "worker.log", "monitor.json"):
        (OUTPUT / name).unlink(missing_ok=True)
    result = run([sys.executable, str(Path(__file__).resolve()), "--worker"], OUTPUT / "worker.log", 224, 110)
    result.update(purpose=PURPOSE, memory_limit_mib=224, timeout_seconds=110)
    (OUTPUT / "monitor.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["returncode"] == 0:
        report = json.loads((OUTPUT / "worker.json").read_text(encoding="utf-8"))
        if not report.get("selector_restored") or not (OUTPUT / "junit.xml").is_file():
            return 1
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
