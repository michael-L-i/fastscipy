"""Qualify older-Python wheels without weakening the full current-stack gate.

Only the explicit component set below is claimed by this gate. Each wheel runs
its installed ABI canary, the component project suites, and paired upstream
SciPy tests. Reports must be fresh and remain attached to release artifacts.
"""

import argparse
import json
import os
from pathlib import Path
import sys

# -I deliberately removes checkout imports. These are validation helpers, not
# the package under test; fastscipy must still resolve to the installed wheel.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_compatibility import inspect  # noqa: E402
from platform_suite import ROOT, run  # noqa: E402

COMPONENTS = ("ndimage", "special", "spatial")


def main():
    import scipy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["FASTSCIPY_STRICT"] = "1"
    os.environ.setdefault("SCIPY_AVAILABLE_MEM", "2048Mib")
    report = {"scope": "older-Python component qualification", "required_components": COMPONENTS, "runs": {}}
    report["installed"] = inspect(COMPONENTS)
    summary = output / "summary.json"
    summary.write_text(json.dumps(report, indent=2))
    commands = {
        "project": [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            str(ROOT / "tests/test_compatibility.py"),
            *(str(ROOT / "tests" / ("test_" + component + ".py")) for component in COMPONENTS),
            "-q",
            "--tb=short",
            "--junitxml=" + str(output / "project.xml"),
        ],
    }
    for component in COMPONENTS:
        commands["upstream-" + component] = [
            sys.executable,
            "-I",
            str(ROOT / "tools/scipy_suite.py"),
            "--require-fast",
            "--output-dir",
            str(output / ("upstream-" + component)),
            "--pyargs",
            "scipy." + component,
            # Old SciPy otherwise shortens JUnit names to tests.test_*.
            # Keep package provenance so only the known special ufunc
            # collection differences receive the comparator's exception.
            "--rootdir",
            str(Path(scipy.__file__).resolve().parent.parent),
        ]
    for name, command in commands.items():
        result = run(command, output / (name + ".log"), 4096, 14400)
        report["runs"][name] = result
        summary.write_text(json.dumps(report, indent=2))
        if result["returncode"] or result["stopped"]:
            raise SystemExit(1)
    report["qualification"] = "passed"
    summary.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
