"""Bounded Windows ARM diagnosis of an upstream NumPy solve access violation.

Run directly as ``python tools/probe_upstream_platform.py``. This diagnostic
never excludes tests or treats a native crash as a successful test outcome.
"""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys


ACCESS_VIOLATION = {0xC0000005, -1073741819}
METHOD = "TestSolve::test_structure_detection"


def last_parameter(log):
    """Recover the last started parametrization from verbose pytest output."""
    matches = re.findall(r"test_structure_detection\[[^\r\n]*?\]", log)
    return matches[-1] if matches else None


def main():
    output = Path("artifacts/upstream-probe")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "diagnostics_only": True,
        "runs": {},
    }
    summary = output / "summary.json"

    def save():
        summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if platform.system() != "Windows" or platform.machine().lower() not in {"arm64", "aarch64"}:
        report["skipped"] = "The observed upstream crash is specific to native Windows ARM."
        save()
        print(report["skipped"], flush=True)
        return

    # Direct script execution puts this file's directory on sys.path.
    from platform_suite import run
    import scipy

    report["versions"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "fastscipy")}
    provenance = Path("artifacts/provenance.json")
    report["workflow_provenance"] = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else None
    for key, value in {
        "FASTSCIPY_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "SCIPY_AVAILABLE_MEM": "512Mib",
        # Guarantee that only the explicitly requested plugin patches SciPy.
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
    }.items():
        os.environ[key] = value
    report["budgets"] = {"rss_mb_per_child": 768, "timeout_seconds_per_child": 30, "max_children_serial": 3}
    source = Path(scipy.__file__).parent / "linalg/tests/test_basic.py"
    method_target = f"{source}::{METHOD}"
    report["method_target"] = method_target
    report["source_exists"] = source.is_file()
    save()
    if not source.is_file():
        raise RuntimeError("Installed SciPy does not contain the expected upstream test source")

    def execute(name, target, patched):
        log = output / f"{name}.log"
        command = [sys.executable, "-m", "pytest", "-vv", "--tb=short", "-p", "no:cacheprovider"]
        if patched:
            command.extend(["-p", "fastscipy.pytest_plugin"])
        command.extend([f"--junitxml={output / (name + '.xml')}", target])
        report["runs"][name] = {"command": command, "target": target, "patched": patched, "status": "running"}
        save()
        print("Starting", name, target, flush=True)
        result = run(command, log, memory_mb=768, timeout=30)
        text = log.read_text(encoding="utf-8", errors="replace")
        parameter = last_parameter(text)
        result.update(
            {
                "command": command,
                "target": target,
                "patched": patched,
                "access_violation": result["returncode"] in ACCESS_VIOLATION,
                "last_parameter": parameter,
                "log": str(log),
                "status": "finished",
            }
        )
        report["runs"][name] = result
        save()
        print(name, "last parameter:", parameter, "access violation:", result["access_violation"], flush=True)
        # Full verbose progress and crash stack stay in the uploaded log.
        print("\n".join(text.splitlines()[-8:]), flush=True)
        return result

    # The earlier -q upstream crash did not persist its final parameter ID.
    # First locate it, then reproduce that exact node in independent processes.
    discovery = execute("baseline-discovery", method_target, False)
    target = method_target
    if discovery["access_violation"] and discovery["last_parameter"]:
        target = f"{source}::TestSolve::{discovery['last_parameter']}"
        baseline = execute("baseline-exact", target, False)
    else:
        baseline = discovery
    patched = execute("patched-exact" if target != method_target else "patched-method", target, True)
    report["paired_target"] = target
    report["baseline_access_violation_reproduced"] = baseline["access_violation"]
    report["patched_access_violation"] = patched["access_violation"]
    report["complete_test_outcomes"] = all(
        item["returncode"] in (0, 1) and item["stopped"] is None for item in report["runs"].values()
    )
    report["quarantine_applied"] = False
    save()
    print("Diagnostic summary:", summary, flush=True)


if __name__ == "__main__":
    main()
