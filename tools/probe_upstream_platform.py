"""Bounded Windows ARM diagnosis of an upstream NumPy solve access violation.

Run directly as ``python tools/probe_upstream_platform.py``. This diagnostic
never excludes tests or treats a native crash as a successful test outcome.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import xml.etree.ElementTree as ET


ACCESS_VIOLATION = {0xC0000005, -1073741819}
METHOD = "TestSolve::test_structure_detection"
COLLECT_CODE = r"""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

class Collect:
    def pytest_collection_finish(self, session):
        cases = []
        for item in session.items:
            params = getattr(getattr(item, "callspec", None), "params", {})
            if (getattr(item, "originalname", None) != "test_structure_detection"
                    or getattr(getattr(item, "cls", None), "__name__", None) != "TestSolve"
                    or params.get("assume_a") != "banded"
                    or params.get("transposed") is not False
                    or np.dtype(params.get("dtype")) != np.dtype(np.complex128)):
                continue
            cases.append({
                "nodeid": item.nodeid,
                "absolute_nodeid": str(item.path) + "::" + item.nodeid.split("::", 1)[1],
                "parameters": {"fortran": params["fortran"], "overwrite": params["overwrite"],
                               "transposed": params["transposed"], "nrhs": list(params["nrhs"]),
                               "assume_a": params["assume_a"], "dtype": str(np.dtype(params["dtype"]))},
            })
        Path(sys.argv[1]).write_text(json.dumps(cases, indent=2), encoding="utf-8")

raise SystemExit(pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", sys.argv[2]], plugins=[Collect()]))
"""
NUMPY_CODE = r"""
import hashlib
import json
import sys
import numpy as np

rng = np.random.default_rng(982345982439826)
n = 20
b = rng.random(size=n)
a = rng.random(size=(n, n))
b = b + rng.random(size=n) * 1j
a = a + rng.random(size=(n, n)) * 1j
a = np.asfortranarray(np.triu(np.tril(a, 2), -1))
a_copy = a.copy(order="A")
b_copy = b.copy()
print(json.dumps({"numpy": np.__version__, "scipy_imported": any(
    name == "scipy" or name.startswith("scipy.") for name in sys.modules),
    "a_shape": a_copy.shape, "a_dtype": str(a_copy.dtype), "a_fortran": bool(a_copy.flags.f_contiguous),
    "b_shape": b_copy.shape, "a_sha256_C_order": hashlib.sha256(a_copy.tobytes()).hexdigest(),
    "b_sha256": hashlib.sha256(b_copy.tobytes()).hexdigest(), "phase": "before np.linalg.solve"}), flush=True)
result = np.linalg.solve(a_copy, b_copy)
print(json.dumps({"phase": "returned", "shape": result.shape,
                  "result_sha256": hashlib.sha256(result.tobytes()).hexdigest()}), flush=True)
"""


def last_parameter(log):
    """Recover the last started parametrization from verbose pytest output."""
    matches = re.findall(r"test_structure_detection\[[^\r\n]*?\]", log)
    return matches[-1] if matches else None


def main():
    started = time.monotonic()
    deadline = started + 90
    output = Path("artifacts/upstream-probe").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "diagnostics_only": True,
        "prior_exact_reproduction_run": "https://github.com/michael-L-i/fastscipy/actions/runs/39375594936",
        "quarantine_applied": False,
        "runs": {},
        "pairs": [],
    }
    summary = output / "summary.json"

    def save():
        report["seconds"] = time.monotonic() - started
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
    report["budgets"] = {
        "rss_mb_per_child": 768,
        "timeout_seconds_per_reproduction_child": 5,
        "timeout_seconds_paired_helper": 30,
        "global_seconds": 90,
        "max_children_serial": 19,
    }
    source = Path(scipy.__file__).parent / "linalg/tests/test_basic.py"
    method_target = f"{source}::{METHOD}"
    report["method_target"] = method_target
    report["source_exists"] = source.is_file()
    if source.is_file():
        report["upstream_test_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    save()
    if not source.is_file():
        raise RuntimeError("Installed SciPy does not contain the expected upstream test source")

    def execute(name, command, timeout=5, **details):
        log = output / f"{name}.log"
        record = {"command": command, **details, "status": "running"}
        report["runs"][name] = record
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            record.update(status="not_started_deadline", complete=False)
            save()
            return None
        save()
        print("Starting", name, details.get("target", ""), flush=True)
        result = run(command, log, memory_mb=768, timeout=min(timeout, remaining))
        text = log.read_text(encoding="utf-8", errors="replace")
        result.update(
            {
                **record,
                "access_violation": result["returncode"] in ACCESS_VIOLATION,
                "last_parameter": last_parameter(text),
                "log": str(log),
                "status": "finished",
            }
        )
        report["runs"][name] = result
        save()
        print(
            name,
            "exit",
            result["returncode"],
            "last parameter:",
            result["last_parameter"],
            "access violation:",
            result["access_violation"],
            flush=True,
        )
        # Full verbose progress and crash stack stay in the uploaded log.
        print("\n".join(text.splitlines()[-5:]), flush=True)
        return result

    # First test NumPy in a fresh interpreter that never imports SciPy. Its
    # generated matrix/RHS exactly match the previously crashing parameter.
    execute(
        "numpy-only",
        [sys.executable, "-X", "faulthandler", "-c", NUMPY_CODE],
        patched=False,
        target="numpy.linalg.solve",
    )

    nodes = output / "collected-nodes.json"
    collection = execute(
        "collect", [sys.executable, "-c", COLLECT_CODE, str(nodes), method_target], patched=False, target=method_target
    )
    if collection is None or collection["returncode"] != 0 or not nodes.is_file():
        report["collection_error"] = "Actual upstream node collection did not complete"
        save()
        return
    cases = json.loads(nodes.read_text(encoding="utf-8"))
    report["collected_cases"] = cases
    combinations = {
        (c["parameters"]["fortran"], c["parameters"]["overwrite"], tuple(c["parameters"]["nrhs"])) for c in cases
    }
    expected = {(f, o, rhs) for f in (False, True) for o in (False, True) for rhs in ((), (5,))}
    if len(cases) != 8 or len({c["nodeid"] for c in cases}) != 8 or combinations != expected:
        report["collection_error"] = "Expected eight actual fortran/overwrite/nrhs parameter combinations"
        save()
        return
    save()
    for index, case in enumerate(cases):
        pair = {"nodeid": case["nodeid"], "parameters": case["parameters"], "runs": {}}
        report["pairs"].append(pair)
        for patched in (False, True):
            name = f"case-{index}-{'patched' if patched else 'baseline'}"
            command = [sys.executable, "-m", "pytest", "-vv", "--tb=short", "-p", "no:cacheprovider"]
            if patched:
                command.extend(["-p", "fastscipy.pytest_plugin"])
            command.extend([f"--junitxml={output / (name + '.xml')}", case["absolute_nodeid"]])
            result = execute(name, command, patched=patched, target=case["absolute_nodeid"])
            pair["runs"]["patched" if patched else "baseline"] = name
            if result is None:
                report["deadline_exhausted"] = True
                save()
                return
        save()
    report["all_case_pairs_attempted"] = len(report["pairs"]) == 8
    # The vendor reproductions above remain unquarantined. Independently check
    # the helper's actual paired skip activation without validating a new wheel.
    helper = Path(__file__).with_name("scipy_suite.py")
    validation = {"status": "not_started", "vendor_reproductions_quarantined": False}
    report["paired_quarantine_validation"] = validation
    if not helper.is_file():
        validation.update(status="skipped", reason="Paired quarantine helper is absent")
    else:
        destination = output / "paired-quarantine-validation"
        result = execute(
            "paired-quarantine-validation",
            [sys.executable, str(helper), "--output-dir", str(destination), method_target],
            timeout=30,
            target=method_target,
            paired=True,
        )
        if result is None:
            validation.update(status="incomplete", reason="Global diagnostic deadline exhausted")
        else:
            validation["returncode"] = result["returncode"]
            validation["stopped"] = result["stopped"]
            sides = {}
            outcomes = []
            try:
                for patched in (False, True):
                    quarantine = json.loads((destination / f"0-{patched}.quarantine.json").read_text(encoding="utf-8"))
                    cases_xml = list(ET.parse(destination / f"0-{patched}.xml").getroot().iter("testcase"))
                    states = {}
                    skipped = []
                    for case in cases_xml:
                        tags = {child.tag for child in case}
                        state = (
                            "failed" if tags & {"failure", "error"} else "skipped" if "skipped" in tags else "passed"
                        )
                        states[f"{case.get('classname')}::{case.get('name')}"] = state
                        if state == "skipped":
                            skipped.append(case.get("name"))
                    sides["patched" if patched else "baseline"] = {
                        "quarantine": quarantine,
                        "testcases": len(cases_xml),
                        "unique_testcases": len(states),
                        "skipped": skipped,
                        "failed": sum(state == "failed" for state in states.values()),
                    }
                    outcomes.append(states)
                baseline, patched = outcomes
                validation.update(
                    sides=sides,
                    changed_outcomes=sorted(
                        key for key in baseline.keys() & patched.keys() if baseline[key] != patched[key]
                    ),
                    missing_or_extra=sorted(baseline.keys() ^ patched.keys()),
                )
                expected_ids = {case["nodeid"] for case in cases}
                expected_names = {node.rsplit("::", 1)[-1] for node in expected_ids}
                valid = result["returncode"] == 0 and result["stopped"] is None
                for side in sides.values():
                    valid &= (
                        side["quarantine"]["active"] is True
                        and len(side["quarantine"]["selected"]) == 8
                        and {item["nodeid"] for item in side["quarantine"]["selected"]} == expected_ids
                        and side["testcases"] == side["unique_testcases"] == 416
                        and len(side["skipped"]) == 8
                        and set(side["skipped"]) == expected_names
                        and side["failed"] == 0
                    )
                valid &= not validation["changed_outcomes"] and not validation["missing_or_extra"]
                validation["status"] = "passed" if valid else "failed"
            except (OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
                validation.update(status="incomplete", error=str(error))
        print("Paired quarantine validation:", validation["status"], flush=True)
    report["complete_test_outcomes"] = all(
        item.get("returncode") in (0, 1) and item.get("stopped") is None for item in report["runs"].values()
    )
    save()
    print("Diagnostic summary:", summary, flush=True)


if __name__ == "__main__":
    main()
