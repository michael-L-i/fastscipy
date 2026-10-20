"""Intel Mac current-Python/unchanged-native diagnostic, never release certification.

Reject native-source changes before reuse. A temporary package overlay retains
current Python files and the installed wheel's exact native binary. Run the full
project suite and all benchmark areas serially, without upstream tests.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "artifacts" / "python-source-intel"
PURPOSE = "current_python_reused_native_diagnostic_not_release_certification"
REMAINING_PURPOSE = "partial_current_python_remaining_tests_and_95_cases_reused_native_not_release_certification"
REMAINING_TESTS = (
    "tests/test_sparse_linalg.py",
    "tests/test_stats.py",
    "tests/test_reference_variance.py",
    "tests/test_bench_reference.py",
    "tests/test_optimize.py",
)


def remaining_suite(root, output):
    """Complete a named subset without relaxing the full platform runner."""
    from platform_suite import run

    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Remaining diagnostic requires an empty suite directory")
    output.mkdir(parents=True, exist_ok=True)
    reference_report = output / "reference-variance.jsonl"
    os.environ["FASTSCIPY_REFERENCE_REPORT"] = str(reference_report)
    report = {
        "purpose": REMAINING_PURPOSE,
        "source_head": os.environ["FASTSCIPY_BUILD_REVISION"],
        "partial_coverage": True,
        "selected_test_files": list(REMAINING_TESTS),
        "expected_benchmark_cases": {"special": 37, "stats": 58},
        "full_407_case_validation": False,
        "upstream_requested": False,
        "runs": {},
        "areas": {},
        "validation_errors": [],
    }

    def save():
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def execute(name, arguments):
        report["runs"][name] = run([sys.executable, *arguments], output / f"{name}.log", 4096, 1200)
        save()

    xml = output / "tests.xml"
    execute("selected_tests", ["-m", "pytest", *REMAINING_TESTS, "-q", "--tb=short", f"--junitxml={xml}"])
    if xml.is_file():
        suites = [dict(suite.attrib) for suite in ET.parse(xml).getroot().iter("testsuite")]
        report["tests"] = suites
        report["test_xml"] = {"file": xml.name, **fingerprint(xml)}
        if not suites or sum(int(s.get("tests", 0)) for s in suites) == 0:
            report["validation_errors"].append("No selected tests recorded")
        if any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors")):
            report["validation_errors"].append("Selected test XML records failures/errors")
    else:
        report["validation_errors"].append("Missing selected test XML")
    for area in ("special", "stats"):
        execute(area, [str(root / "bench/run.py"), area, "--budget", "0.1", "--output-dir", str(output / "cases")])

    proofs = (
        [json.loads(line) for line in reference_report.read_text(encoding="utf-8").splitlines()]
        if reference_report.exists()
        else []
    )
    report["reference_variance"] = proofs

    def valid_proof(record):
        return (
            isinstance(record, dict)
            and record.get("policy") == "intel-accelerate-reference-variance"
            and record.get("reference_sha256") != record.get("repeated_reference_sha256")
            and isinstance(record.get("validator"), str)
            and bool(record["validator"])
            and type(record.get("reference_trials")) is int
            and 1 <= record["reference_trials"] <= 8
            and all(
                isinstance(record.get(k), str)
                and len(record[k]) == 64
                and all(c in "0123456789abcdef" for c in record[k])
                for k in ("reference_sha256", "repeated_reference_sha256", "actual_sha256")
            )
            and record in proofs
        )

    actual_files = {p.stem for p in (output / "cases").glob("*.json")}
    if actual_files != {"special", "stats"}:
        report["validation_errors"].append(f"Unexpected/missing benchmark areas: {sorted(actual_files)}")
    for area, count in report["expected_benchmark_cases"].items():
        path = output / "cases" / f"{area}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["cases"]
        if len(rows) != count:
            report["validation_errors"].append(f"{area}: expected {count} cases, got {len(rows)}")
        bad, exceptions = [], []
        for name, row in rows.items():
            if row.get("fast_path") is not True or "expected_fast_path" not in row:
                bad.append(name)
            if row.get("identical") is True:
                if row.get("validation") != "byte_identical":
                    bad.append(name)
                continue
            evidence = row.get("reference_variance", {})
            if (
                row.get("validation") == "reference_variance"
                and isinstance(evidence, dict)
                and evidence
                and all(valid_proof(record) for record in evidence.values())
            ):
                exceptions.append(name)
            else:
                bad.append(name)
        report["areas"][area] = {
            "cases": len(rows),
            "nonidentical": [name for name, row in rows.items() if row.get("identical") is not True],
            "reference_variance": exceptions,
            "invalid_or_missing_fast_path": sorted(set(bad)),
            "case_json": {"file": f"cases/{area}.json", **fingerprint(path)},
        }
        if bad:
            report["validation_errors"].append(f"{area}: unexpected comparison/fast-path failures")
    save()
    return int(bool(report["validation_errors"]) or any(v["returncode"] != 0 for v in report["runs"].values()))


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def main(targeted=False):
    output = ROOT / "artifacts" / "python-source-intel-remaining" if targeted else OUTPUT
    purpose = REMAINING_PURPOSE if targeted else PURPOSE
    timeout = 1200 if targeted else 2400
    if sys.platform != "darwin" or platform.machine().lower() != "x86_64":
        print(json.dumps({"purpose": purpose, "skipped": "Intel Mac only"}), flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = ROOT / "artifacts" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("purpose") != "diagnostics_only" or not provenance.get("wheel_reused"):
        raise RuntimeError("This diagnostic requires an explicitly reused wheel with diagnostics provenance")
    wheel_source = provenance["wheel_artifact_run_head_sha"]
    if not isinstance(wheel_source, str) or len(wheel_source) != 40:
        raise RuntimeError("Missing full reused-wheel source revision")
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{wheel_source}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode:
        # Actions checkout is shallow; fetch only the explicitly recorded source.
        git("fetch", "--no-tags", "--depth=1", "origin", wheel_source)
        git("cat-file", "-e", f"{wheel_source}^{{commit}}")
    native_paths = ("src", "Cargo.toml", "Cargo.lock")
    if (
        git("diff", wheel_source, "--", *native_paths).strip()
        or git("ls-files", "--others", "--exclude-standard", "--", *native_paths).strip()
    ):
        raise RuntimeError("Native source differs from the reused wheel; a fresh native build is required")
    installed = Path(sysconfig.get_paths()["platlib"]) / "fastscipy"
    natives = list(installed.glob("_native*.so"))
    if len(natives) != 1:
        raise RuntimeError("Expected exactly one installed Intel native extension")
    native = natives[0]
    native_info = fingerprint(native)
    wheels = list((ROOT / "prior-wheel" / "dist" / "platform").glob("*.whl"))
    if len(wheels) != 1 or [wheels[0].name] != provenance["wheel_files"]:
        raise RuntimeError("Reused-wheel artifact does not match workflow provenance")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith(f"/fastscipy/{native.name}") or name == f"fastscipy/{native.name}"
        ]
        if len(members) != 1:
            raise RuntimeError("Expected one matching native extension in reused wheel")
        digest = hashlib.sha256()
        with archive.open(members[0]) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != native_info["sha256"]:
            raise RuntimeError("Installed native extension differs from reused wheel")
    for key in ("FASTSCIPY_ANY_PLATFORM", "FASTSCIPY_ANY_VERSION"):
        os.environ.pop(key, None)
    for key, value in {
        "FASTSCIPY_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        os.environ[key] = value
    os.environ["FASTSCIPY_STRICT"] = "1"
    head = git("rev-parse", "HEAD").decode().strip()
    os.environ["FASTSCIPY_BUILD_REVISION"] = head
    source = ROOT / "python" / "fastscipy"
    package_files = {str(p.relative_to(source)): fingerprint(p) for p in sorted(source.rglob("*.py"))}
    report = {
        "purpose": purpose,
        "source_head": head,
        "python_source_dirty": bool(git("status", "--porcelain", "--", "python/fastscipy").strip()),
        "python_package_files": package_files,
        "native_source_matches_wheel_revision": wheel_source,
        "native_git_diff_empty": True,
        "installed_native_path": str(native),
        "native": native_info,
        "wheel": {"file": wheel.name, **fingerprint(wheel)},
        "wheel_provenance": provenance,
        "wheel_provenance_file": fingerprint(provenance_path),
        "memory_limit_mib": 4096,
        "timeout_seconds": timeout,
        "upstream_requested": False,
        "partial_coverage": targeted,
        "diagnostic_code_files": {
            name: fingerprint(ROOT / name)
            for name in (
                "tools/probe_python_source_platform.py",
                "tools/probe_python_remaining_platform.py",
                "tools/platform_suite.py",
                "bench/run.py",
                "bench/bench_special.py",
                "bench/bench_stats.py",
                *REMAINING_TESTS,
            )
        },
    }
    (output / "sources.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from platform_suite import run

    with tempfile.TemporaryDirectory(prefix="fastscipy-python-source-") as directory:
        overlay = Path(directory)
        package = overlay / "fastscipy"
        shutil.copytree(
            source, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.dylib", "*.pyd")
        )
        shutil.copy2(native, package / native.name)
        if fingerprint(package / native.name) != native_info:
            raise RuntimeError("Native binary changed while preparing overlay")
        for name, expected in package_files.items():
            if fingerprint(package / name) != expected:
                raise RuntimeError(f"Python source changed while preparing overlay: {name}")
        os.environ["PYTHONPATH"] = str(overlay)
        # Verify the actual import before the runner starts any tests or benches.
        bootstrap = """
import json, os, pathlib, runpy, sys
import fastscipy
from fastscipy import _native
package = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(fastscipy.__file__).resolve() == package / "__init__.py"
assert pathlib.Path(_native.__file__).resolve() == package / sys.argv[2]
assert not any(key in os.environ for key in ("FASTSCIPY_ANY_PLATFORM", "FASTSCIPY_ANY_VERSION"))
pathlib.Path(sys.argv[3]).write_text(json.dumps({
    "purpose": sys.argv[4], "package_path": fastscipy.__file__,
    "native_path": _native.__file__, "overlay_import_verified": True,
    "environment": {key: os.environ.get(key) for key in (
        "FASTSCIPY_ANY_PLATFORM", "FASTSCIPY_ANY_VERSION", "FASTSCIPY_STRICT",
        "FASTSCIPY_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "FASTSCIPY_BUILD_REVISION",
    )},
}, indent=2), encoding="utf-8")
runner = sys.argv[5]
if sys.argv[7] == "remaining":
    sys.path.insert(0, str(pathlib.Path(runner).parent))
    helpers = runpy.run_path(str(pathlib.Path(runner).with_name("probe_python_source_platform.py")))
    raise SystemExit(helpers["remaining_suite"](pathlib.Path(runner).parent.parent, pathlib.Path(sys.argv[6])))
sys.argv = [runner, "--output", sys.argv[6], "--max-rss-mb", "4096", "--timeout", sys.argv[8]]
runpy.run_path(runner, run_name="__main__")
"""
        command = [
            sys.executable,
            "-c",
            bootstrap,
            str(package),
            native.name,
            str(output / "overlay-import.json"),
            purpose,
            str(ROOT / "tools" / "platform_suite.py"),
            str(output / "suite"),
            "remaining" if targeted else "full",
            str(timeout),
        ]
        result = run(command, output / "suite.log", 4096, timeout)
    summary_path = output / "suite" / "summary.json"
    if summary_path.is_file():
        # Preserve the runner's original bytes, then label the standalone summary
        # so it cannot be mistaken for a freshly built candidate wheel report.
        shutil.copyfile(summary_path, summary_path.with_name("summary.runner.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["diagnostic_provenance"] = {
            "purpose": purpose,
            "python_source_head": head,
            "native_wheel_source": wheel_source,
            "native_sha256": native_info["sha256"],
            "source_manifest": "../sources.json",
            "unmodified_runner_summary": "summary.runner.json",
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result.update(purpose=purpose, memory_limit_mib=4096, timeout_seconds=timeout)
    (output / "monitor.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["returncode"] == 0:
        if not (output / "overlay-import.json").is_file() or not (output / "suite" / "summary.json").is_file():
            return 1
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
