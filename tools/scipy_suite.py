"""Run SciPy's test suite without and with fastscipy and compare every test's outcome.

    python tools/scipy_suite.py [pytest arguments]      (default: --pyargs scipy)
    python tools/scipy_suite.py --pyargs scipy.ndimage -n 2
    python tools/scipy_suite.py --partitioned --output-dir artifacts/upstream --pyargs scipy

Partitioned runs use a fresh interpreter per top-level SciPy package. An optional
empty output directory retains raw JUnit files and a run manifest, including if
a later partition fails; it never reuses earlier results. Incomplete pairs are
recorded and excluded from comparisons; remaining partitions still run, and
any incomplete pair makes the final exit non-zero.

Exits non-zero if a test that ran in both runs has a different outcome, or a
test runs only with fastscipy. Only the known collection differences in
SciPy's special ufunc tests are allowed: SciPy parametrizes these tests on
`isinstance(f, np.ufunc)`, which the wrappers are not. An expected failure
(xfail) counts as passing, since SciPy marks flaky tests that way.
"""

import argparse
import collections
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def ufunc_collection_test(key):
    # Partition labels distinguish executions that JUnit gives the same name.
    partition, separator, original = key.partition(" / ")
    if separator:
        key = original
    module, _, name = key.partition("::")
    # A package-local pytest root shortens JUnit module names. Only the special
    # partition supplies the provenance needed to recognize these short names.
    if (
        separator
        and partition == "scipy.special"
        and module in ("tests.test_nan_inputs", "tests.test_ufunc_signatures")
    ):
        module = "scipy.special." + module
    function, bracket, _ = name.partition("[")
    allowed = {
        "scipy.special.tests.test_nan_inputs": ("test_nan_inputs",),
        "scipy.special.tests.test_ufunc_signatures": ("test_ufunc_signatures", "test_nep50"),
    }
    return bool(bracket) and any(
        (module == expected or module.endswith("." + expected)) and function in functions
        for expected, functions in allowed.items()
    )


class IncompleteRun(RuntimeError):
    def __init__(self, message, returncode=None):
        super().__init__(message)
        self.returncode = returncode


def outcomes(path):
    result = {}
    for _, case in ET.iterparse(path, events=("end",)):
        if case.tag != "testcase":
            continue
        state = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                state = "failed"
            elif child.tag == "skipped" and child.get("type") != "pytest.xfail":
                state = "skipped"
        key = f"{case.get('classname')}::{case.get('name')}"
        if key in result:
            raise IncompleteRun(f"Duplicate JUnit test identifier: {key}")
        result[key] = state
        case.clear()
    return result


def run(args, patched, junit):
    plugin = ["-p", "fastscipy.pytest_plugin"] if patched else []
    quarantine = Path(junit).with_suffix(".quarantine.json")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "scipy_upstream_quarantine",
        *plugin,
        f"--upstream-quarantine-report={quarantine}",
        f"--junitxml={junit}",
        *args,
    ]
    env = os.environ.copy()
    tools = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = tools + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print("$", " ".join(cmd), flush=True)
    try:
        result = subprocess.run(cmd, check=False, env=env)
    except OSError as error:
        raise IncompleteRun(f"Could not launch pytest: {error}") from error
    if result.returncode not in (0, 1):
        raise IncompleteRun(f"pytest did not complete: exit {result.returncode}", result.returncode)
    try:
        results = outcomes(junit)
    except (OSError, ET.ParseError, IncompleteRun) as error:
        raise IncompleteRun(f"Invalid JUnit results: {error}", result.returncode) from error
    if not results:
        raise IncompleteRun("pytest collected no tests", result.returncode)
    try:
        quarantine_report = json.loads(quarantine.read_text(encoding="utf-8"))
        if quarantine_report["plugin"] != "scipy_upstream_quarantine" or not isinstance(
            quarantine_report["selected"], list
        ):
            raise ValueError("Invalid quarantine report")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise IncompleteRun(f"Invalid paired-quarantine report: {error}", result.returncode) from error
    return results, result.returncode


def main():
    # Leave headroom for collection and temporary arrays inside the outer RSS
    # watchdog. SciPy's large-array tests use this to skip oversized cases.
    os.environ.setdefault("SCIPY_AVAILABLE_MEM", "2048Mib")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--partitioned", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    options, args = parser.parse_known_args()
    partitioned = options.partitioned
    if options.output_dir is not None:
        options.output_dir = options.output_dir.resolve()
        if options.output_dir.exists() and any(options.output_dir.iterdir()):
            parser.error("Choose an empty output directory so results from different runs cannot mix")
        options.output_dir.mkdir(parents=True, exist_ok=True)
    args = args or ["--pyargs", "scipy"]
    batches = [args]
    if partitioned:
        import scipy

        if args[:2] != ["--pyargs", "scipy"]:
            raise ValueError("--partitioned expects --pyargs scipy, optionally followed by pytest options")
        root = Path(scipy.__file__).parent

        def is_test_file(path):
            return path.name.startswith("test_") or path.name.endswith("_test.py")

        def test_files(path):
            return (file for file in path.rglob("*.py") if is_test_file(file))

        packages = [path.name for path in sorted(root.iterdir()) if path.is_dir() and any(test_files(path))]
        root_tests = [path for path in root.glob("*.py") if is_test_file(path)]
        if not packages or root_tests:
            raise RuntimeError("Review SciPy's test layout before partitioning")
        batches = [["--pyargs", f"scipy.{package}", *args[2:]] for package in packages]
        print("Serial upstream partitions:", ", ".join(packages), flush=True)
    directory = tempfile.TemporaryDirectory() if options.output_dir is None else nullcontext(str(options.output_dir))
    with directory as tmp:
        if options.output_dir is not None:
            try:
                revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parent.parent,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = None
            metadata = {
                "python": sys.version,
                "executable": sys.executable,
                "revision": revision,
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "fastscipy")},
                "scipy_available_memory": os.environ["SCIPY_AVAILABLE_MEM"],
                "batches": batches,
                "paired_quarantine": {
                    "plugin": "scipy_upstream_quarantine",
                    "source_sha256": hashlib.sha256(
                        Path(__file__).with_name("scipy_upstream_quarantine.py").read_bytes()
                    ).hexdigest(),
                    "reproduction": "https://github.com/michael-L-i/fastscipy/actions/runs/39376117965",
                    "policy": "Visible, exactly scoped upstream-only skips in both baseline and patched runs",
                },
            }
            (Path(tmp) / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        plain, fast = {}, {}
        records, incomplete = [], []
        for index, batch in enumerate(batches):
            pair = []
            for patched in (False, True):
                junit = Path(tmp) / f"{index}-{patched}.xml"
                record = {
                    "partition": index,
                    "args": batch,
                    "patched": patched,
                    "junit": junit.name,
                    "quarantine": junit.with_suffix(".quarantine.json").name,
                    "status": "running",
                    "complete": False,
                }
                records.append(record)
                report = Path(tmp) / "partition-results.json"
                report.write_text(json.dumps(records, indent=2), encoding="utf-8")
                try:
                    result, returncode = run(batch, patched, str(junit))
                except IncompleteRun as error:
                    result = None
                    record.update(status="incomplete", returncode=error.returncode, error=str(error))
                    print(f"Incomplete partition {index}, patched={patched}: {error}", flush=True)
                else:
                    record.update(status="completed", complete=True, returncode=returncode, tests=len(result))
                record["junit_exists"] = junit.exists()
                quarantine = junit.with_suffix(".quarantine.json")
                record["quarantine_exists"] = quarantine.exists()
                if quarantine.exists():
                    try:
                        record["quarantine_report"] = json.loads(quarantine.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        record["quarantine_report_invalid"] = True
                report.write_text(json.dumps(records, indent=2), encoding="utf-8")
                pair.append(result)
            if any(result is None for result in pair):
                incomplete.append(index)
                continue
            for target, result in zip((plain, fast), pair):
                if partitioned:
                    result = {f"{batch[1]} / {key}": state for key, state in result.items()}
                target.update(result)
    both = plain.keys() & fast.keys()
    changed = sorted(k for k in both if plain[k] != fast[k])
    # Empty parametrizations show up as a single skipped "[NOTSET]" test.
    extra = sorted(
        k
        for k in fast.keys() - plain.keys()
        if not (fast[k] == "skipped" and k.endswith("[NOTSET]") and ufunc_collection_test(k))
    )
    unexpected_missing = sorted(k for k in plain.keys() - fast.keys() if not ufunc_collection_test(k))
    missing = collections.Counter(k.split("::")[0].rsplit(".", 1)[-1] for k in plain.keys() - fast.keys())
    print(f"\n{len(both)} tests ran in both; {len(changed)} changed outcome")
    print(f"{len(incomplete)} incomplete partition pairs excluded from comparison: {incomplete}")
    for k in changed:
        print(f"  {plain[k]} -> {fast[k]}: {k}")
    print(f"{sum(missing.values())} collected only without fastscipy: {dict(missing)}")
    for k in extra:
        print(f"  only with fastscipy ({fast[k]}): {k}")
    for k in unexpected_missing:
        print(f"  unexpectedly missing with fastscipy ({plain[k]}): {k}")
    sys.exit(1 if changed or extra or unexpected_missing or incomplete else 0)


if __name__ == "__main__":
    main()
