"""Run SciPy's test suite without and with fastscipy and compare every test's outcome.

    python tools/scipy_suite.py [pytest arguments]      (default: --pyargs scipy)
    python tools/scipy_suite.py --pyargs scipy.ndimage -n 2
    python tools/scipy_suite.py --partitioned --output-dir artifacts/upstream --pyargs scipy

Partitioned runs use a fresh interpreter per top-level SciPy package. An optional
empty output directory retains raw JUnit files and a run manifest, including if
a later partition fails; it never reuses earlier results.

Exits non-zero if a test that ran in both runs has a different outcome, or a
test runs only with fastscipy. Only the known collection differences in
SciPy's special ufunc tests are allowed: SciPy parametrizes these tests on
`isinstance(f, np.ufunc)`, which the wrappers are not. An expected failure
(xfail) counts as passing, since SciPy marks flaky tests that way.
"""

import argparse
import collections
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def ufunc_collection_test(key):
    # Partition labels distinguish executions that JUnit gives the same name.
    key = key.split(" / ", 1)[-1]
    module, _, name = key.partition("::")
    function, bracket, _ = name.partition("[")
    allowed = {
        "scipy.special.tests.test_nan_inputs": ("test_nan_inputs",),
        "scipy.special.tests.test_ufunc_signatures": ("test_ufunc_signatures", "test_nep50"),
    }
    return bool(bracket) and any(
        (module == expected or module.endswith("." + expected)) and function in functions
        for expected, functions in allowed.items()
    )


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
            raise RuntimeError(f"Duplicate JUnit test identifier: {key}")
        result[key] = state
        case.clear()
    return result


def run(args, patched, junit):
    plugin = ["-p", "fastscipy.pytest_plugin"] if patched else []
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *plugin, f"--junitxml={junit}", *args]
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pytest did not complete: exit {result.returncode}")
    results = outcomes(junit)
    if not results:
        raise RuntimeError("pytest collected no tests")
    return results


def main():
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
                "batches": batches,
            }
            (Path(tmp) / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        plain, fast = {}, {}
        for index, batch in enumerate(batches):
            for target, patched in ((plain, False), (fast, True)):
                result = run(batch, patched, f"{tmp}/{index}-{patched}.xml")
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
    for k in changed:
        print(f"  {plain[k]} -> {fast[k]}: {k}")
    print(f"{sum(missing.values())} collected only without fastscipy: {dict(missing)}")
    for k in extra:
        print(f"  only with fastscipy ({fast[k]}): {k}")
    for k in unexpected_missing:
        print(f"  unexpectedly missing with fastscipy ({plain[k]}): {k}")
    sys.exit(1 if changed or extra or unexpected_missing else 0)


if __name__ == "__main__":
    main()
