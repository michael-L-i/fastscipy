"""Run SciPy's test suite without and with fastscipy and compare every test's outcome.

    python tools/scipy_suite.py [pytest arguments]      (default: --pyargs scipy)
    python tools/scipy_suite.py --pyargs scipy.ndimage -n 2

Exits non-zero if a test that ran in both runs has a different outcome, or a
test runs only with fastscipy. Tests collected only without fastscipy are
reported: SciPy parametrizes 684 `scipy.special` tests on
`isinstance(f, np.ufunc)`, which the wrappers are not. An expected failure
(xfail) counts as passing, since SciPy marks flaky tests that way.
"""

import collections
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def outcomes(path):
    result = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        state = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                state = "failed"
            elif child.tag == "skipped" and child.get("type") != "pytest.xfail":
                state = "skipped"
        result[f"{case.get('classname')}::{case.get('name')}"] = state
    return result


def run(args, patched, junit):
    plugin = ["-p", "fastscipy.pytest_plugin"] if patched else []
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *plugin, f"--junitxml={junit}", *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)
    return outcomes(junit)


def main():
    args = sys.argv[1:] or ["--pyargs", "scipy"]
    with tempfile.TemporaryDirectory() as tmp:
        plain = run(args, False, f"{tmp}/plain.xml")
        fast = run(args, True, f"{tmp}/fast.xml")
    both = plain.keys() & fast.keys()
    changed = sorted(k for k in both if plain[k] != fast[k])
    # Empty parametrizations show up as a single skipped "[NOTSET]" test.
    extra = sorted(k for k in fast.keys() - plain.keys() if fast[k] != "skipped")
    missing = collections.Counter(k.split("::")[0].rsplit(".", 1)[-1] for k in plain.keys() - fast.keys())
    print(f"\n{len(both)} tests ran in both; {len(changed)} changed outcome")
    for k in changed:
        print(f"  {plain[k]} -> {fast[k]}: {k}")
    print(f"{sum(missing.values())} collected only without fastscipy: {dict(missing)}")
    for k in extra:
        print(f"  only with fastscipy ({fast[k]}): {k}")
    sys.exit(1 if changed or extra else 0)


if __name__ == "__main__":
    main()
