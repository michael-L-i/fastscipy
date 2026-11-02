"""Check an installed wheel without changing its NumPy/SciPy environment.

This is a small ABI/dispatch canary, not a replacement for numerical suites.
Run outside the checkout's Python source directory, with version overrides unset.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sysconfig


def inspect(expect_active=()):
    import numpy as np
    import scipy
    from scipy import ndimage, special

    import fastscipy
    from fastscipy import _native

    assert not os.environ.get("FASTSCIPY_ANY_VERSION"), "Version overrides invalidate compatibility checks"
    assert not os.environ.get("FASTSCIPY_ANY_PLATFORM"), "Platform overrides invalidate compatibility checks"
    site = Path(sysconfig.get_paths()["platlib"]).resolve()
    for module in (fastscipy, _native):
        assert Path(module.__file__).resolve().is_relative_to(site), f"Not installed: {module.__file__}"

    # Test a native NumPy allocation/write through the dynamically loaded C API,
    # even when this SciPy minor is deliberately fallback-only.
    x = np.arange(65, dtype=np.float64)
    weights = np.array([1.0, 2.0, 1.0])
    expected = ndimage.correlate1d(x, weights, mode="reflect")
    actual = np.empty_like(x)
    assert _native.ndimage.corr_correlate1d(x, weights, 0, actual, 2, 0.0, 0, 0)
    assert actual.tobytes() == expected.tobytes(), "Native NumPy ABI canary disagrees with SciPy"

    originals = (ndimage.correlate1d, special.expit)
    samples = (expected, special.expit(x))
    status = fastscipy.patch(threads=2)
    try:
        for component in expect_active:
            assert status.get(component) == "on", (component, status)
        assert ndimage.correlate1d(x, weights, mode="reflect").tobytes() == samples[0].tobytes()
        assert special.expit(x).tobytes() == samples[1].tobytes()
        smokes = {}
        for component, state in status.items():
            if state != "on":
                continue
            module = __import__("fastscipy._" + component, fromlist=["smoke"])
            before = sum(v for (_, kind), v in fastscipy.stats().items() if kind == "fast")
            module.smoke()
            after = sum(v for (_, kind), v in fastscipy.stats().items() if kind == "fast")
            smokes[component] = after - before
            assert after > before, f"No acceleration in {component} smoke"
    finally:
        fastscipy.unpatch()
    assert not fastscipy.status()
    assert (ndimage.correlate1d, special.expit) == originals
    binary = Path(_native.__file__)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "fastscipy": importlib.metadata.version("fastscipy"),
        "installed_package": fastscipy.__file__,
        "native_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "native_build": _native._build_info(),
        "status": status,
        "smoke_fast_calls": smokes,
        "scope": "installed-wheel ABI and dispatch canary; not full numerical qualification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-active", nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.expect_active)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
