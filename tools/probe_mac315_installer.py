"""Remote-only official CPython 3.15 installer smoke; no Rust or project tests.

Match cibuildwheel 4.3.1's package and choices XML. A monitored Python worker
owns the sudo installer child; inaccessible installer RSS remains explicit.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "artifacts" / "mac315-installer"
PURPOSE = "official_mac315_installer_monitor_smoke_not_release_validation"
URL = "https://www.python.org/ftp/python/3.15.0/python-3.15.0-macos11.pkg"
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.15/bin/python3.15")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def worker():
    assert importlib.metadata.version("cibuildwheel") == "4.3.1"
    choices = Path(str(importlib.resources.files("cibuildwheel") / "resources" / "free-threaded-enable-315.xml"))
    assert choices.is_file()
    package = OUTPUT / "python-3.15.0-macos11.pkg"
    total = 0
    with urlopen(URL, timeout=30) as response, package.open("wb") as stream:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            total += len(chunk)
            if total > 256 * 1024**2:
                raise RuntimeError("Unexpectedly large official Python installer")
            stream.write(chunk)
    command = ["sudo", "installer", "-pkg", str(package), "-applyChoiceChangesXML", str(choices), "-target", "/"]
    save(
        "installer-inputs.json",
        {
            "purpose": PURPOSE,
            "url": URL,
            "package": fingerprint(package),
            "cibuildwheel": "4.3.1",
            "choices_path": str(choices),
            "choices": fingerprint(choices),
            "command": command,
            "original_failure_run": "39464216282",
            "match": (
                "Same official package, cibuildwheel choices XML, and sudo installer arguments "
                "as both failed Mac315 jobs"
            ),
        },
    )
    print(json.dumps({"installer_command": command}), flush=True)
    subprocess.run(command, check=True)
    # The choices XML enables an additional free-threaded runtime. Verify the
    # ordinary framework executable used by cp315, not the optional cp315t.
    verify = """
import json, platform, sys, sysconfig
print(json.dumps({"version": list(sys.version_info[:3]), "python": sys.version,
    "machine": platform.machine().lower(), "executable": sys.executable,
    "abiflags": sys.abiflags, "SOABI": sysconfig.get_config_var("SOABI"),
    "Py_GIL_DISABLED": sysconfig.get_config_var("Py_GIL_DISABLED")}))
"""
    installed = json.loads(subprocess.check_output([str(PYTHON), "-I", "-c", verify], text=True, timeout=30))
    assert installed["version"] == [3, 15, 0], installed
    assert installed["machine"] == os.environ["EXPECTED_ARCH"], installed
    assert installed["Py_GIL_DISABLED"] in (None, 0) and "t" not in installed["abiflags"], installed
    assert installed["SOABI"].startswith("cpython-315-"), installed
    installed.update(purpose=PURPOSE, native_arch_verified=True, ordinary_non_free_threaded_ABI_verified=True)
    save("installed-python.json", installed)
    print(json.dumps(installed, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "darwin" or os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("This privileged installer probe may run only on remote GitHub macOS runners")
    expected = os.environ["EXPECTED_ARCH"]
    assert expected in ("arm64", "x86_64") and platform.machine().lower() == expected
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker()

    from platform_suite import run

    report = {
        "purpose": PURPOSE,
        "source_head": os.environ["GITHUB_SHA"],
        "run_id": os.environ["GITHUB_RUN_ID"],
        "host_python": sys.version,
        "host_platform": platform.platform(),
        "expected_arch": expected,
        "memory_limit_mib": 1024,
        "timeout_seconds": 600,
        "no_rust_build": True,
        "no_project_tests": True,
        "script": fingerprint(Path(__file__)),
        "monitor_source": fingerprint(ROOT / "tools/platform_suite.py"),
    }
    save("provenance.json", report)
    result = run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        OUTPUT / "installer.log",
        1024,
        600,
        allow_privileged_installer=True,
    )
    report["monitor"] = result
    report["installed_python_verified"] = (OUTPUT / "installed-python.json").is_file()
    save("report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return result["returncode"] or int(not report["installed_python_verified"])


if __name__ == "__main__":
    raise SystemExit(main())
