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
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "artifacts" / "python-source-intel"
PURPOSE = "current_python_reused_native_diagnostic_not_release_certification"


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def main():
    if sys.platform != "darwin" or platform.machine().lower() != "x86_64":
        print(json.dumps({"purpose": PURPOSE, "skipped": "Intel Mac only"}), flush=True)
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
        "purpose": PURPOSE,
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
        "timeout_seconds": 1200,
        "upstream_requested": False,
    }
    (OUTPUT / "sources.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
import json, pathlib, runpy, sys
import fastscipy
from fastscipy import _native
package = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(fastscipy.__file__).resolve() == package / "__init__.py"
assert pathlib.Path(_native.__file__).resolve() == package / sys.argv[2]
pathlib.Path(sys.argv[3]).write_text(json.dumps({
    "purpose": sys.argv[4], "package_path": fastscipy.__file__,
    "native_path": _native.__file__, "overlay_import_verified": True,
}, indent=2), encoding="utf-8")
runner = sys.argv[5]
sys.argv = [runner, "--output", sys.argv[6], "--max-rss-mb", "4096", "--timeout", "1200"]
runpy.run_path(runner, run_name="__main__")
"""
        command = [
            sys.executable,
            "-c",
            bootstrap,
            str(package),
            native.name,
            str(OUTPUT / "overlay-import.json"),
            PURPOSE,
            str(ROOT / "tools" / "platform_suite.py"),
            str(OUTPUT / "suite"),
        ]
        result = run(command, OUTPUT / "suite.log", 4096, 1200)
    summary_path = OUTPUT / "suite" / "summary.json"
    if summary_path.is_file():
        # Preserve the runner's original bytes, then label the standalone summary
        # so it cannot be mistaken for a freshly built candidate wheel report.
        shutil.copyfile(summary_path, summary_path.with_name("summary.runner.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["diagnostic_provenance"] = {
            "purpose": PURPOSE,
            "python_source_head": head,
            "native_wheel_source": wheel_source,
            "native_sha256": native_info["sha256"],
            "source_manifest": "../sources.json",
            "unmodified_runner_summary": "summary.runner.json",
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result.update(purpose=PURPOSE, memory_limit_mib=4096, timeout_seconds=1200)
    (OUTPUT / "monitor.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["returncode"] == 0:
        if not (OUTPUT / "overlay-import.json").is_file() or not (OUTPUT / "suite" / "summary.json").is_file():
            return 1
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
