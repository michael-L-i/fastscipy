"""Verify a compiler-free PyPI installation and small application workflows."""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import tomllib

import psutil

ROOT = Path(__file__).resolve().parent.parent


def run(command, output, name, env, memory_mb=512, timeout=300):
    started, peak, stopped = time.monotonic(), 0, None
    with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                tree = [parent, *parent.children(recursive=True)]
                rss = 0
                for child in tree:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if rss > memory_mb * 1024**2 or time.monotonic() - started > timeout:
                    stopped = "memory" if rss > memory_mb * 1024**2 else "timeout"
                    for child in reversed(tree):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.2)
        code = process.wait()
    return {
        "command": command,
        "returncode": code,
        "seconds": time.monotonic() - started,
        "peak_rss_bytes": peak,
        "memory_limit_mb": memory_mb,
        "stopped": stopped,
    }


def main():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert os.environ["FASTSCIPY_EXPECTED_VERSION"] == version, "Checkout version differs from requested release"
    assert sys.version_info[:2] == (3, 13), "Platform pilot uses CPython 3.13"
    output = ROOT / "artifacts" / "installed-release"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Choose a fresh checkout/output directory")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "FASTSCIPY_ANY_VERSION", "FASTSCIPY_ANY_PLATFORM"):
        env.pop(key, None)
    env.update(
        FASTSCIPY_NUM_THREADS="2",
        FASTSCIPY_STRICT="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
        BLIS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        FASTSCIPY_REFERENCE_REPORT=str(output / "reference-variance.jsonl"),
    )
    report = {
        "purpose": "clean PyPI wheel installation and application pilots; separate from release certification",
        "version": version,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source_sha": env.get("GITHUB_SHA"),
        "native_threads": 2,
        "blas_threads": 1,
        "runs": {},
    }

    def execute(name, command, **limits):
        result = run(command, output, name, env, **limits)
        report["runs"][name] = result
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if result["returncode"]:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")

    with tempfile.TemporaryDirectory(prefix="fastscipy-pypi-pilot-") as temporary:
        venv = Path(temporary) / "env"
        execute("create_environment", [sys.executable, "-I", "-m", "venv", str(venv)])
        python = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        execute(
            "install_pypi",
            [
                python,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--index-url",
                "https://pypi.org/simple",
                "--no-cache-dir",
                "--only-binary=:all:",
                "--report",
                str(output / "pip-install.json"),
                f"fastscipy=={version}",
                "packaging",
            ],
            memory_mb=1024,
            timeout=600,
        )
        execute("components", [python, "-I", str(ROOT / "tools/check_release.py"), "--installed"])
        execute("image_signal", [python, "-I", str(ROOT / "tools/pilot_image_signal.py")])
        execute("sensor_statistics", [python, "-I", str(ROOT / "tools/pilot_stats.py")])
        execute("dependencies", [python, "-I", "-m", "pip", "check"])
    report["passed"] = True
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
