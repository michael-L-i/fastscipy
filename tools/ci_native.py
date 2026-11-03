"""Cache native compilation for development CI; always import current Python sources.

The cached wheel supplies metadata and a runner-local extension, not a release
artifact. Release workflows deliberately do not use this tool.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
from zipfile import ZipFile

from check_native_build import inspect_build, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".ci-native-cache"


def identity(profile):
    recipe = hashlib.sha256()
    for name in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "LICENSES/scipy-LICENSE.txt",
        "tools/ci_native.py",
        ".github/actions/native/action.yml",
        ".github/workflows/ci.yml",
        ".cargo/config",
        ".cargo/config.toml",
        "rust-toolchain",
        "rust-toolchain.toml",
    ):
        path = ROOT / name
        content = hashlib.sha256(path.read_bytes()).digest() if path.exists() else b"absent"
        recipe.update(name.encode() + b"\0" + content)
    return {
        "native_source": source_fingerprint(ROOT),
        "recipe": recipe.hexdigest(),
        "profile": profile,
        "python": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "platform": sysconfig.get_platform(),
        "machine": platform.machine(),
        "image": [os.environ.get(k, "") for k in ("ImageOS", "ImageVersion")],
        "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
        "maturin": importlib.metadata.version("maturin"),
        "numpy": importlib.metadata.version("numpy"),
        "build_env": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(("CARGO_PROFILE_", "CARGO_TARGET_", "PYO3_"))
            or k
            in (
                "RUSTFLAGS",
                "RUSTC",
                "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER",
                "CARGO_BUILD_RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_BUILD_TARGET",
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "MACOSX_DEPLOYMENT_TARGET",
                "SDKROOT",
            )
        },
    }


def install(profile, reuse):
    expected = identity(profile)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    if not reuse:
        from platform_suite import run

        shutil.rmtree(CACHE, ignore_errors=True)
        CACHE.mkdir()
        result = run(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--locked",
                "--profile",
                profile,
                "--auditwheel",
                "skip",
                "--out",
                str(CACHE),
                "--interpreter",
                sys.executable,
            ],
            artifacts / "build.log",
            6144,
            10800,
        )
        (artifacts / "build.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if result["returncode"]:
            raise RuntimeError(f"Native build failed: {result}")
        wheel = next(CACHE.glob("*.whl"))
        metadata = dict(
            identity=expected,
            wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
            build_commit=os.environ.get("GITHUB_SHA"),
        )
        (CACHE / "identity.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata = json.loads((CACHE / "identity.json").read_text(encoding="utf-8"))
    if metadata["identity"] != expected:
        raise RuntimeError("Native cache identity mismatch; remove this cache and rebuild")
    wheels = list(CACHE.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Native cache must contain exactly one wheel")
    if hashlib.sha256(wheels[0].read_bytes()).hexdigest() != metadata["wheel_sha256"]:
        raise RuntimeError("Native cache wheel checksum mismatch; remove this cache and rebuild")
    # Install metadata normally; PYTHONPATH below selects checkout code.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheels[0])], check=True
    )
    package = ROOT / "python/fastscipy"
    with ZipFile(wheels[0]) as wheel:
        natives = [n for n in wheel.namelist() if n.startswith("fastscipy/_native.") and n.endswith((".so", ".pyd"))]
        if len(natives) != 1:
            raise RuntimeError("Expected one native extension in the development wheel")
        for old in package.glob("_native.*"):
            if old.suffix in (".so", ".pyd"):
                old.unlink()
        (package / Path(natives[0]).name).write_bytes(wheel.read(natives[0]))
    sys.path.insert(0, str(ROOT / "python"))
    import fastscipy

    if Path(fastscipy.__file__).resolve() != package / "__init__.py":
        raise RuntimeError("CI must import Python code from the current checkout")
    report = inspect_build(ROOT)
    report.update(
        native_cache_hit=reuse,
        native_build_commit=metadata["build_commit"],
        cached_wheel_sha256=metadata["wheel_sha256"],
        cache_identity=expected,
        python_source=str(package),
        checkout_sha=os.environ.get("GITHUB_SHA"),
        validation="development_source",
    )
    (artifacts / "native-cache.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "match":
        raise RuntimeError(f"Native provenance check failed: {report['status']}")
    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as stream:
        stream.write(f"PYTHONPATH={ROOT / 'python'}\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("key", "install"))
    parser.add_argument("--profile", choices=("dev-release", "release"), required=True)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.command == "key":
        digest = hashlib.sha256(json.dumps(identity(args.profile), sort_keys=True).encode()).hexdigest()
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(f"key=native-v1-{digest}\n")
    else:
        install(args.profile, args.reuse)


if __name__ == "__main__":
    main()
