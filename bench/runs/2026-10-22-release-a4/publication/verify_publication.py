"""Post-publication verification plan. Execute only after final release gates pass.

Uses only stdlib in the coordinator. No Rust/build tooling is invoked. Numeric
smokes run solely in newly created binary-only CP313/optional CP315 Mac ARM venvs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

RUN = "39738398406"
SOURCE = "46b33f2882f0b10a126169bed0a088705cf402a0"
VERSION = "0.1.0a4"
TARGETS = {"macos-arm64", "macos-x86_64", "windows-amd64", "windows-arm64", "manylinux-aarch64", "manylinux-x86_64", "musllinux-aarch64", "musllinux-x86_64"}


def call(command, *, cwd=None, env=None, timeout=300):
    return subprocess.check_output(command, cwd=cwd, env=env, timeout=timeout, text=True)


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New empty directory outside the checkout")
    parser.add_argument("--python313", required=True, help="Absolute native Mac ARM CPython 3.13 executable")
    parser.add_argument("--python315", help="Optional absolute native Mac ARM CPython 3.15 executable")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    # This guard runs before downloading artifacts, creating environments, or pip.
    run = json.loads(call(["gh", "run", "view", RUN, "--repo", "michael-L-i/fastscipy", "--json", "status,conclusion,headSha,headBranch,event,jobs,url"]))
    save(output / "release-run.json", run)
    jobs = run["jobs"]
    wheels = [j for j in jobs if j["name"].startswith("Wheel /")]
    required = {"Source distribution and version guards", "Verify complete release artifact set", "pypi"}
    by_name = {j["name"]: j for j in jobs}
    if not (run["headSha"] == SOURCE and run["headBranch"] == f"v{VERSION}" and run["event"] == "push" and run["status"] == "completed" and run["conclusion"] == "success" and len(wheels) == 32 and all(j["conclusion"] == "success" for j in wheels) and all(by_name.get(n, {}).get("conclusion") == "success" for n in required)):
        raise RuntimeError("Exact tagged source, all 32 wheel jobs, complete-set gate and PyPI publisher must have succeeded")
    artifact_dir = output / "run-artifacts"
    call(["gh", "run", "download", RUN, "--repo", "michael-L-i/fastscipy", "--pattern", "release-dist-*", "--dir", str(artifact_dir)], timeout=600)
    files = [p for p in artifact_dir.rglob("*") if p.is_file()]
    manifest = {}
    slots = set()
    for path in files:
        if path.name in manifest:
            raise RuntimeError(f"Duplicate artifact filename: {path.name}")
        manifest[path.name] = fingerprint(path)
        if path.name == f"fastscipy-{VERSION}.tar.gz":
            continue
        prefix = f"fastscipy-{VERSION}-"
        if not path.name.startswith(prefix) or not path.name.endswith(".whl"):
            raise RuntimeError(f"Unexpected distribution {path.name}")
        py, abi, platforms = path.name[len(prefix):-4].split("-")
        if py not in {"cp312", "cp313", "cp314", "cp315"} or abi != py:
            raise RuntimeError(f"Unexpected interpreter/ABI {path.name}")
        families = set()
        for tag in platforms.split("."):
            if tag.startswith("macosx_"):
                target = "macos-" + ("arm64" if tag.endswith("_arm64") else "x86_64")
            elif tag.startswith("win_"):
                target = "windows-" + tag[4:]
            else:
                family = "musllinux" if tag.startswith("musllinux_") else "manylinux"
                target = family + "-" + ("aarch64" if tag.endswith("_aarch64") else "x86_64")
            families.add(target)
        if len(families) != 1:
            raise RuntimeError(f"Multiple target families in {path.name}")
        slot = (py, next(iter(families)))
        if slot in slots:
            raise RuntimeError(f"Duplicate wheel slot {slot}")
        slots.add(slot)
    expected = {(py, target) for py in ("cp312", "cp313", "cp314", "cp315") for target in TARGETS}
    if len(manifest) != 33 or slots != expected or f"fastscipy-{VERSION}.tar.gz" not in manifest:
        raise RuntimeError("Expected exact 32-wheel plus one-sdist artifact set")
    save(output / "artifact-manifest.json", manifest)
    with urlopen(f"https://pypi.org/pypi/fastscipy/{VERSION}/json", timeout=30) as response:
        published = json.load(response)
    save(output / "pypi-release.json", published)
    remote = {entry["filename"]: {"size": entry["size"], "sha256": entry["digests"]["sha256"]} for entry in published["urls"]}
    if published["info"]["version"] != VERSION or len(published["urls"]) != 33 or remote != manifest or any(e["yanked"] for e in published["urls"]):
        raise RuntimeError("PyPI filenames/sizes/SHA256 do not exactly match the 33 validated run artifacts")
    save(output / "verification.json", {"run": RUN, "source": SOURCE, "version": VERSION, "published_distribution_hashes_match": True, "distribution_count": 33})
    print(json.dumps({"verified": True, "output": str(output), "distribution_count": 33}))

if __name__ == "__main__":
    main()
