"""Check that the imported extension was built from the requested native sources.

Run before development tests/benchmarks, including in a source archive without
Git. This checks source identity, not binary reproducibility or compatibility
with an arbitrary compiler. Installed wheels can report metadata without having
their Rust sources available; they cannot claim a source match in that case.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REBUILD = "Rebuild with CARGO_BUILD_JOBS=1 maturin develop --profile dev-release in the intended source tree."


def source_fingerprint(root):
    """SHA-256 of native build inputs, using build.rs's versioned framing."""
    root = Path(root)
    files = [root / name for name in ("Cargo.toml", "Cargo.lock", "build.rs")]
    if not (root / "src/lib.rs").is_file():
        raise FileNotFoundError(f"Native sources unavailable: {root / 'src/lib.rs'}")
    files.extend(path for path in (root / "src").rglob("*") if path.is_file())
    entries = sorted((path.relative_to(root).as_posix(), path) for path in files)
    digest = hashlib.sha256(b"fastscipy-native-source-v1\0")
    for name, path in entries:
        encoded = name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def inspect_build(root, native=None):
    """Return an explicit status and artifact metadata; never imply unknown=match."""
    report = {
        "extension": None,
        "binary_sha256": None,
        "source_root": str(Path(root).resolve()),
        "build": None,
        "expected_source_sha256": None,
    }
    try:
        report["expected_source_sha256"] = source_fingerprint(root)
    except OSError as error:
        report["source_error"] = str(error)
    try:
        if native is None:
            from fastscipy import _native as native

        path = Path(native.__file__).resolve()
        report["extension"] = str(path)
        report["binary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except (ImportError, OSError) as error:
        report.update(status="native_unavailable", native_error=str(error))
        return report
    getter = getattr(native, "_build_info", None)
    if getter is None:
        report["status"] = "metadata_missing"
        return report
    if not callable(getter):
        report["status"] = "metadata_invalid"
        return report
    info = getter()
    report["build"] = info
    if (
        not isinstance(info, dict)
        or info.get("source_schema") != "1"
        or not isinstance(info.get("source_sha256"), str)
        or re.fullmatch("[0-9a-f]{64}", info["source_sha256"]) is None
    ):
        report["status"] = "metadata_invalid"
    elif report["expected_source_sha256"] is None:
        report["status"] = "source_unavailable"
    elif info["source_sha256"] == report["expected_source_sha256"]:
        report["status"] = "match"
    else:
        report["status"] = "mismatch"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = inspect_build(args.source_root)
    print(json.dumps(report, indent=2))
    if report["status"] != "match":
        print(f"Native build check: {report['status']}. {REBUILD}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
