"""Native provenance detects stale artifacts without depending on Git or paths."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_build_under_test", ROOT / "tools/check_native_build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "src/nested").mkdir(parents=True)
    for name, content in {
        "Cargo.toml": b"manifest",
        "Cargo.lock": b"dependencies",
        "build.rs": b"build script",
        "src/lib.rs": b"native entry point",
        "src/nested/kernel.rs": b"native kernel",
        "src/nested/data.bin": b"\x00\xff",
    }.items():
        (root / name).write_bytes(content)
    return root


@pytest.fixture
def native(source, tmp_path):
    path = tmp_path / "extension.so"
    path.write_bytes(b"stand-in binary")
    info = {"source_schema": "1", "source_sha256": build.source_fingerprint(source)}
    return SimpleNamespace(__file__=str(path), _build_info=lambda: dict(info))


def test_fingerprint_is_path_independent_and_ignores_python_docs(source, tmp_path):
    expected = build.source_fingerprint(source)
    moved = tmp_path / "different checkout"
    shutil.copytree(source, moved)
    (moved / "python").mkdir()
    (moved / "python/wrapper.py").write_text("changed Python wrapper")
    (moved / "README.md").write_text("changed docs")
    assert not (moved / ".git").exists()
    assert build.source_fingerprint(moved) == expected


@pytest.mark.parametrize("name", ["Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs", "src/nested/data.bin"])
def test_native_input_changes_are_mismatches(source, native, name):
    (source / name).write_bytes(b"different bytes")
    report = build.inspect_build(source, native)
    assert report["status"] == "mismatch"
    assert report["expected_source_sha256"] != report["build"]["source_sha256"]


@pytest.mark.parametrize("change", ["add", "remove", "rename"])
def test_native_file_set_changes_are_mismatches(source, native, change):
    path = source / "src/nested/kernel.rs"
    if change == "add":
        (source / "src/extra.rs").write_bytes(path.read_bytes())
    elif change == "remove":
        path.unlink()
    else:
        path.rename(source / "src/nested/renamed.rs")
    assert build.inspect_build(source, native)["status"] == "mismatch"


def test_match_records_loaded_binary_identity(source, native):
    report = build.inspect_build(source, native)
    assert report["status"] == "match"
    assert report["extension"] == str(Path(native.__file__).resolve())
    assert report["binary_sha256"] == hashlib.sha256(b"stand-in binary").hexdigest()


def test_legacy_extension_is_explicitly_unverified(source, native):
    del native._build_info
    report = build.inspect_build(source, native)
    assert report["status"] == "metadata_missing"
    assert report["build"] is None
    assert report["expected_source_sha256"] is not None


def test_missing_binary_is_explicitly_unverified(source, native):
    Path(native.__file__).unlink()
    report = build.inspect_build(source, native)
    assert report["status"] == "native_unavailable"
    assert report["binary_sha256"] is None
    assert report["native_error"]


@pytest.mark.parametrize("info", [None, {}, {"source_schema": "2"}, {"source_schema": "1", "source_sha256": 1}])
def test_malformed_or_unknown_metadata_is_unverified(source, native, info):
    native._build_info = lambda: info
    assert build.inspect_build(source, native)["status"] == "metadata_invalid"


def test_wheel_without_sources_cannot_claim_a_match(tmp_path, native):
    report = build.inspect_build(tmp_path / "installed-wheel-only", native)
    assert report["status"] == "source_unavailable"
    assert report["build"]["source_sha256"]
    assert report["expected_source_sha256"] is None


def test_imported_extension_matches_current_native_source():
    report = build.inspect_build(ROOT)
    assert report["status"] == "match", f"{report['status']}: {build.REBUILD}"
    assert all(report["build"][key] for key in ("rustc", "target", "profile", "opt_level", "crate_version"))


def test_cli_fails_closed_when_source_is_unavailable(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_native_build.py"), "--source-root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "source_unavailable"
    assert "Native build check" in result.stderr


def test_platform_preflight_stops_before_numerical_work(tmp_path, monkeypatch):
    # Exercise the real entry point with legacy metadata, without invoking a
    # test runner or benchmark subprocess or requiring a second native build.
    from fastscipy import _native

    spec = importlib.util.spec_from_file_location("platform_preflight_under_test", ROOT / "tools/platform_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    monkeypatch.delattr(_native, "_build_info")
    monkeypatch.setattr(sys, "argv", ["platform_suite.py", "--tests-only", "--output", str(tmp_path / "report")])
    monkeypatch.setattr(suite, "run", lambda *args, **kwargs: pytest.fail("stale native started numerical work"))
    with pytest.raises(SystemExit) as raised:
        suite.main()
    assert raised.value.code == 2
    report = json.loads((tmp_path / "report/native-build.json").read_text())
    assert report["status"] == "metadata_missing"
