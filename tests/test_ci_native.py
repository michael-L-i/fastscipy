"""The compilation cache must ignore Python edits but reject changed build inputs."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cache_tool(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = importlib.import_module("ci_native")
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "CACHE", tmp_path / "cache")
    for name in (
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/lib.rs",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "LICENSES/scipy-LICENSE.txt",
        "tools/ci_native.py",
        ".github/actions/native/action.yml",
        ".github/workflows/ci.yml",
        "python/fastscipy/__init__.py",
    ):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    monkeypatch.setattr(tool.subprocess, "check_output", lambda *a, **kw: "rustc test")
    monkeypatch.setattr(tool.importlib.metadata, "version", lambda name: "test")
    return tool


def test_python_changes_reuse_native_identity(cache_tool):
    before = cache_tool.identity("dev-release")
    package = cache_tool.ROOT / "python/fastscipy"
    (package / "__init__.py").write_text("new Python implementation")
    (package / "added.py").write_text("new module")
    assert cache_tool.identity("dev-release") == before


@pytest.mark.parametrize(
    "name",
    [
        "src/lib.rs",
        "src/new.rs",
        "Cargo.lock",
        "pyproject.toml",
        ".github/actions/native/action.yml",
        ".github/workflows/ci.yml",
        ".cargo/config.toml",
    ],
)
def test_build_inputs_invalidate_cache(cache_tool, name):
    before = cache_tool.identity("dev-release")
    path = cache_tool.ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("changed")
    assert cache_tool.identity("dev-release") != before


def test_environment_and_profile_invalidate_cache(cache_tool, monkeypatch):
    before = cache_tool.identity("dev-release")
    assert cache_tool.identity("release") != before
    for name in ("RUSTFLAGS", "ImageVersion", "CARGO_PROFILE_DEV_RELEASE_CODEGEN_UNITS"):
        with monkeypatch.context() as context:
            context.setenv(name, "different")
            assert cache_tool.identity("dev-release") != before
    monkeypatch.setattr(cache_tool.sysconfig, "get_config_var", lambda key: "different ABI")
    assert cache_tool.identity("dev-release") != before


@pytest.mark.parametrize("corruption", ["identity", "checksum", "missing-wheel"])
def test_invalid_cache_fails_before_install(cache_tool, monkeypatch, corruption):
    cache_tool.CACHE.mkdir()
    expected = cache_tool.identity("dev-release")
    metadata = dict(identity=expected, wheel_sha256="invalid", build_commit="test")
    if corruption == "identity":
        metadata["identity"] = {}
    if corruption != "missing-wheel":
        (cache_tool.CACHE / "test.whl").write_bytes(b"corrupt")
    (cache_tool.CACHE / "identity.json").write_text(json.dumps(metadata))

    def forbidden(*args, **kwargs):
        pytest.fail("must reject cache before installing anything")

    monkeypatch.setattr(cache_tool.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="cache"):
        cache_tool.install("dev-release", reuse=True)
