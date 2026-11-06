"""Small synthetic archive checks; these fixtures are not built release wheels."""

import importlib.util
import io
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile

import pytest


@pytest.fixture
def release():
    spec = importlib.util.spec_from_file_location(
        "check_release_under_test", Path(__file__).parents[1] / "tools/check_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_archives(directory, interpreters):
    platforms = [
        "macosx_14_0_arm64",
        "macosx_14_0_x86_64",
        "win_amd64",
        "win_arm64",
        "manylinux_2_28_aarch64",
        "manylinux_2_28_x86_64",
        "musllinux_1_2_aarch64",
        "musllinux_1_2_x86_64",
    ]
    project = {"name": "fastscipy", "version": "0.1.0a2", "requires-python": ">=3.12"}
    for python in interpreters:
        for platform in platforms:
            with zipfile.ZipFile(directory / f"fastscipy-0.1.0a2-{python}-{python}-{platform}.whl", "w") as archive:
                archive.writestr(
                    "fastscipy-0.1.0a2.dist-info/METADATA",
                    "Name: fastscipy\nVersion: 0.1.0a2\nLicense-Expression: BSD-3-Clause\n"
                    "Requires-Python: >=3.12\n\nFixture",
                )
                archive.writestr("fastscipy-0.1.0a2.dist-info/licenses/LICENSE", "fixture")
                archive.writestr("fastscipy-0.1.0a2.dist-info/licenses/scipy-LICENSE.txt", "fixture")
                archive.writestr(
                    "fastscipy/_native.fixture." + ("pyd" if platform.startswith("win_") else "so"), "fixture"
                )
    with tarfile.open(directory / "fastscipy-0.1.0a2.tar.gz", "w:gz") as archive:
        for name in (
            "Cargo.toml",
            "Cargo.lock",
            "pyproject.toml",
            "src/lib.rs",
            "python/fastscipy/__init__.py",
            "LICENSE",
            "LICENSES/scipy-LICENSE.txt",
        ):
            info = tarfile.TarInfo("fastscipy-0.1.0a2/" + name)
            archive.addfile(info, io.BytesIO())
        metadata = (
            "Name: fastscipy\nVersion: 0.1.0a2\nLicense-Expression: BSD-3-Clause\nRequires-Python: >=3.12\n\nFixture"
        ).encode()
        info = tarfile.TarInfo("fastscipy-0.1.0a2/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    return project


def test_eight_wheel_preflight_is_explicit_and_exact(release, tmp_path):
    project = make_archives(tmp_path, ["cp313"])
    with pytest.raises(AssertionError, match="Missing"):
        release.archives(tmp_path, project, complete=True)
    release.archives(tmp_path, project, complete=True, complete_python="cp313")
    next(tmp_path.glob("*win_arm64.whl")).unlink()
    with pytest.raises(AssertionError, match="Missing"):
        release.archives(tmp_path, project, complete=True, complete_python="cp313")


def test_default_complete_gate_retains_32_and_requires_21_older_wheels(release, tmp_path):
    project = make_archives(tmp_path, ["cp312", "cp313", "cp314", "cp315"])
    with pytest.raises(AssertionError, match="Missing"):
        release.archives(tmp_path, project, complete=True)
    with pytest.raises(AssertionError, match="unexpected"):
        release.archives(tmp_path, project, complete=True, complete_python="cp313")
    make_archives(tmp_path, ["cp39", "cp310", "cp311"])
    # SciPy does not provide these dependency wheels for the older interpreters.
    for suffix in ("cp39-cp39-musllinux_1_2_aarch64", "cp39-cp39-win_arm64", "cp310-cp310-win_arm64"):
        next(tmp_path.glob("*" + suffix + ".whl")).unlink()
    release.archives(tmp_path, project, complete=True)
    assert len(list(tmp_path.glob("*.whl"))) == 53
    next(tmp_path.glob("*cp312-cp312-win_arm64.whl")).unlink()
    with pytest.raises(AssertionError, match="Missing"):
        release.archives(tmp_path, project, complete=True)


@pytest.mark.parametrize(
    "event,destination,ref,accepted",
    [
        ("workflow_dispatch", "none", "refs/heads/main", True),
        ("workflow_dispatch", "pypi", "refs/heads/main", False),
        ("workflow_dispatch", "testpypi", "refs/heads/main", False),
        ("push", "none", "refs/heads/main", False),
        ("workflow_dispatch", "none", "refs/tags/v0.1.0a2", False),
        ("push", "none", "refs/tags/v0.1.0a2", False),
    ],
)
def test_validation_only_source_guard(release, monkeypatch, event, destination, ref, accepted):
    monkeypatch.setenv("VALIDATION_ONLY", "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("RELEASE_TARGET", destination)
    monkeypatch.setenv("GITHUB_REF", ref)
    if accepted:
        release.versions()
    else:
        with pytest.raises(AssertionError, match="Validation-only"):
            release.versions()


def test_source_archive_version_must_match(release, tmp_path):
    project = make_archives(tmp_path, [])
    next(tmp_path.glob("*.tar.gz")).rename(tmp_path / "fastscipy-0.1.0a1.tar.gz")
    with pytest.raises(AssertionError, match="fastscipy-0.1.0a1"):
        release.archives(tmp_path, project)


def test_dependency_metadata_includes_core_and_extra_bounds(release):
    project = {
        "name": "fastscipy",
        "version": "0.1.0a2",
        "requires-python": ">=3.9,<3.16",
        "dependencies": ["numpy>=1.19.5,<2.8", "scipy>=1.10,<1.19"],
        "optional-dependencies": {"test": ["pytest", "hypothesis; python_version >= '3.9'"]},
    }
    text = (
        "Name: fastscipy\nVersion: 0.1.0a2\nLicense-Expression: BSD-3-Clause\n"
        "Requires-Python: <3.16,>=3.9\n"
        "Requires-Dist: numpy<2.8,>=1.19.5\nRequires-Dist: scipy<1.19,>=1.10\n"
        'Requires-Dist: pytest; extra == "test"\n'
        'Requires-Dist: hypothesis; python_version >= "3.9" and extra == "test"\n'
    )
    release.check_metadata(Parser().parsestr(text), project)
    for malformed in (
        text.replace("scipy<1.19,>=1.10", "scipy==1.18.1"),
        text.replace('Requires-Dist: pytest; extra == "test"\n', ""),
        text + "Requires-Dist: unexpected-package\n",
    ):
        with pytest.raises(AssertionError, match="Dependency metadata mismatch"):
            release.check_metadata(Parser().parsestr(malformed), project)


@pytest.mark.parametrize("minor", range(9, 16))
def test_installed_gate_preserves_interpreter_scope(minor, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "release_wheel_gate", Path(__file__).parents[1] / "tools/test_release_wheel.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    commands = gate.commands(tmp_path, (3, minor))
    assert all(command[1] == "-I" for command in commands)
    if minor < 12:
        assert len(commands) == 1
        assert Path(commands[0][2]).name == "compatibility_release.py"
    else:
        assert len(commands) == 2
        assert Path(commands[0][2]).name == "check_release.py"
        assert commands[0][-1] == "--installed"
        assert Path(commands[1][2]).name == "platform_suite.py"
        assert commands[1][-1] == ("--upstream" if minor == 13 else "--tests-only")
        assert commands[1][-3:-1] == ["--max-rss-mb", "4096"]
    assert str(tmp_path) in commands[-1]
