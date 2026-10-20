"""Validate release versions, archives, and a freshly installed wheel."""

import argparse
import ast
from email.parser import BytesParser
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import sysconfig
import tarfile
import tomllib
import zipfile

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parent.parent


def versions():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())["package"]
    locked = next(p["version"] for p in lock if p["name"] == "fastscipy")
    tree = ast.parse((ROOT / "python/fastscipy/__init__.py").read_text())
    declared = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
    )
    assert Version(version) == Version(cargo) == Version(locked) == Version(declared), "Version mismatch"
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/"):
        assert ref == f"refs/tags/v{version}", f"Tag must be v{version}"
    if os.environ.get("RELEASE_TARGET") == "pypi":
        assert ref == f"refs/tags/v{version}", "PyPI publishing requires the matching version tag"
    return project


def archives(directory, project, complete=False):
    found = list(directory.glob("*.whl")) + list(directory.glob("*.tar.gz"))
    assert found, f"No distributions in {directory}"
    targets = set()
    for path in found:
        if path.suffix == ".whl":
            name, version, _, tags = parse_wheel_filename(path.name)
            assert name == project["name"] and version == Version(project["version"]), path.name
            allowed = re.compile(
                r"(?:macosx_\d+_\d+_(?:arm64|x86_64)|"
                r"(?:manylinux_2_\d+|manylinux2014|musllinux_1_2)_(?:aarch64|x86_64)|win_(?:amd64|arm64))"
            )
            assert all(allowed.fullmatch(tag.platform) for tag in tags), path.name
            assert all(tag.interpreter in {"cp312", "cp313", "cp314", "cp315"} for tag in tags), path.name
            assert all(tag.abi == tag.interpreter for tag in tags), path.name
            for tag in tags:
                if tag.platform.startswith("macosx_"):
                    target = "macos-" + ("arm64" if tag.platform.endswith("_arm64") else "x86_64")
                elif tag.platform.startswith("win_"):
                    target = "windows-" + tag.platform.removeprefix("win_")
                else:
                    family = "musllinux" if tag.platform.startswith("musllinux") else "manylinux"
                    target = family + "-" + ("aarch64" if tag.platform.endswith("_aarch64") else "x86_64")
                targets.add((tag.interpreter, target))
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                meta = BytesParser().parsebytes(
                    archive.read(next(n for n in names if n.endswith(".dist-info/METADATA")))
                )
                assert meta["Name"] == project["name"] and meta["Version"] == project["version"]
                assert meta["License-Expression"] == "BSD-3-Clause"
                assert SpecifierSet(meta["Requires-Python"]) == SpecifierSet(project["requires-python"])
                for license in ("LICENSE", "scipy-LICENSE.txt"):
                    assert any(n.endswith("/" + license) for n in names), f"Missing {license}"
                assert any(n.startswith("fastscipy/_native.") and n.endswith((".so", ".pyd")) for n in names)
                assert not any(n.startswith(("bench/", "examples/", "tests/")) for n in names)
                body = meta.get_payload()
                assert "](docs/" not in body and "](examples/" not in body, "Relative PyPI README links"
        else:
            with tarfile.open(path) as archive:
                names = {n.split("/", 1)[-1] for n in archive.getnames()}
                for required in (
                    "Cargo.toml",
                    "Cargo.lock",
                    "pyproject.toml",
                    "src/lib.rs",
                    "python/fastscipy/__init__.py",
                    "LICENSE",
                    "LICENSES/scipy-LICENSE.txt",
                ):
                    assert required in names, f"Missing {required}"
                assert not any(n.startswith(("bench/", "examples/", "docs/")) for n in names)
        print("Validated", path.name)
    if complete:
        platforms = {
            "macos-arm64",
            "macos-x86_64",
            "windows-amd64",
            "windows-arm64",
            "manylinux-aarch64",
            "manylinux-x86_64",
            "musllinux-aarch64",
            "musllinux-x86_64",
        }
        expected = {(python, target) for python in ("cp312", "cp313", "cp314", "cp315") for target in platforms}
        assert targets == expected, f"Missing {expected - targets}; unexpected {targets - expected}"
        assert len(list(directory.glob("*.whl"))) == 32, "Expected exactly 32 wheels"
        assert len(list(directory.glob("*.tar.gz"))) == 1, "Expected exactly one source distribution"


def installed(project):
    import fastscipy
    from fastscipy import _native

    assert not os.environ.get("FASTSCIPY_ANY_PLATFORM"), "Release checks must use supported platforms without overrides"
    assert not os.environ.get("FASTSCIPY_ANY_VERSION"), "Release checks must use supported versions without overrides"
    assert platform.system() in {"Darwin", "Linux", "Windows"}, platform.platform()
    assert platform.machine().lower() in {"arm64", "aarch64", "amd64", "x86_64"}, platform.machine()
    site = Path(sysconfig.get_paths()["platlib"]).resolve()
    assert Path(fastscipy.__file__).resolve().is_relative_to(site), "Imported checkout instead of wheel"
    assert Path(_native.__file__).resolve().is_relative_to(site), "Native extension outside installed wheel"
    assert fastscipy.__version__ == importlib.metadata.version("fastscipy") == project["version"]
    status = fastscipy.patch(threads=2)
    assert all(value == "on" for value in status.values()), status
    # Exercise every component on small inputs, including the installed Rust code.
    for component in status:
        before = sum(v for (_, kind), v in fastscipy.stats().items() if kind == "fast")
        importlib.import_module("fastscipy._" + component).smoke()
        after = sum(v for (_, kind), v in fastscipy.stats().items() if kind == "fast")
        assert after > before, f"No fast path in {component} smoke check"
    assert sum(v for (_, kind), v in fastscipy.stats().items() if kind == "fast") > 0
    fastscipy.unpatch()
    assert not fastscipy.status()
    print("Installed wheel:", fastscipy.__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--complete", action="store_true", help="Require all eight platforms and four Python versions")
    args = parser.parse_args()
    project = versions()
    if args.dist:
        archives(args.dist, project, args.complete)
    if args.installed:
        installed(project)
    print("Release version:", project["version"])


if __name__ == "__main__":
    main()
