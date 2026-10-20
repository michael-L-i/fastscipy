"""Exercise the upstream runner with subprocesses mocked; no SciPy tests run."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def suite():
    spec = importlib.util.spec_from_file_location(
        "scipy_suite_under_test", Path(__file__).parents[1] / "tools/scipy_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [None, "crash", "exit2", "malformed", "duplicate", "empty"])
def test_incomplete_middle_partition_keeps_remaining_coverage(suite, monkeypatch, tmp_path, capsys, failure):
    scipy_root = tmp_path / "scipy"
    for name in ("a", "b", "c"):
        package = scipy_root / name
        package.mkdir(parents=True)
        (package / "test_stub.py").touch()
    monkeypatch.setitem(sys.modules, "scipy", SimpleNamespace(__file__=str(scipy_root / "__init__.py")))
    monkeypatch.setattr(suite.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(suite.subprocess, "check_output", lambda *args, **kwargs: "test-revision")
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys, "argv", ["scipy_suite.py", "--partitioned", "--output-dir", str(output), "--pyargs", "scipy"]
    )
    calls = []

    def subprocess_run(cmd, check, env):
        assert "scipy_upstream_quarantine" in cmd
        assert env["PYTHONPATH"].split(suite.os.pathsep)[0] == str(Path(suite.__file__).parent)
        quarantine = Path(next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--upstream-quarantine-report=")))
        quarantine.write_text(json.dumps({"plugin": "scipy_upstream_quarantine", "selected": [], "active": False}))
        junit = Path(next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--junitxml=")))
        index = int(junit.stem.split("-")[0])
        patched = "fastscipy.pytest_plugin" in cmd
        calls.append((index, patched))
        # Evidence must already identify an active run if an outer watchdog
        # kills the harness before the child returns.
        records = json.loads((output / "partition-results.json").read_text())
        assert records[-1]["status"] == "running"
        assert records[-1]["complete"] is False
        xml = '<testsuite><testcase classname="tests.example" name="test_ok"/></testsuite>'
        returncode = 0
        if index == 1 and not patched:
            if failure in ("crash", "exit2"):
                # Even plausible partial XML cannot make a crashed run complete.
                returncode = -1073741819 if failure == "crash" else 2
            elif failure == "malformed":
                xml = "<testsuite><testcase"
            elif failure == "duplicate":
                xml = xml.replace("</testsuite>", '<testcase classname="tests.example" name="test_ok"/></testsuite>')
            elif failure == "empty":
                xml = "<testsuite/>"
        elif index == 1 and failure:
            xml = xml.replace("test_ok", "test_only_patched")
        junit.write_text(xml)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(suite.subprocess, "run", subprocess_run)
    with pytest.raises(SystemExit) as exit_info:
        suite.main()
    assert exit_info.value.code == (1 if failure else 0)
    assert calls == [(index, patched) for index in range(3) for patched in (False, True)]
    records = json.loads((output / "partition-results.json").read_text())
    assert len(records) == 6
    assert records[2]["complete"] is (not bool(failure))
    assert records[2]["status"] == ("incomplete" if failure else "completed")
    assert records[2]["junit_exists"] is True
    assert ("error" in records[2]) is bool(failure)
    assert all(record["complete"] for i, record in enumerate(records) if i != 2)
    assert (output / records[2]["junit"]).exists()
    assert json.loads((output / "manifest.json").read_text())["revision"] == "test-revision"
    text = capsys.readouterr().out
    assert f"{2 if failure else 3} tests ran in both; 0 changed outcome" in text
    assert (
        f"{1 if failure else 0} incomplete partition pairs excluded from comparison: {[1] if failure else []}" in text
    )
    assert "only with fastscipy" not in text


def test_pytest_test_failure_is_complete(suite, monkeypatch, tmp_path):
    junit = tmp_path / "failure.xml"
    junit.write_text('<testsuite><testcase classname="tests.example" name="test_bad"><failure/></testcase></testsuite>')

    def subprocess_run(cmd, **kwargs):
        quarantine = Path(next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--upstream-quarantine-report=")))
        quarantine.write_text(json.dumps({"plugin": "scipy_upstream_quarantine", "selected": [], "active": False}))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(suite.subprocess, "run", subprocess_run)
    results, returncode = suite.run(["--pyargs", "scipy.example"], False, str(junit))
    assert returncode == 1
    assert results == {"tests.example::test_bad": "failed"}


@pytest.fixture
def quarantine():
    spec = importlib.util.spec_from_file_location(
        "scipy_quarantine_under_test", Path(__file__).parents[1] / "tools/scipy_upstream_quarantine.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observed_scope(quarantine, source):
    return {
        "platform": "win32",
        "machine": "arm64",
        "implementation": "cpython",
        "python_platform": "win-arm64",
        "python": [3, 13, 16],
        "numpy": "2.5.4",
        "scipy": "1.18.1",
        "test_source": str(source),
        "test_source_sha256": quarantine.TEST_SOURCE_SHA256,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("platform", "linux"),
        ("machine", "amd64"),
        ("implementation", "pypy"),
        ("python_platform", "win-amd64"),
        ("python", [3, 13, 17]),
        ("numpy", "2.5.5"),
        ("scipy", "1.18.2"),
        ("test_source_sha256", "changed-source"),
    ],
)
def test_quarantine_refuses_unproven_scope(quarantine, tmp_path, field, value):
    observed = observed_scope(quarantine, tmp_path / "test_basic.py")
    assert quarantine.applicable(observed)
    observed[field] = value
    assert not quarantine.applicable(observed)


def test_quarantine_actual_source_hash_refusal(quarantine, monkeypatch, tmp_path):
    source = tmp_path / "linalg/tests/test_basic.py"
    source.parent.mkdir(parents=True)
    source.write_text("original upstream source")
    monkeypatch.setattr(quarantine.sys, "platform", "win32")
    monkeypatch.setattr(quarantine.sys, "version_info", (3, 13, 16))
    monkeypatch.setattr(quarantine.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(quarantine.sysconfig, "get_platform", lambda: "win-arm64")
    monkeypatch.setattr(
        quarantine.importlib.metadata, "version", lambda name: {"numpy": "2.5.4", "scipy": "1.18.1"}[name]
    )
    monkeypatch.setattr(
        quarantine.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    monkeypatch.setattr(quarantine, "TEST_SOURCE_SHA256", quarantine.hashlib.sha256(source.read_bytes()).hexdigest())
    assert quarantine.applicable(quarantine.scope())
    source.write_text("changed upstream source")
    assert not quarantine.applicable(quarantine.scope())


def test_quarantine_visibly_skips_exact_eight_of_416(quarantine, monkeypatch, tmp_path):
    import itertools
    import numpy as np

    source = tmp_path / "test_basic.py"
    observed = observed_scope(quarantine, source)
    monkeypatch.setattr(quarantine, "scope", lambda: observed)
    report = tmp_path / "quarantine.json"
    config = SimpleNamespace(getoption=lambda name: str(report))
    quarantine.pytest_configure(config)
    items = []
    structures = [
        "diagonal",
        "tridiagonal",
        "banded",
        "lower triangular",
        "upper triangular",
        "pos",
        "positive definite",
        "symmetric",
        "hermitian",
        "general",
        "sym",
        "her",
        "gen",
    ]
    cls = type("TestSolve", (), {})
    for fortran, overwrite, transposed, rhs, structure, dtype in itertools.product(
        (True, False), (True, False), (True, False), ((), (5,)), structures, (np.float64, np.complex128)
    ):
        markers = []
        params = dict(
            fortran=fortran, overwrite=overwrite, transposed=transposed, nrhs=rhs, assume_a=structure, dtype=dtype
        )
        nodeid = (
            "test_basic.py::TestSolve::test_structure_detection"
            f"[{fortran}-{overwrite}-{transposed}-nrhs{int(bool(rhs))}-{structure}-{dtype.__name__}]"
        )
        items.append(
            SimpleNamespace(
                path=source,
                cls=cls,
                originalname="test_structure_detection",
                nodeid=nodeid,
                callspec=SimpleNamespace(params=params),
                add_marker=markers.append,
                markers=markers,
            )
        )
    quarantine.pytest_collection_modifyitems(config, items)
    assert len(items) == 416
    marked = [item for item in items if item.markers]
    assert len(marked) == 8
    assert all(
        item.markers[0].name == "skip" and quarantine.REPRODUCTION in item.markers[0].kwargs["reason"]
        for item in marked
    )
    saved = json.loads(report.read_text())
    assert saved["active"] is True
    assert {entry["nodeid"] for entry in saved["selected"]} == {item.nodeid for item in marked}
    # Identical class/params in another source, class, or method remain covered.
    item = marked[0]
    item.path = tmp_path / "another_test.py"
    assert not quarantine.matches(item, observed)
    item.path = source
    item.cls = type("OtherClass", (), {})
    assert not quarantine.matches(item, observed)
    item.cls = cls
    item.originalname = "another_method"
    assert not quarantine.matches(item, observed)


def test_missing_quarantine_report_fails_closed(suite, monkeypatch, tmp_path):
    junit = tmp_path / "result.xml"
    junit.write_text('<testsuite><testcase classname="example" name="test_ok"/></testsuite>')
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    def subprocess_run(cmd, check, env):
        assert env["PYTHONPATH"] == str(Path(suite.__file__).parent) + suite.os.pathsep + "existing-path"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", subprocess_run)
    with pytest.raises(suite.IncompleteRun, match="Invalid paired-quarantine report"):
        suite.run(["--pyargs", "scipy.example"], False, str(junit))
    assert suite.os.environ["PYTHONPATH"] == "existing-path"
