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

    def subprocess_run(cmd, check):
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
    monkeypatch.setattr(suite.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    results, returncode = suite.run(["--pyargs", "scipy.example"], False, str(junit))
    assert returncode == 1
    assert results == {"tests.example::test_bad": "failed"}
