"""Small process doubles for privileged installer monitoring; no native kernels."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

SPEC = importlib.util.spec_from_file_location(
    "platform_monitor_under_test", Path(__file__).parents[1] / "tools/platform_suite.py"
)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class Child:
    def __init__(self, pid, name, rss=0, *, denied=False, kill_error=None):
        self.pid, self.process_name, self.rss = pid, name, rss
        self.denied, self.kill_error, self.killed = denied, kill_error, False
        self.children_error = None
        self.descendants = []

    def name(self):
        if self.process_name is None:
            raise psutil.AccessDenied(self.pid)
        return self.process_name

    def memory_info(self):
        if self.denied:
            raise psutil.AccessDenied(self.pid)
        if isinstance(self.rss, BaseException):
            raise self.rss
        return SimpleNamespace(rss=self.rss)

    def children(self, recursive):
        assert recursive
        if self.children_error:
            raise self.children_error
        return self.descendants

    def kill(self):
        self.killed = True
        if self.kill_error:
            raise self.kill_error


class Command:
    pid = 100

    def __init__(self, samples=1, code=0):
        self.samples, self.code = samples, code
        self.polls, self.killed, self.waited = 0, False, False

    def poll(self):
        self.polls += 1
        return None if self.polls <= self.samples else self.code

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        assert timeout == 10
        self.waited = True
        return -9 if self.killed else self.code


@pytest.fixture
def fake(monkeypatch):
    command, parent = Command(), Child(100, "python", 8)
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **k: command)
    monkeypatch.setattr(monitor.psutil, "Process", lambda pid: parent)
    monkeypatch.setattr(monitor, "time", SimpleNamespace(sleep=lambda seconds: None, monotonic=lambda: 0))
    monkeypatch.setattr(monitor.sys, "platform", "darwin")
    return command, parent


def resource_file(log):
    return json.loads(log.with_suffix(".log.monitor.json").read_text())


@pytest.mark.parametrize("name", ["sudo", "installer", "installd", "package_script_service", "system_installd"])
def test_opted_in_darwin_installer_retains_unknown_rss(tmp_path, fake, name):
    command, parent = fake
    child = Child(200, name, denied=True)
    parent.descendants = [child, Child(201, "rustc", 16)]
    log = tmp_path / "build.log"
    result = monitor.run(["build"], log, 1, 10, allow_privileged_installer=True)
    assert result["returncode"] == 0 and result["stopped"] is None
    assert result["peak_rss_bytes"] == 24  # readable compiler memory is still counted
    assert result["rss_complete"] is False and result["peak_rss_is_lower_bound"] is True
    assert result["inaccessible_pids"] == [200]
    assert result["inaccessible_processes"] == [
        {
            "pid": 200,
            "name": name,
            "first_denied_seconds": 0,
            "last_denied_seconds": 0,
            "denied_samples": 1,
            "allowed_installer": True,
            "operation": "memory_info",
        }
    ]
    assert resource_file(log) == result
    assert command.waited and not command.killed and not child.killed


@pytest.mark.parametrize("name", ["rustc", "cc", "unknown", None])
def test_unknown_or_compiler_denial_fails_and_cleans_up(tmp_path, fake, name):
    command, parent = fake
    child = Child(200, name, denied=True)
    parent.descendants = [child]
    log = tmp_path / "build.log"
    result = monitor.run(["build"], log, 1, 10, allow_privileged_installer=True)
    assert result["returncode"] != 0 and result["stopped"] == "monitor access denied"
    assert result["monitor_error"]["pid"] == 200
    assert result["inaccessible_processes"][0]["allowed_installer"] is False
    assert result["rss_complete"] is False and result["peak_rss_is_lower_bound"] is True
    assert child.killed and command.killed and command.waited
    assert resource_file(log) == result


@pytest.mark.parametrize("opt_in,system", [(False, "darwin"), (True, "linux")])
def test_installer_exception_requires_explicit_darwin_phase(tmp_path, fake, monkeypatch, opt_in, system):
    command, parent = fake
    parent.descendants = [Child(200, "installer", denied=True)]
    monkeypatch.setattr(monitor.sys, "platform", system)
    result = monitor.run(["test"], tmp_path / "test.log", 1, 10, allow_privileged_installer=opt_in)
    assert result["returncode"] != 0 and command.killed


@pytest.mark.parametrize("operation", ["memory", "enumeration"])
def test_monitored_root_denial_is_never_allowed(tmp_path, fake, operation):
    command, parent = fake
    parent.process_name = "installer"
    if operation == "memory":
        parent.denied = True
    else:
        parent.children_error = psutil.AccessDenied(parent.pid)
    result = monitor.run(["build"], tmp_path / "build.log", 1, 10, allow_privileged_installer=True)
    assert result["returncode"] != 0 and result["inaccessible_pids"] == [parent.pid]
    assert command.killed and command.waited


@pytest.mark.parametrize("limit", ["memory", "time"])
def test_installer_denial_does_not_disable_limits(tmp_path, fake, monkeypatch, limit):
    command, parent = fake
    installer = Child(200, "installer", denied=True, kill_error=psutil.AccessDenied(200))
    compiler = Child(201, "rustc", 2 * 1024**2)
    parent.descendants = [installer, compiler]
    if limit == "time":
        compiler.rss = 0
        monkeypatch.setattr(monitor.time, "monotonic", iter([0, 1, 20, 20]).__next__)
    result = monitor.run(["build"], tmp_path / "build.log", 1, 10, allow_privileged_installer=True)
    assert result["stopped"] == ("memory limit" if limit == "memory" else "timeout")
    assert result["returncode"] != 0 and result["rss_complete"] is False
    assert result["cleanup_errors"][0]["pid"] == 200
    assert command.killed and command.waited and compiler.killed


def test_unexpected_monitor_error_is_re_raised_after_cleanup(tmp_path, fake):
    command, parent = fake
    error = RuntimeError("unexpected monitor bug")
    child = Child(200, "rustc", error)
    parent.descendants = [child]
    log = tmp_path / "build.log"
    with pytest.raises(RuntimeError, match="unexpected monitor bug") as raised:
        monitor.run(["build"], log, 1, 10)
    assert raised.value is error
    report = resource_file(log)
    assert report["returncode"] != 0 and report["stopped"] == "monitor error"
    assert report["monitor_error"]["type"] == "RuntimeError" and report["rss_complete"] is False
    assert child.killed and command.killed and command.waited


def test_compiler_exit_failure_is_not_changed_to_success(tmp_path, fake):
    command, _ = fake
    command.code = 137
    result = monitor.run(["build"], tmp_path / "build.log", 1, 10)
    assert result["returncode"] == 137
    assert result["stopped"] is None and result["rss_complete"] is True


def test_installer_denial_period_covers_multiple_samples(tmp_path, fake, monkeypatch):
    command, parent = fake
    command.samples = 2
    parent.descendants = [Child(200, "installer", denied=True)]
    monkeypatch.setattr(monitor.time, "monotonic", iter([0, 1, 1, 3, 3, 5]).__next__)
    result = monitor.run(["build"], tmp_path / "build.log", 1, 10, allow_privileged_installer=True)
    denied = result["inaccessible_processes"][0]
    assert denied["first_denied_seconds"] == 1 and denied["last_denied_seconds"] == 3
    assert denied["denied_samples"] == 2 and result["seconds"] == 5


def test_compatibility_records_are_not_observed_reference_variance():
    old = {"policy": "intel-accelerate-reference-variance", "reference_trials": 1}
    new = {"policy": "intel-accelerate-numerical-compatibility", "reference_trials": None}
    other = {"policy": "unknown-future-policy"}
    records = [old, new, other]
    result = monitor.validation_record_groups(records)
    assert result["reference_variance"] == [old]
    assert result["numerical_compatibility"] == [new]
    assert result["validation_records"] is records  # retain every original record
