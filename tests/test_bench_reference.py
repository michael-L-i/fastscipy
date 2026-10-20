"""Keep byte-identity and acceleration failures visible in benchmark rows."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fastscipy import _patch


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(monkeypatch):
    module = load("benchmark_under_test", "bench/run.py")
    monkeypatch.setattr(module, "timeit", lambda *args: 1.0)
    monkeypatch.setattr(module.fastscipy, "set_num_threads", lambda n: None)
    monkeypatch.setattr(module, "fast_calls", lambda: 0)
    return module


def different():
    return np.nextafter(1.0, 2.0) if _patch.enabled else 1.0


def test_recorded_variance_stays_nonidentical(runner, monkeypatch):
    monkeypatch.setattr(runner, "reference_variance", lambda *args: {"policy": "recorded"})
    row = runner.run_case(different, 0, 2, False)
    assert not row["identical"]
    assert row["validation"] == "reference_variance"
    assert set(row["reference_variance"]) == {"1t", "nt"}


def test_one_accepted_variance_cannot_hide_other_thread_mismatch(runner, monkeypatch):
    results = iter(({"policy": "recorded"}, None))
    monkeypatch.setattr(runner, "reference_variance", lambda *args: next(results))
    row = runner.run_case(different, 0, 2, False)
    assert not row["identical"] and row["validation"] == "mismatch"


def test_variance_cannot_hide_missing_fast_path(runner, monkeypatch):
    monkeypatch.setattr(runner, "reference_variance", lambda *args: {"policy": "recorded"})
    row = runner.run_case(different, 0, 2, True)
    assert row["validation"] == "reference_variance" and not row["fast_path"]


def test_bfgs_uses_known_minimum_instead_of_only_comparing_two_answers():
    validate = load("optimize_benchmark_under_test", "bench/bench_optimize.py")._bfgs_reference_validation
    validate(np.ones(10), np.ones(10))
    for point in (
        np.ones(9),
        np.ones(10, dtype=np.float32),
        np.ones(20)[::2],
        np.full(10, 1.00005),
        np.full(10, np.nan),
    ):
        with pytest.raises(AssertionError):
            validate(point, point)
