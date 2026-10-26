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
    row = runner.run_case(different, 0, 2, False, reference_call=different)
    assert not row["identical"]
    assert row["validation"] == "reference_variance"
    assert set(row["reference_variance"]) == {"1t", "nt"}


def test_one_accepted_variance_cannot_hide_other_thread_mismatch(runner, monkeypatch):
    results = iter(({"policy": "recorded"}, None))
    monkeypatch.setattr(runner, "reference_variance", lambda *args: next(results))
    row = runner.run_case(different, 0, 2, False, reference_call=different)
    assert not row["identical"] and row["validation"] == "mismatch"


def test_variance_cannot_hide_missing_fast_path(runner, monkeypatch):
    monkeypatch.setattr(runner, "reference_variance", lambda *args: {"policy": "recorded"})
    row = runner.run_case(different, 0, 2, True, reference_call=different)
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


def test_paired_cases_isolate_rng_and_captured_scipy_functions(tmp_path, monkeypatch):
    import fastscipy

    module = load("paired_benchmark_under_test", "bench/run.py")
    monkeypatch.setattr(module, "HERE", tmp_path)
    (tmp_path / "bench_capture.py").write_text(
        "import numpy as np\n"
        "from scipy.spatial.distance import pdist\n"
        "from scipy.spatial import cKDTree\n"
        "rng = np.random.default_rng(12)\n"
        "def cases():\n"
        "    for i in range(2):\n"
        "        x = rng.random((8, 3))\n"
        "        tree = cKDTree(x)\n"
        "        yield str(i), lambda f=pdist, x=x, t=tree: (f, x, t)\n"
    )
    for _, call, reference_call, _, _ in module.paired_cases("capture"):
        candidate, candidate_x, candidate_tree = call()
        reference, reference_x, reference_tree = reference_call()
        assert hasattr(candidate, "__wrapped__")
        assert not hasattr(reference, "__wrapped__")
        assert type(candidate_tree) is not type(reference_tree)
        assert np.array_equal(candidate_x, reference_x)
        assert not np.shares_memory(candidate_x, reference_x)
        with module.unpatched():
            before = fastscipy.stats()
            reference(reference_x)
            reference_tree.query(reference_x)
            assert fastscipy.stats() == before
    assert fastscipy.status()["spatial"] == "on"


def test_unpatched_restores_active_components_after_error():
    import fastscipy
    import scipy.spatial.distance as distance

    module = load("unpatched_benchmark_under_test", "bench/run.py")
    active = {name for name, state in fastscipy.status().items() if state == "on"}
    with pytest.raises(RuntimeError, match="fixture failed"):
        with module.unpatched():
            assert not hasattr(distance.pdist, "__wrapped__")
            raise RuntimeError("fixture failed")
    assert {name for name, state in fastscipy.status().items() if state == "on"} == active
    assert hasattr(distance.pdist, "__wrapped__")


def test_timing_reports_median_and_excludes_calibration(monkeypatch):
    module = load("timing_benchmark_under_test", "bench/run.py")
    clock = [0.0]
    durations = iter([20, 30, 40, 50, 60, 70])
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    def call():
        clock[0] += next(durations) * 1e-6

    assert module.timeit(call, 0) == pytest.approx(50e-6)
    assert list(durations) == []


def test_timing_batches_tiny_calls(monkeypatch):
    module = load("tiny_timing_benchmark_under_test", "bench/run.py")
    clock, calls = [0.0], [0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    def call():
        calls[0] += 1
        clock[0] += 1e-7

    assert module.timeit(call, 0) == pytest.approx(1e-7)
    assert calls[0] >= 500


def test_headline_excludes_controls_but_keeps_their_rows(runner, monkeypatch, tmp_path, capsys):
    import json

    monkeypatch.setattr(runner.sys, "argv", ["run.py", "fake", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(
        runner,
        "paired_cases",
        lambda area: iter([("fast", None, None, True, None), ("control", None, None, False, None)]),
    )

    def result(call, budget, threads, expected, validation, *, reference_call):
        return dict(
            scipy_ms=1,
            **{"1t_ms": 1, "nt_ms": 1},
            identical=True,
            fast_path=True,
            x1t=4 if expected else 0.5,
            xnt=4 if expected else 0.5,
            validation="byte_identical",
        )

    monkeypatch.setattr(runner, "run_case", result)
    runner.main()
    output = capsys.readouterr().out
    assert "**4.0**" in output
    assert "slower than SciPy" in output and "fake: control" in output
    report = json.loads((tmp_path / "fake.json").read_text())
    assert set(report["cases"]) == {"fast", "control"}
    assert report["meta"]["headline_cases"] == 1
    assert report["meta"]["reference"] == "unpatched_scipy"


def test_reference_rejects_a_captured_wrapper(runner):
    from scipy.spatial.distance import pdist

    x = np.arange(12.0).reshape(4, 3)
    with pytest.raises(RuntimeError, match="reference reached a fastscipy wrapper"):
        runner.run_case(lambda: pdist(x), 0, 2, reference_call=lambda: pdist(x))
