"""Call accounting remains transparent to dispatch and pytest worker aggregation."""

import inspect
from types import SimpleNamespace

import pytest

import fastscipy
from fastscipy import _patch, pytest_plugin


@pytest.fixture
def counts(monkeypatch):
    result = _patch._CallCounts()
    monkeypatch.setattr(_patch, "counts", result)
    return result


def test_unseen_count_does_not_change_stats(counts):
    assert counts["unseen", "fast"] == 0
    assert not counts
    assert fastscipy.stats() == {}
    counts["called", "fast"] += 1
    snapshot = fastscipy.stats()
    assert type(snapshot) is dict
    counts["called", "fast"] += 1
    assert snapshot == {("called", "fast"): 1}
    snapshot.clear()
    assert fastscipy.stats() == {("called", "fast"): 2}


def test_pytest_worker_counts_roundtrip(counts):
    counts["sample", "fast"] += 3
    counts["sample", "scipy"] += 2
    output = {}
    session = SimpleNamespace(config=SimpleNamespace(workeroutput=output))
    pytest_plugin.pytest_sessionfinish(session)
    counts.clear()
    node = SimpleNamespace(workeroutput=output)
    pytest_plugin.pytest_testnodedown(node, None)
    pytest_plugin.pytest_testnodedown(node, None)
    assert fastscipy.stats() == {("sample", "fast"): 6, ("sample", "scipy"): 4}


@pytest.mark.parametrize("mode", ["fast", "decline", "disabled", "error", "strict"])
def test_wrapper_accounting_and_exception_contract(mode, counts, monkeypatch):
    calls = []
    error = RuntimeError("fast failure")

    def original(x, *, scale=1):
        """Original documentation."""
        calls.append("original")
        return x * scale

    def fast(x, *, scale=1):
        calls.append("fast")
        if mode in ("error", "strict"):
            raise error
        return NotImplemented if mode == "decline" else x * scale

    wrapper = _patch.wrap(original, fast, "accounting")
    assert wrapper.__wrapped__ is original
    assert wrapper.fast is fast
    assert wrapper.__name__ == original.__name__
    assert wrapper.__doc__ == original.__doc__
    assert inspect.signature(wrapper) == inspect.signature(original)
    monkeypatch.setattr(_patch, "STRICT", mode == "strict")
    monkeypatch.setattr(_patch, "enabled", mode != "disabled")
    if mode == "strict":
        with pytest.raises(RuntimeError) as raised:
            wrapper(3, scale=4)
        assert raised.value is error
        assert calls == ["fast"]
        assert not counts
        return
    assert wrapper(3, scale=4) == 12
    expected = {
        "fast": (["fast"], {("accounting", "fast"): 1}),
        "decline": (["fast", "original"], {("accounting", "scipy"): 1}),
        "disabled": (["original"], {("accounting", "scipy"): 1}),
        "error": (["fast", "original"], {("accounting", "error"): 1, ("accounting", "scipy"): 1}),
    }
    assert (calls, fastscipy.stats()) == expected[mode]
