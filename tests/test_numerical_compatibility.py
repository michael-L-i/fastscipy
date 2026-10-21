"""Explicit numerical contracts never masquerade as byte/reference-variance proof."""

import json

import numpy as np
import pytest

from fastscipy import testing


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)


@pytest.fixture
def sensitive(monkeypatch, tmp_path):
    monkeypatch.setattr(testing, "_alignment_sensitive_reference", lambda: True)
    destination = tmp_path / "numerical.jsonl"
    monkeypatch.setenv("FASTSCIPY_REFERENCE_REPORT", str(destination))
    return destination


def test_explicit_contract_does_not_claim_reference_variance(sensitive):
    actual = np.nextafter(1.0, 2.0)
    values = iter((1.0, actual))
    with pytest.warns(testing.NumericalCompatibilityWarning, match="not established"):
        assert testing.check(lambda: next(values), fast=False, numerical_validation=close) == actual
    record = json.loads(sensitive.read_text())
    assert record["policy"] == "intel-accelerate-numerical-compatibility"
    assert record["reference_variance_observed"] is False
    assert "reference_trials" not in record and "repeated_reference_sha256" not in record
    assert record["reference_sha256"] != record["actual_sha256"]
    with pytest.raises(StopIteration):
        next(values)


def test_default_remains_exact(sensitive):
    values = iter((1.0, np.nextafter(1.0, 2.0)))
    with pytest.raises(AssertionError, match="results differ"):
        testing.check(lambda: next(values), fast=False)
    assert not sensitive.exists()


def test_other_backend_remains_exact(sensitive, monkeypatch):
    monkeypatch.setattr(testing, "_alignment_sensitive_reference", lambda: False)
    values = iter((1.0, np.nextafter(1.0, 2.0)))
    with pytest.raises(AssertionError, match="results differ"):
        testing.check(lambda: next(values), fast=False, numerical_validation=close)
    assert not sensitive.exists()


def test_fast_path_is_required_before_validator(sensitive):
    values = iter((1.0, np.nextafter(1.0, 2.0)))

    def unexpected(a, b):
        raise AssertionError("Validator must not fake acceleration")

    with pytest.raises(AssertionError, match="fast path did not run"):
        testing.check(lambda: next(values), numerical_validation=unexpected)
    assert not sensitive.exists()


@pytest.mark.parametrize("answer", [False, np.bool_(True), np.bool_(False), "yes"])
def test_invalid_validator_result_is_rejected(sensitive, answer):
    with pytest.raises(AssertionError, match="numerical validator rejected"):
        testing.numerical_compatibility(1.0, 2.0, lambda a, b: answer)
    assert not sensitive.exists()


def test_inaccurate_result_is_rejected(sensitive):
    with pytest.raises(AssertionError):
        testing.numerical_compatibility(1.0, 2.0, close)
    assert not sensitive.exists()


def test_equal_result_needs_no_exception(sensitive):
    def unexpected(a, b):
        raise AssertionError("Exact comparisons need no numerical exception")

    assert testing.check(lambda: 1.0, fast=False, numerical_validation=unexpected) == 1.0
    assert not sensitive.exists()


def test_conflicting_policies_are_rejected(sensitive):
    with pytest.raises(ValueError, match="Choose"):
        testing.check(lambda: 1.0, reference_validation=close, numerical_validation=close)
