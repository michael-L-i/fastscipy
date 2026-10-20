"""Reference variation never excuses an unvalidated or deterministic error."""

import json

import numpy as np
import pytest

from fastscipy import testing


def close(expected, actual):
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)


@pytest.fixture
def sensitive(monkeypatch):
    monkeypatch.setattr(testing, "_alignment_sensitive_reference", lambda: True)


def test_records_observed_reference_variation(sensitive, monkeypatch, tmp_path):
    destination = tmp_path / "report.jsonl"
    monkeypatch.setenv("FASTSCIPY_REFERENCE_REPORT", str(destination))
    with pytest.warns(testing.ReferenceVarianceWarning, match="Byte identity is not established"):
        record = testing.reference_variance(lambda: np.nextafter(1.0, 2.0), 1.0, 1.0, close)
    assert record["reference_sha256"] != record["repeated_reference_sha256"]
    assert record["reference_trials"] == 1
    assert json.loads(destination.read_text()) == record


@pytest.mark.parametrize("validator", [None, close])
def test_deterministic_mismatch_is_not_accepted(sensitive, validator):
    assert testing.reference_variance(lambda: 1.0, 1.0, np.nextafter(1.0, 2.0), validator) is None


def test_bad_actual_fails_even_with_unstable_reference(sensitive):
    with pytest.raises(AssertionError):
        testing.reference_variance(lambda: np.nextafter(1.0, 2.0), 1.0, 2.0, close)


@pytest.mark.parametrize("answer", [False, np.bool_(False), "yes"])
def test_false_or_ambiguous_validator_result_fails_closed(sensitive, answer):
    with pytest.raises(AssertionError, match="validator rejected"):
        testing.reference_variance(lambda: np.nextafter(1.0, 2.0), 1.0, 2.0, lambda a, b: answer)


def test_bad_repeated_reference_does_not_establish_exception(sensitive):
    with pytest.raises(AssertionError):
        testing.reference_variance(lambda: 2.0, 1.0, np.nextafter(1.0, 2.0), close)


def test_other_platform_cannot_opt_in(monkeypatch):
    monkeypatch.setattr(testing, "_alignment_sensitive_reference", lambda: False)

    def unexpected(*args):
        raise AssertionError("Should not run numerical fallback on another backend")

    assert testing.reference_variance(unexpected, 1.0, 2.0, unexpected) is None


def test_missing_fast_path_still_fails(sensitive):
    values = iter((1.0, np.nextafter(1.0, 2.0)))
    with pytest.raises(AssertionError, match="fast path did not run"):
        testing.check(lambda: next(values), reference_validation=close)


def test_default_check_remains_exact(sensitive):
    values = iter((1.0, np.nextafter(1.0, 2.0)))
    with pytest.raises(AssertionError, match="results differ"):
        testing.check(lambda: next(values), fast=False)


def test_equal_calls_do_not_repeat_or_run_validator(sensitive):
    values = iter((1.0, 1.0))

    def unexpected(*args):
        raise AssertionError("Exact results need no exception")

    assert testing.check(lambda: next(values), fast=False, reference_validation=unexpected) == 1.0
    with pytest.raises(StopIteration):
        next(values)
