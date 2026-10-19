"""Regression tests for the identity helper used by fast-path coverage."""

import pytest

from fastscipy import _patch
from fastscipy.testing import check


def test_check_exception_runs_patched_call():
    states = []

    def call():
        states.append(_patch.enabled)
        raise ValueError("invalid input")

    assert check(call) is None
    assert states == [False, True]


def test_check_detects_missing_patched_exception():
    def call():
        if not _patch.enabled:
            raise ValueError("invalid input")
        return 42

    with pytest.raises(AssertionError, match="the patched call did not"):
        check(call, fast=False)


def test_check_detects_different_patched_exception():
    def call():
        if not _patch.enabled:
            raise ValueError("invalid input")
        raise TypeError("different error")

    with pytest.raises(AssertionError, match="the patched call raised"):
        check(call, fast=False)


def test_check_success_still_requires_fast_path():
    with pytest.raises(AssertionError, match="fast path did not run"):
        check(lambda: 42)
    assert check(lambda: 42, fast=False) == 42
