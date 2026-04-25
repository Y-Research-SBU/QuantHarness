"""Tests for the Kronos timeframe trust guard (REL-376 partial fix)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kronos_agent import (  # noqa: E402
    KRONOS_TRUSTED_TIMEFRAMES,
    is_timeframe_trusted_for_kronos,
)


# ════════════════════════════════════════════════════════════════
# Trusted set membership
# ════════════════════════════════════════════════════════════════


def test_only_1d_is_trusted_currently():
    assert KRONOS_TRUSTED_TIMEFRAMES == frozenset({"1d"})


def test_1d_is_trusted():
    assert is_timeframe_trusted_for_kronos("1d") is True


@pytest.mark.parametrize(
    "tf",
    ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1w"],
)
def test_non_1d_timeframes_are_distrusted(tf):
    """Anything except 1d should be blocked from Kronos until the horizon
    sweep finishes (audit showed 27.9% accuracy on 1h horizon=24)."""
    assert is_timeframe_trusted_for_kronos(tf) is False


# ════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════


def test_none_timeframe_is_distrusted():
    assert is_timeframe_trusted_for_kronos(None) is False


def test_empty_timeframe_is_distrusted():
    assert is_timeframe_trusted_for_kronos("") is False


def test_unknown_timeframe_is_distrusted():
    assert is_timeframe_trusted_for_kronos("3h") is False


def test_uppercase_is_distrusted():
    """Strict case match — accidental capitalization should fail closed."""
    assert is_timeframe_trusted_for_kronos("1D") is False


# ════════════════════════════════════════════════════════════════
# Frozenset immutability
# ════════════════════════════════════════════════════════════════


def test_trusted_set_is_immutable():
    """Module-level attribute must be frozenset so misuse can't widen it."""
    assert isinstance(KRONOS_TRUSTED_TIMEFRAMES, frozenset)
    with pytest.raises((AttributeError, TypeError)):
        KRONOS_TRUSTED_TIMEFRAMES.add("1h")  # type: ignore[attr-defined]
