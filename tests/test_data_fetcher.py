"""Unit tests for data_fetcher.

Focuses on the NaN-bar regression that left crypto positions stuck OPEN with
pnl=NULL on the paper-trading runner. yfinance returns an incomplete "today"
daily bar (NaN OHLCV) for crypto on weekends; callers were taking
``df["Close"].iloc[-1]`` and getting NaN, which silently bypassed the
``price <= stop_loss`` check in ``PaperTradingEngine.check_stops`` (NaN
comparisons are always False) and caused ``mark_to_market`` to skip the
symbol entirely.

The fix in ``data_fetcher.fetch_market_data`` drops rows where ``Close`` is
non-finite before returning. These tests pin that contract.
"""
from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data_fetcher import fetch_market_data


def _make_yf_frame(rows):
    """Build a yfinance-style frame with a Datetime index and OHLCV columns."""
    df = pd.DataFrame(
        rows,
        columns=["Datetime", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    return df


@pytest.fixture(autouse=True)
def _no_cache():
    """Disable Redis cache so the function exercises the yfinance branch."""
    with patch("data_fetcher.get_cache", None):
        yield


def test_fetch_drops_trailing_nan_close_bar():
    """Today's partial bar (NaN Close) must be dropped before return."""
    raw = _make_yf_frame([
        ("2026-04-21", 100.0, 110.0,  95.0, 105.0, 1_000),
        ("2026-04-22", 105.0, 115.0, 100.0, 112.0, 1_200),
        ("2026-04-23", 112.0, 120.0, 108.0, 118.0,   900),
        ("2026-04-24", 118.0, 122.0, 115.0, 120.0, 1_100),
        # Today: yfinance returns a row with all NaNs (weekend partial bar).
        ("2026-04-25", np.nan, np.nan, np.nan, np.nan, 0),
    ])

    with patch("data_fetcher.yf.download", return_value=raw):
        out = fetch_market_data("BTC-USD", "1d", bars=5, use_cache=False)

    assert not out.empty
    assert len(out) == 4, "the NaN row must be filtered out"
    last_close = float(out["Close"].iloc[-1])
    assert math.isfinite(last_close)
    assert last_close == pytest.approx(120.0)
    # No NaN should remain anywhere in the Close column.
    assert out["Close"].notna().all()


def test_fetch_drops_interior_nan_close_bar():
    """A NaN bar in the middle of the series must also be dropped."""
    raw = _make_yf_frame([
        ("2026-04-21", 100.0, 110.0,  95.0, 105.0, 1_000),
        ("2026-04-22", np.nan, np.nan, np.nan, np.nan, 0),
        ("2026-04-23", 112.0, 120.0, 108.0, 118.0,   900),
    ])

    with patch("data_fetcher.yf.download", return_value=raw):
        out = fetch_market_data("ETH-USD", "1d", bars=10, use_cache=False)

    assert len(out) == 2
    assert out["Close"].notna().all()
    assert out["Close"].tolist() == [105.0, 118.0]


def test_fetch_returns_empty_when_only_nan_bars():
    """If every bar is NaN we should return an empty frame, not bogus data."""
    raw = _make_yf_frame([
        ("2026-04-24", np.nan, np.nan, np.nan, np.nan, 0),
        ("2026-04-25", np.nan, np.nan, np.nan, np.nan, 0),
    ])

    with patch("data_fetcher.yf.download", return_value=raw):
        out = fetch_market_data("FAKE-USD", "1d", bars=5, use_cache=False)

    # All rows dropped -> empty frame, but the function must not raise.
    assert out.empty or out["Close"].notna().all()
    if not out.empty:
        # Defensive: if any row survives it must be finite.
        assert math.isfinite(float(out["Close"].iloc[-1]))


def test_fetch_passes_through_clean_data():
    """When yfinance returns clean data, the function must not alter values."""
    raw = _make_yf_frame([
        ("2026-04-23", 100.0, 105.0,  98.0, 102.0, 1_000),
        ("2026-04-24", 102.0, 108.0, 101.0, 107.0, 1_200),
        ("2026-04-25", 107.0, 110.0, 105.0, 109.0, 1_100),
    ])

    with patch("data_fetcher.yf.download", return_value=raw):
        out = fetch_market_data("SPY", "1d", bars=5, use_cache=False)

    assert len(out) == 3
    assert out["Close"].tolist() == [102.0, 107.0, 109.0]


def test_fetch_bars_limit_applies_after_nan_drop():
    """`bars` must clamp the *cleaned* series, not the raw one."""
    rows = [
        (f"2026-04-{d:02d}", 100.0 + d, 110.0 + d, 95.0 + d, 105.0 + d, 1000)
        for d in range(10, 25)
    ]
    # Append a final NaN bar to simulate the partial-day case.
    rows.append(("2026-04-25", np.nan, np.nan, np.nan, np.nan, 0))
    raw = _make_yf_frame(rows)

    with patch("data_fetcher.yf.download", return_value=raw):
        out = fetch_market_data("AAPL", "1d", bars=5, use_cache=False)

    assert len(out) == 5
    assert out["Close"].notna().all()
    # The clamped tail must come from the cleaned series, ending at 2026-04-24.
    assert float(out["Close"].iloc[-1]) == pytest.approx(105.0 + 24)
