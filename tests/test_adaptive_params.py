"""Tests for adaptive_params (Level 2 of the self-improvement system)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_params import (
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    TIMEFRAME_PARAM_OVERRIDES,
    AdaptiveParams,
    _clamp,
    get_timeframe_defaults,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oscillating_bars(n: int = 300, base: float = 100.0, amp: float = 15.0) -> pd.DataFrame:
    """OHLCV where price oscillates wide enough to trigger RSI extremes."""
    rng = np.random.default_rng(42)
    x = np.arange(n)
    prices = base + amp * np.sin(x * 0.2) + rng.normal(0, 0.5, size=n)
    return pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "Open": prices,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": np.ones(n) * 1000,
        }
    )


def _make_closed_trades(n: int, atr_pct: float = 0.02, win_ratio: float = 0.6) -> list:
    """Build a list of synthetic closed trade dicts."""
    trades = []
    rng = np.random.default_rng(7)
    for i in range(n):
        is_win = i < int(n * win_ratio)
        pnl_pct = atr_pct * (2.0 if is_win else -1.0) * (1 + rng.normal(0, 0.1))
        trades.append(
            {
                "pnl_pct": pnl_pct,
                "atr_pct": atr_pct,
                "metadata": {"atr_pct": atr_pct},
            }
        )
    return trades


# ---------------------------------------------------------------------------
# Defaults / bounds
# ---------------------------------------------------------------------------


def test_default_params_contains_expected_keys():
    assert set(DEFAULT_PARAMS) >= {
        "rsi_overbought",
        "rsi_oversold",
        "sl_atr_mult",
        "tp_atr_mult",
        "kronos_min_confidence",
        "min_signal_strength",
    }


def test_param_bounds_respected_by_clamp():
    for name, (lo, hi) in PARAM_BOUNDS.items():
        assert _clamp(name, lo - 1000) == lo
        assert _clamp(name, hi + 1000) == hi
        assert _clamp(name, (lo + hi) / 2) == pytest.approx((lo + hi) / 2)


def test_params_bounded_rsi_range():
    # rsi_overbought in [50, 90], rsi_oversold in [10, 50]
    assert PARAM_BOUNDS["rsi_overbought"] == (50.0, 90.0)
    assert PARAM_BOUNDS["rsi_oversold"] == (10.0, 50.0)
    assert PARAM_BOUNDS["sl_atr_mult"] == (0.5, 3.0)


# ---------------------------------------------------------------------------
# get_params
# ---------------------------------------------------------------------------


def test_default_params_returned_when_no_data(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    params = ap.get_params("momentum", "BTC-USD")
    for k, v in DEFAULT_PARAMS.items():
        assert params[k] == pytest.approx(v)


def test_set_and_get_roundtrip(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    ap.set_param("mean_reversion", "BTC-USD", "rsi_overbought", 78.0, sample_size=100)
    out = ap.get_params("mean_reversion", "BTC-USD")
    assert out["rsi_overbought"] == pytest.approx(78.0)
    # Unrelated params should remain defaults
    assert out["sl_atr_mult"] == pytest.approx(DEFAULT_PARAMS["sl_atr_mult"])


def test_set_param_clamps_out_of_range(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    ap.set_param("momentum", "BTC-USD", "rsi_overbought", 200.0)
    out = ap.get_params("momentum", "BTC-USD")
    assert out["rsi_overbought"] == pytest.approx(90.0)  # clamped to upper bound


def test_latest_value_wins(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    ap.set_param("momentum", "ETH-USD", "sl_atr_mult", 1.0)
    ap.set_param("momentum", "ETH-USD", "sl_atr_mult", 2.5)
    out = ap.get_params("momentum", "ETH-USD")
    assert out["sl_atr_mult"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# optimize_rsi_thresholds
# ---------------------------------------------------------------------------


def test_optimize_rsi_thresholds_with_sufficient_data(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    bars = _oscillating_bars(n=300)
    out = ap.optimize_rsi_thresholds(bars)
    assert "rsi_overbought" in out and "rsi_oversold" in out
    assert PARAM_BOUNDS["rsi_overbought"][0] <= out["rsi_overbought"] <= PARAM_BOUNDS["rsi_overbought"][1]
    assert PARAM_BOUNDS["rsi_oversold"][0] <= out["rsi_oversold"] <= PARAM_BOUNDS["rsi_oversold"][1]
    assert out["rsi_oversold"] < out["rsi_overbought"]


def test_optimize_rsi_thresholds_with_insufficient_data(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    bars = _oscillating_bars(n=10)
    out = ap.optimize_rsi_thresholds(bars)
    assert out["rsi_overbought"] == DEFAULT_PARAMS["rsi_overbought"]
    assert out["rsi_oversold"] == DEFAULT_PARAMS["rsi_oversold"]


# ---------------------------------------------------------------------------
# optimize_stop_distances
# ---------------------------------------------------------------------------


def test_optimize_stop_distances_with_sufficient_trades(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    trades = _make_closed_trades(50, atr_pct=0.02, win_ratio=0.6)
    out = ap.optimize_stop_distances(trades)
    assert "sl_atr_mult" in out and "tp_atr_mult" in out
    lo, hi = PARAM_BOUNDS["sl_atr_mult"]
    assert lo <= out["sl_atr_mult"] <= hi
    assert out["tp_atr_mult"] > out["sl_atr_mult"]


def test_optimize_stop_distances_with_empty_trades(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    out = ap.optimize_stop_distances([])
    assert out["sl_atr_mult"] == DEFAULT_PARAMS["sl_atr_mult"]
    assert out["tp_atr_mult"] == DEFAULT_PARAMS["tp_atr_mult"]


def test_optimize_stop_distances_skips_trades_without_atr(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    trades = [{"pnl_pct": 0.01, "metadata": {}} for _ in range(20)]
    out = ap.optimize_stop_distances(trades)
    # Without atr_pct info we fall back to defaults.
    assert out["sl_atr_mult"] == DEFAULT_PARAMS["sl_atr_mult"]


# ---------------------------------------------------------------------------
# High-level optimize()
# ---------------------------------------------------------------------------


def test_optimize_persists_values(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    bars = _oscillating_bars(n=300)
    trades = _make_closed_trades(40)
    merged = ap.optimize("mean_reversion", "BTC-USD", trades=trades, bars=bars)
    assert "rsi_overbought" in merged
    # Verify that calling get_params again returns the persisted (not default) values.
    fresh = ap.get_params("mean_reversion", "BTC-USD")
    assert fresh["rsi_overbought"] == pytest.approx(merged["rsi_overbought"])


def test_optimize_with_no_data_returns_defaults(tmp_db_path):
    ap = AdaptiveParams(db_path=tmp_db_path)
    merged = ap.optimize("momentum", "XYZ-USD")
    for k, v in DEFAULT_PARAMS.items():
        assert merged[k] == pytest.approx(v)


# ---------------------------------------------------------------------------
# Timeframe-aware defaults
# ---------------------------------------------------------------------------


def test_timeframe_overrides_defined_for_intraday():
    for tf in ("5m", "15m", "1h", "4h", "1d"):
        assert tf in TIMEFRAME_PARAM_OVERRIDES


def test_5m_uses_tighter_stops_than_4h():
    five_m = TIMEFRAME_PARAM_OVERRIDES["5m"]["sl_atr_mult"]
    four_h = TIMEFRAME_PARAM_OVERRIDES["4h"]["sl_atr_mult"]
    assert five_m < four_h


def test_get_timeframe_defaults_overlays():
    out = get_timeframe_defaults("5m")
    assert out["sl_atr_mult"] == 0.75
    assert out["tp_atr_mult"] == 1.5
    # Untouched defaults still pass through.
    assert out["rsi_overbought"] == DEFAULT_PARAMS["rsi_overbought"]


def test_get_timeframe_defaults_unknown_returns_defaults():
    out = get_timeframe_defaults("7d")
    for k, v in DEFAULT_PARAMS.items():
        assert out[k] == v


def test_get_timeframe_defaults_none_returns_defaults():
    out = get_timeframe_defaults(None)
    for k, v in DEFAULT_PARAMS.items():
        assert out[k] == v
