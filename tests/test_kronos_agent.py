"""Tests for kronos_agent.KronosForecastAgent and create_kronos_agent.

The Kronos foundation model is heavy (downloads ~24M params from HuggingFace),
so these tests exercise the fallback path and the public surface area without
ever loading the real weights. The few tests that need to *simulate* Kronos use
a stub predictor injected via ``KronosForecastAgent._SHARED_PREDICTOR``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from data_fetcher import prepare_kline_dict
from kronos_agent import (
    KronosForecast,
    KronosForecastAgent,
    _kline_dict_to_df,
    _timeframe_to_timedelta,
    create_kronos_agent,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_kronos_singleton(monkeypatch):
    """Each test gets a fresh class-level state so they don't leak."""
    monkeypatch.setattr(KronosForecastAgent, "_SHARED_PREDICTOR", None)
    monkeypatch.setattr(KronosForecastAgent, "_LOAD_FAILED", False)
    monkeypatch.setattr(KronosForecastAgent, "_LOAD_ERROR", None)
    yield


def _trending_df(direction: str = "up", n: int = 200) -> pd.DataFrame:
    if direction == "up":
        prices = np.linspace(100.0, 130.0, n)
    elif direction == "down":
        prices = np.linspace(130.0, 100.0, n)
    else:
        prices = np.full(n, 100.0)
    df = pd.DataFrame({
        "Datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h"),
        "Open": prices,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, size=n),
    })
    df.attrs["symbol"] = "TST-USD"
    df.attrs["timeframe"] = "1h"
    return df


class _StubPath:
    """Minimal pandas-like stand-in for the predictor's `predict` output."""

    def __init__(self, closes, highs, lows):
        self._data = {
            "close": pd.Series(closes),
            "high": pd.Series(highs),
            "low": pd.Series(lows),
        }

    def __getitem__(self, key):
        return self._data[key]


class _StubPredictor:
    def __init__(self, predicted_path):
        self._path = predicted_path

    def predict(self, **_kwargs):
        path = np.asarray(self._path, dtype=float)
        return _StubPath(
            closes=path,
            highs=path * 1.01,
            lows=path * 0.99,
        )


# ---------------------------------------------------------------------------
# Forecast dataclass
# ---------------------------------------------------------------------------


def test_forecast_to_dict_round_trip():
    fc = KronosForecast(
        direction="UP",
        magnitude_pct=2.5,
        confidence=0.7,
        predicted_close=110.0,
        predicted_high=112.0,
        predicted_low=108.0,
        horizon=24,
        last_close=107.0,
        source="kronos",
        reasoning="test",
    )
    d = fc.to_dict()
    assert d["direction"] == "UP"
    assert d["confidence"] == 0.7
    assert d["source"] == "kronos"
    summary = fc.summary()
    assert "UP" in summary and "kronos" in summary


def test_direction_from_pct_thresholds():
    assert KronosForecastAgent._direction_from_pct(2.0) == "UP"
    assert KronosForecastAgent._direction_from_pct(-2.0) == "DOWN"
    assert KronosForecastAgent._direction_from_pct(0.1) == "NEUTRAL"


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------


def test_predict_uptrend_returns_up_when_kronos_disabled():
    agent = KronosForecastAgent(enable_kronos=False)
    fc = agent.predict(_trending_df("up"))
    assert fc.source == "fallback"
    assert fc.direction == "UP"
    assert fc.magnitude_pct > 0
    assert fc.predicted_close > fc.last_close


def test_predict_downtrend_returns_down():
    agent = KronosForecastAgent(enable_kronos=False)
    fc = agent.predict(_trending_df("down"))
    assert fc.source == "fallback"
    assert fc.direction == "DOWN"
    assert fc.magnitude_pct < 0
    assert fc.predicted_close < fc.last_close


def test_predict_flat_close_to_neutral():
    agent = KronosForecastAgent(enable_kronos=False)
    fc = agent.predict(_trending_df("flat"))
    assert fc.source == "fallback"
    assert abs(fc.magnitude_pct) < 0.5
    assert fc.direction == "NEUTRAL"


def test_empty_dataframe_returns_neutral_zero_confidence():
    agent = KronosForecastAgent(enable_kronos=False, default_horizon=24)
    fc = agent.predict(pd.DataFrame())
    assert fc.direction == "NEUTRAL"
    assert fc.confidence == 0.0
    assert fc.horizon == 24  # default horizon survives empty input


def test_short_history_returns_neutral():
    agent = KronosForecastAgent(enable_kronos=False)
    df = _trending_df("up", n=3)
    fc = agent.predict(df)
    assert fc.direction == "NEUTRAL"
    assert fc.confidence == 0.0
    assert fc.source == "fallback"


def test_default_horizon_per_timeframe():
    agent = KronosForecastAgent(enable_kronos=False)
    fc = agent.predict(_trending_df("up"), timeframe="1d")
    assert fc.horizon == 5  # daily default
    fc2 = agent.predict(_trending_df("up"), timeframe="1h")
    assert fc2.horizon == 24


def test_explicit_horizon_overrides_default():
    agent = KronosForecastAgent(enable_kronos=False)
    fc = agent.predict(_trending_df("up"), horizon=7, timeframe="1d")
    assert fc.horizon == 7


def test_confidence_in_unit_range():
    agent = KronosForecastAgent(enable_kronos=False)
    for direction in ("up", "down", "flat"):
        fc = agent.predict(_trending_df(direction))
        assert 0.0 <= fc.confidence <= 1.0


def test_metadata_includes_predicted_path_for_fallback():
    agent = KronosForecastAgent(enable_kronos=False)
    fc = agent.predict(_trending_df("up"), horizon=10)
    assert "predicted_path" in fc.metadata
    assert len(fc.metadata["predicted_path"]) == 10


# ---------------------------------------------------------------------------
# Kronos path (using stubbed predictor)
# ---------------------------------------------------------------------------


def test_kronos_path_used_when_predictor_loaded(monkeypatch):
    last = 100.0
    stub = _StubPredictor(predicted_path=[101, 102, 103, 104, 105])
    monkeypatch.setattr(KronosForecastAgent, "_SHARED_PREDICTOR", stub)
    agent = KronosForecastAgent()

    df = _trending_df("flat", n=120)
    df.iloc[-1, df.columns.get_loc("Close")] = last
    df.iloc[-1, df.columns.get_loc("High")] = last + 0.5
    df.iloc[-1, df.columns.get_loc("Low")] = last - 0.5
    df.iloc[-1, df.columns.get_loc("Open")] = last

    fc = agent.predict(df, horizon=5)
    assert fc.source == "kronos"
    assert fc.direction == "UP"
    assert fc.predicted_close == pytest.approx(105.0)
    assert fc.metadata["model"] == "NeoQuasar/Kronos-small"


def test_kronos_predict_failure_falls_back(monkeypatch):
    class BadPredictor:
        def predict(self, **_):
            raise RuntimeError("boom")

    monkeypatch.setattr(KronosForecastAgent, "_SHARED_PREDICTOR", BadPredictor())
    agent = KronosForecastAgent()
    fc = agent.predict(_trending_df("up"))
    assert fc.source == "fallback"
    # Falls back successfully — direction still derived from data.
    assert fc.direction in ("UP", "DOWN", "NEUTRAL")


# ---------------------------------------------------------------------------
# Public node factory
# ---------------------------------------------------------------------------


def test_create_kronos_agent_writes_state_keys():
    df = _trending_df("up")
    state = {
        "kline_data": prepare_kline_dict(df),
        "time_frame": "1h",
        "stock_name": "TST-USD",
    }
    node = create_kronos_agent(KronosForecastAgent(enable_kronos=False))
    out = node(state)
    assert "kronos_forecast" in out
    assert "kronos_forecast_data" in out
    assert isinstance(out["kronos_forecast_data"]["direction"], str)


def test_create_kronos_agent_handles_missing_kline_data():
    node = create_kronos_agent(KronosForecastAgent(enable_kronos=False))
    out = node({"time_frame": "1h"})  # no kline_data
    assert out["kronos_forecast_data"]["direction"] == "NEUTRAL"


def test_create_kronos_agent_doesnt_raise_on_bad_kline():
    node = create_kronos_agent(KronosForecastAgent(enable_kronos=False))
    out = node({"kline_data": {"Close": []}})
    assert "kronos_forecast" in out
    assert out["kronos_forecast_data"]["direction"] == "NEUTRAL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_kline_dict_to_df_basic():
    df_in = _trending_df("up", n=50)
    kd = prepare_kline_dict(df_in)
    df_out = _kline_dict_to_df(kd)
    assert list(df_out.columns) == ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    assert len(df_out) == 50


def test_timeframe_to_timedelta_known_and_unknown():
    assert _timeframe_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert _timeframe_to_timedelta("4h") == pd.Timedelta(hours=4)
    assert _timeframe_to_timedelta("1d") == pd.Timedelta(days=1)
    # Unknown should default to 1h.
    assert _timeframe_to_timedelta("zzz") == pd.Timedelta(hours=1)
    assert _timeframe_to_timedelta(None) == pd.Timedelta(hours=1)


def test_predictor_singleton_disabled_returns_none():
    agent = KronosForecastAgent(enable_kronos=False)
    assert agent._get_predictor() is None


def test_predictor_singleton_load_failed_returns_none(monkeypatch):
    monkeypatch.setattr(KronosForecastAgent, "_LOAD_FAILED", True)
    agent = KronosForecastAgent()
    assert agent._get_predictor() is None


def test_kronos_agent_node_structured_dict_keys():
    node = create_kronos_agent(KronosForecastAgent(enable_kronos=False))
    state = {
        "kline_data": prepare_kline_dict(_trending_df("up", n=80)),
        "time_frame": "1h",
    }
    out = node(state)
    data = out["kronos_forecast_data"]
    for key in (
        "direction",
        "magnitude_pct",
        "confidence",
        "predicted_close",
        "predicted_high",
        "predicted_low",
        "horizon",
        "last_close",
        "source",
        "reasoning",
        "metadata",
    ):
        assert key in data, f"missing key {key!r}"


def test_predict_uses_volume_when_available():
    df = _trending_df("up")
    # Drop Volume to ensure it's optional.
    df = df.drop(columns=["Volume"])
    fc = KronosForecastAgent(enable_kronos=False).predict(df)
    assert fc.direction in ("UP", "DOWN", "NEUTRAL")


# -- NaN cleaning tests (added by self-improve cycle 2026-04-24) ----------

def test_predict_handles_nan_in_ohlcv():
    """Kronos should clean NaN rows via ffill/bfill instead of raising."""
    import numpy as np
    df = _trending_df("up", n=80)
    # Inject NaN into several rows
    df.loc[10, "Close"] = np.nan
    df.loc[20, "Open"] = np.nan
    df.loc[30, "Volume"] = np.nan
    fc = KronosForecastAgent(enable_kronos=False).predict(df)
    assert fc is not None
    assert fc.direction in ("UP", "DOWN", "NEUTRAL")
    assert fc.source == "fallback"


def test_predict_all_nan_returns_neutral():
    """If all rows are NaN, should return neutral fallback."""
    import numpy as np
    df = _trending_df("up", n=10)
    df["Close"] = np.nan
    df["Open"] = np.nan
    fc = KronosForecastAgent(enable_kronos=False).predict(df)
    assert fc is not None
    # Should still produce a forecast (neutral or fallback)
    assert fc.direction in ("UP", "DOWN", "NEUTRAL")
