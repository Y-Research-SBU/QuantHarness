"""Tests for the Kronos-powered strategies in strategies.py.

Each strategy accepts a ``kronos_runner`` callable for dependency injection.
We pass deterministic stubs so tests don't depend on the real model.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from kronos_agent import KronosForecast
from market_config import StrategyType
from strategies import (
    KronosDivergenceStrategy,
    KronosMomentumConfirmStrategy,
    MultiTimeframeKronosStrategy,
    Signal,
    _extract_kronos_forecast,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _df(direction: str = "up", n: int = 200) -> pd.DataFrame:
    if direction == "up":
        prices = np.linspace(100.0, 130.0, n)
    elif direction == "down":
        prices = np.linspace(130.0, 100.0, n)
    elif direction == "rangebound":
        # Random walk centred at 100; deterministic seed so RSI stays moderate.
        rng = np.random.default_rng(123)
        steps = rng.standard_normal(n) * 0.4
        prices = 100.0 + np.cumsum(steps - np.mean(steps))
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


def _forecast(direction: str, magnitude_pct: float, confidence: float = 0.7) -> KronosForecast:
    return KronosForecast(
        direction=direction,
        magnitude_pct=magnitude_pct,
        confidence=confidence,
        predicted_close=100.0 * (1 + magnitude_pct / 100.0),
        predicted_high=110.0,
        predicted_low=90.0,
        horizon=24,
        last_close=100.0,
        source="stub",
        reasoning="stub forecast",
    )


def _runner_returning(forecast: KronosForecast):
    def _r(_df_arg, *args, **kwargs):
        return forecast
    return _r


# ---------------------------------------------------------------------------
# _extract_kronos_forecast helper
# ---------------------------------------------------------------------------


def test_extract_kronos_forecast_handles_none():
    assert _extract_kronos_forecast(None) is None


def test_extract_kronos_forecast_finds_value():
    payload = {"kronos_forecast_data": {"direction": "UP", "magnitude_pct": 1.0, "confidence": 0.5}}
    out = _extract_kronos_forecast(payload)
    assert out["direction"] == "UP"


def test_extract_kronos_forecast_ignores_unrelated_keys():
    assert _extract_kronos_forecast({"indicator_report": "neutral"}) is None


# ---------------------------------------------------------------------------
# KronosMomentumConfirmStrategy
# ---------------------------------------------------------------------------


def test_kronos_momentum_confirm_long_signal():
    df = _df("rangebound")
    runner = _runner_returning(_forecast("UP", magnitude_pct=3.0, confidence=0.8))
    strat = KronosMomentumConfirmStrategy(min_pct=2.0, kronos_runner=runner)
    sig = strat.generate_signal(df)
    assert sig is not None
    assert sig.direction == "LONG"
    assert sig.strategy == StrategyType.KRONOS_MOMENTUM_CONFIRM
    assert 0.0 < sig.strength <= 1.0
    assert sig.entry_price == pytest.approx(float(df["Close"].iloc[-1]))


def test_kronos_momentum_confirm_short_signal():
    df = _df("rangebound")
    runner = _runner_returning(_forecast("DOWN", magnitude_pct=-3.0, confidence=0.8))
    strat = KronosMomentumConfirmStrategy(min_pct=2.0, kronos_runner=runner)
    sig = strat.generate_signal(df)
    assert sig is not None
    assert sig.direction == "SHORT"


def test_kronos_momentum_confirm_skips_below_min_pct():
    df = _df("rangebound")
    runner = _runner_returning(_forecast("UP", magnitude_pct=0.5, confidence=0.8))
    strat = KronosMomentumConfirmStrategy(min_pct=2.0, kronos_runner=runner)
    assert strat.generate_signal(df) is None


def test_kronos_momentum_confirm_skips_low_confidence():
    df = _df("rangebound")
    runner = _runner_returning(_forecast("UP", magnitude_pct=4.0, confidence=0.05))
    strat = KronosMomentumConfirmStrategy(min_confidence=0.3, kronos_runner=runner)
    assert strat.generate_signal(df) is None


def test_kronos_momentum_confirm_pattern_bearish_blocks_long():
    df = _df("rangebound")
    runner = _runner_returning(_forecast("UP", magnitude_pct=3.0, confidence=0.8))
    strat = KronosMomentumConfirmStrategy(kronos_runner=runner)
    sig = strat.generate_signal(df, agent_reports={"pattern_report": "Clear bearish head and shoulders breakdown"})
    assert sig is None


def test_kronos_momentum_confirm_uses_cached_forecast(monkeypatch):
    df = _df("rangebound")
    # No runner — strategy should pick up forecast from agent_reports.
    strat = KronosMomentumConfirmStrategy(min_pct=1.0)
    monkeypatch.setattr(
        "strategies._get_shared_kronos_agent",
        lambda: pytest.fail("should not load shared agent"),
    )
    cached = _forecast("UP", magnitude_pct=2.5, confidence=0.7).to_dict()
    sig = strat.generate_signal(df, agent_reports={"kronos_forecast_data": cached})
    assert sig is not None
    assert sig.direction == "LONG"


def test_kronos_momentum_returns_none_when_runner_fails():
    def boom(*_a, **_k):
        raise RuntimeError("oh no")
    df = _df("rangebound")
    strat = KronosMomentumConfirmStrategy(kronos_runner=boom)
    assert strat.generate_signal(df) is None


# ---------------------------------------------------------------------------
# KronosDivergenceStrategy
# ---------------------------------------------------------------------------


def test_kronos_divergence_short_in_uptrend():
    df = _df("up")
    runner = _runner_returning(_forecast("DOWN", magnitude_pct=-2.0, confidence=0.7))
    strat = KronosDivergenceStrategy(min_pct=1.0, kronos_runner=runner)
    sig = strat.generate_signal(df)
    assert sig is not None
    assert sig.direction == "SHORT"
    assert sig.strategy == StrategyType.KRONOS_DIVERGENCE


def test_kronos_divergence_long_in_downtrend():
    df = _df("down")
    runner = _runner_returning(_forecast("UP", magnitude_pct=2.0, confidence=0.7))
    strat = KronosDivergenceStrategy(min_pct=1.0, kronos_runner=runner)
    sig = strat.generate_signal(df)
    assert sig is not None
    assert sig.direction == "LONG"


def test_kronos_divergence_skips_aligned_kronos():
    df = _df("up")
    # Kronos agrees with uptrend → no divergence trade.
    runner = _runner_returning(_forecast("UP", magnitude_pct=2.0, confidence=0.7))
    strat = KronosDivergenceStrategy(kronos_runner=runner)
    assert strat.generate_signal(df) is None


def test_kronos_divergence_skips_low_confidence():
    df = _df("up")
    runner = _runner_returning(_forecast("DOWN", magnitude_pct=-3.0, confidence=0.1))
    strat = KronosDivergenceStrategy(min_confidence=0.4, kronos_runner=runner)
    assert strat.generate_signal(df) is None


def test_kronos_divergence_skips_when_no_clear_trend():
    df = _df("rangebound")
    runner = _runner_returning(_forecast("DOWN", magnitude_pct=-2.0, confidence=0.7))
    strat = KronosDivergenceStrategy(kronos_runner=runner)
    assert strat.generate_signal(df) is None


def test_kronos_divergence_runner_failure_returns_none():
    def boom(*_a, **_k):
        raise RuntimeError("boom")
    strat = KronosDivergenceStrategy(kronos_runner=boom)
    assert strat.generate_signal(_df("up")) is None


# ---------------------------------------------------------------------------
# MultiTimeframeKronosStrategy
# ---------------------------------------------------------------------------


def test_multi_timeframe_long_when_all_horizons_agree():
    df = _df("rangebound")

    def runner(_df_arg, horizon, *args, **kwargs):
        return _forecast("UP", magnitude_pct=2.0 + horizon * 0.05, confidence=0.6)

    strat = MultiTimeframeKronosStrategy(horizons=(6, 12, 24), kronos_runner=runner)
    sig = strat.generate_signal(df)
    assert sig is not None
    assert sig.direction == "LONG"
    assert sig.strategy == StrategyType.MULTI_TIMEFRAME_KRONOS


def test_multi_timeframe_short_when_all_horizons_agree():
    df = _df("rangebound")

    def runner(_df_arg, horizon, *args, **kwargs):
        return _forecast("DOWN", magnitude_pct=-2.0 - horizon * 0.05, confidence=0.6)

    strat = MultiTimeframeKronosStrategy(horizons=(6, 12, 24), kronos_runner=runner)
    sig = strat.generate_signal(df)
    assert sig is not None
    assert sig.direction == "SHORT"


def test_multi_timeframe_skips_when_horizons_disagree():
    df = _df("rangebound")
    forecasts = iter([
        _forecast("UP", magnitude_pct=2.0, confidence=0.7),
        _forecast("DOWN", magnitude_pct=-2.0, confidence=0.7),
        _forecast("UP", magnitude_pct=2.0, confidence=0.7),
    ])

    def runner(_df_arg, horizon, *args, **kwargs):
        return next(forecasts)

    strat = MultiTimeframeKronosStrategy(horizons=(6, 12, 24), kronos_runner=runner)
    assert strat.generate_signal(df) is None


def test_multi_timeframe_skips_when_one_low_confidence():
    df = _df("rangebound")
    forecasts = iter([
        _forecast("UP", magnitude_pct=2.0, confidence=0.7),
        _forecast("UP", magnitude_pct=2.0, confidence=0.05),
        _forecast("UP", magnitude_pct=2.0, confidence=0.7),
    ])

    def runner(_df_arg, horizon, *args, **kwargs):
        return next(forecasts)

    strat = MultiTimeframeKronosStrategy(
        horizons=(6, 12, 24), min_confidence=0.25, kronos_runner=runner
    )
    assert strat.generate_signal(df) is None


def test_multi_timeframe_short_history_returns_none():
    df = _df("rangebound", n=20)
    runner = _runner_returning(_forecast("UP", magnitude_pct=2.0, confidence=0.7))
    strat = MultiTimeframeKronosStrategy(kronos_runner=runner)
    assert strat.generate_signal(df) is None


def test_multi_timeframe_handles_runner_returning_none():
    df = _df("rangebound")

    def runner(_df_arg, horizon, *args, **kwargs):
        return None

    strat = MultiTimeframeKronosStrategy(kronos_runner=runner)
    assert strat.generate_signal(df) is None


def test_multi_timeframe_signal_has_metadata():
    df = _df("rangebound")
    runner = _runner_returning(_forecast("UP", magnitude_pct=2.0, confidence=0.6))
    strat = MultiTimeframeKronosStrategy(horizons=(6, 12), kronos_runner=runner)
    sig = strat.generate_signal(df)
    assert sig is not None
    assert "kronos_horizons" in sig.metadata
    assert sig.metadata["kronos_horizons"] == [6, 12]
    assert len(sig.metadata["kronos_forecasts"]) == 2


# ---------------------------------------------------------------------------
# Integration: registry & run_all_strategies
# ---------------------------------------------------------------------------


def test_kronos_strategies_registered():
    from strategies import STRATEGIES
    for st in (
        StrategyType.KRONOS_MOMENTUM_CONFIRM,
        StrategyType.KRONOS_DIVERGENCE,
        StrategyType.MULTI_TIMEFRAME_KRONOS,
    ):
        assert st in STRATEGIES
        assert STRATEGIES[st].strategy_type == st


def test_run_all_strategies_includes_kronos_signals():
    from strategies import run_all_strategies, STRATEGIES, KronosMomentumConfirmStrategy

    df = _df("rangebound")
    # Replace the registered momentum-confirm strategy with one carrying our stub runner.
    original = STRATEGIES[StrategyType.KRONOS_MOMENTUM_CONFIRM]
    STRATEGIES[StrategyType.KRONOS_MOMENTUM_CONFIRM] = KronosMomentumConfirmStrategy(
        min_pct=2.0,
        kronos_runner=_runner_returning(_forecast("UP", magnitude_pct=3.0, confidence=0.8)),
    )
    try:
        signals = run_all_strategies(
            df,
            enabled_strategies=[StrategyType.KRONOS_MOMENTUM_CONFIRM],
        )
    finally:
        STRATEGIES[StrategyType.KRONOS_MOMENTUM_CONFIRM] = original
    assert len(signals) == 1
    assert signals[0].direction == "LONG"
    assert signals[0].strategy == StrategyType.KRONOS_MOMENTUM_CONFIRM
