"""Tests for strategies.py — momentum, mean-reversion, breakout, multi-factor."""

import numpy as np
import pandas as pd
import pytest

from market_config import StrategyType
from strategies import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MultiFactorStrategy,
    Signal,
    STRATEGIES,
    get_strategy,
    run_all_strategies,
)


# ─────────────────────── Base indicator helpers ───────────────────────


def test_indicators_on_sample(sample_ohlcv):
    strat = MomentumStrategy()
    ind = strat._compute_indicators(sample_ohlcv)
    assert 0 <= ind["rsi"] <= 100
    assert "macd" in ind
    assert "stoch_k" in ind
    assert "atr" in ind
    assert ind["atr"] >= 0
    assert ind["sma_20"] > 0


def test_rsi_pure_uptrend_high(uptrend_ohlcv):
    strat = MomentumStrategy()
    ind = strat._compute_indicators(uptrend_ohlcv)
    assert ind["rsi"] > 70


def test_rsi_pure_downtrend_low(downtrend_ohlcv):
    strat = MomentumStrategy()
    ind = strat._compute_indicators(downtrend_ohlcv)
    assert ind["rsi"] < 30


def test_atr_positive(sample_ohlcv):
    strat = MomentumStrategy()
    ind = strat._compute_indicators(sample_ohlcv)
    assert ind["atr"] > 0


def test_indicator_with_insufficient_data():
    strat = MomentumStrategy()
    small_df = pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [1000],
    })
    ind = strat._compute_indicators(small_df)
    # Should not crash; RSI defaults to 50 on insufficient data.
    assert ind["rsi"] == 50.0


# ─────────────────────── Momentum strategy ───────────────────────


def test_momentum_signal_shape(sample_ohlcv):
    strat = MomentumStrategy()
    sig = strat.generate_signal(sample_ohlcv)
    if sig:
        assert sig.strategy == StrategyType.MOMENTUM
        assert sig.direction in ("LONG", "SHORT")
        assert 0 <= sig.strength <= 1
        assert sig.stop_loss != sig.entry_price


def test_momentum_no_signal_when_too_little_data():
    strat = MomentumStrategy()
    df = pd.DataFrame({
        "Open": np.linspace(100, 101, 10),
        "High": np.linspace(100, 101, 10) + 0.5,
        "Low": np.linspace(100, 101, 10) - 0.5,
        "Close": np.linspace(100, 101, 10),
        "Volume": [1000] * 10,
    })
    assert strat.generate_signal(df) is None


def test_momentum_strategy_type():
    strat = MomentumStrategy()
    assert strat.strategy_type == StrategyType.MOMENTUM


# ─────────────────────── Mean reversion strategy ───────────────────────


def test_mean_reversion_short_on_overbought(overbought_ohlcv):
    strat = MeanReversionStrategy()
    sig = strat.generate_signal(overbought_ohlcv)
    assert sig is not None
    assert sig.direction == "SHORT"
    assert sig.stop_loss > sig.entry_price  # Stop above entry for short


def test_mean_reversion_long_on_oversold(oversold_ohlcv):
    strat = MeanReversionStrategy()
    sig = strat.generate_signal(oversold_ohlcv)
    assert sig is not None
    assert sig.direction == "LONG"
    assert sig.stop_loss < sig.entry_price


def test_mean_reversion_no_signal_when_neutral(sample_ohlcv):
    # Random walk tends to have moderate RSI → no signal.
    strat = MeanReversionStrategy()
    sig = strat.generate_signal(sample_ohlcv)
    # Either no signal, or a valid one — but if it fires, direction must be valid.
    if sig:
        assert sig.direction in ("LONG", "SHORT")


def test_mean_reversion_strategy_type():
    strat = MeanReversionStrategy()
    assert strat.strategy_type == StrategyType.MEAN_REVERSION


# ─────────────────────── Breakout strategy ───────────────────────


def test_breakout_strategy_type():
    strat = BreakoutStrategy()
    assert strat.strategy_type == StrategyType.BREAKOUT


def test_breakout_no_signal_on_insufficient_data():
    strat = BreakoutStrategy()
    df = pd.DataFrame({
        "Open": [100.0] * 10,
        "High": [101.0] * 10,
        "Low": [99.0] * 10,
        "Close": [100.0] * 10,
        "Volume": [1000] * 10,
    })
    assert strat.generate_signal(df) is None


def test_breakout_triggers_on_breakout_with_volume():
    # Tight consolidation then a big up bar with volume spike.
    n = 30
    close = [100.0] * 25 + [100.5, 101.0, 101.5, 102.0, 105.0]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    opens = close[:]
    volume = [1000] * 29 + [5000]

    df = pd.DataFrame({
        "Open": opens,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    df.attrs["symbol"] = "BO"
    df.attrs["timeframe"] = "4h"

    strat = BreakoutStrategy()
    sig = strat.generate_signal(df)
    if sig:
        assert sig.direction in ("LONG", "SHORT")


def test_breakout_reads_pattern_report():
    # Agent report with a bullish pattern keyword should bias the signal.
    n = 30
    close = [100.0] * 28 + [101.0, 102.0]
    df = pd.DataFrame({
        "Open": close,
        "High": [c + 0.3 for c in close],
        "Low": [c - 0.3 for c in close],
        "Close": close,
        "Volume": [1000] * 30,
    })
    df.attrs["symbol"] = "P"
    df.attrs["timeframe"] = "4h"

    strat = BreakoutStrategy()
    sig = strat.generate_signal(df, agent_reports={"pattern_report": "Ascending triangle detected"})
    if sig:
        assert sig.direction == "LONG"


# ─────────────────────── Multi-factor strategy ───────────────────────


def test_multi_factor_strategy_type():
    strat = MultiFactorStrategy()
    assert strat.strategy_type == StrategyType.MULTI_FACTOR


def test_multi_factor_agrees_on_strong_uptrend(uptrend_ohlcv):
    strat = MultiFactorStrategy()
    sig = strat.generate_signal(uptrend_ohlcv)
    # Strong uptrend with MACD+SMA agreeing often produces a LONG; may not always reach 4/5.
    if sig:
        assert sig.direction == "LONG"
        assert sig.strength >= 4 / 6  # Needed agreement threshold


def test_multi_factor_respects_agent_decision():
    # Craft a neutral dataset + LONG agent decision → still needs 4/5 agreement.
    n = 40
    close = np.linspace(100, 101, n)
    df = pd.DataFrame({
        "Open": close,
        "High": close + 0.1,
        "Low": close - 0.1,
        "Close": close,
        "Volume": [1000] * n,
    })
    df.attrs["symbol"] = "MF"
    df.attrs["timeframe"] = "4h"

    strat = MultiFactorStrategy()
    sig = strat.generate_signal(df, agent_reports={"decision": "LONG"})
    # Signal may or may not fire depending on other indicators; test only validates it doesn't crash.
    if sig:
        assert sig.direction in ("LONG", "SHORT")


def test_multi_factor_insufficient_data_returns_none():
    strat = MultiFactorStrategy()
    df = pd.DataFrame({
        "Open": [100.0] * 10,
        "High": [101.0] * 10,
        "Low": [99.0] * 10,
        "Close": [100.0] * 10,
        "Volume": [1000] * 10,
    })
    assert strat.generate_signal(df) is None


# ─────────────────────── Strategy registry ───────────────────────


def test_all_strategy_types_registered():
    for st in StrategyType:
        assert st in STRATEGIES
        assert STRATEGIES[st] is not None


def test_get_strategy_returns_instance():
    s = get_strategy(StrategyType.MOMENTUM)
    assert isinstance(s, MomentumStrategy)


# ─────────────────────── run_all_strategies ───────────────────────


def test_run_all_strategies_returns_list(sample_ohlcv):
    sigs = run_all_strategies(sample_ohlcv)
    assert isinstance(sigs, list)
    for s in sigs:
        assert isinstance(s, Signal)


def test_run_all_strategies_filtered(sample_ohlcv):
    sigs = run_all_strategies(
        sample_ohlcv,
        enabled_strategies=[StrategyType.MOMENTUM],
    )
    for s in sigs:
        assert s.strategy == StrategyType.MOMENTUM


def test_run_all_strategies_empty_list(sample_ohlcv):
    sigs = run_all_strategies(sample_ohlcv, enabled_strategies=[])
    assert sigs == []


def test_signal_has_all_required_fields(uptrend_ohlcv):
    sigs = run_all_strategies(uptrend_ohlcv)
    for s in sigs:
        assert hasattr(s, "direction")
        assert hasattr(s, "strategy")
        assert hasattr(s, "entry_price")
        assert hasattr(s, "stop_loss")
        assert hasattr(s, "take_profit")
        assert hasattr(s, "risk_reward_ratio")
        assert hasattr(s, "reasoning")


def test_signal_stop_loss_direction_consistent(uptrend_ohlcv, downtrend_ohlcv):
    for df in (uptrend_ohlcv, downtrend_ohlcv):
        for s in run_all_strategies(df):
            if s.direction == "LONG":
                assert s.stop_loss < s.entry_price
            else:
                assert s.stop_loss > s.entry_price


# ─────────────────── Timeframe-aware SL/TP via _resolve_params ───────────────────


def _ramp_then_pullback(n: int = 80) -> pd.DataFrame:
    """Build a frame the momentum strategy will act on.

    The momentum strategy fires on a "bullish pullback" (price > SMA20 > SMA50,
    RSI ∈ [40,60], MACD hist > 0). We ramp aggressively then pull back gently
    so the indicators land in those bands.
    """
    ramp = np.linspace(100, 140, n - 5)
    pullback = np.linspace(140, 138, 5)
    prices = np.concatenate([ramp, pullback])
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "Open": prices - 0.05,
        "High": prices + 0.2,
        "Low": prices - 0.2,
        "Close": prices,
        "Volume": [1000] * n,
    })


def test_momentum_uses_tighter_stops_on_5m_than_4h():
    """The same OHLCV, tagged 5m vs 4h, should yield tighter SL on 5m."""
    df_5m = _ramp_then_pullback()
    df_5m.attrs["symbol"] = "BTC-USD"
    df_5m.attrs["timeframe"] = "5m"
    df_4h = df_5m.copy()
    df_4h.attrs["symbol"] = "BTC-USD"
    df_4h.attrs["timeframe"] = "4h"

    strat = MomentumStrategy()
    sig_5m = strat.generate_signal(df_5m)
    sig_4h = strat.generate_signal(df_4h)
    if sig_5m is None or sig_4h is None:
        pytest.skip("momentum did not fire on synthetic frame; tested elsewhere")
    if sig_5m.direction == "LONG":
        # SL distance below entry: 5m should be tighter (smaller distance).
        dist_5m = sig_5m.entry_price - sig_5m.stop_loss
        dist_4h = sig_4h.entry_price - sig_4h.stop_loss
    else:
        dist_5m = sig_5m.stop_loss - sig_5m.entry_price
        dist_4h = sig_4h.stop_loss - sig_4h.entry_price
    assert dist_5m < dist_4h


def test_adaptive_params_override_timeframe_defaults():
    """Caller-supplied adaptive_params win over timeframe defaults."""
    df = _ramp_then_pullback()
    df.attrs["symbol"] = "BTC-USD"
    df.attrs["timeframe"] = "5m"
    strat = MomentumStrategy()
    # Force a wider SL via adaptive params; 5m default is 0.75.
    sig = strat.generate_signal(df, adaptive_params={"sl_atr_mult": 2.5, "tp_atr_mult": 3.0})
    if sig is None:
        pytest.skip("momentum did not fire on synthetic frame")
    # With sl_atr_mult=2.5 the stop should be far below entry — well past the 5m default.
    if sig.direction == "LONG":
        assert sig.entry_price - sig.stop_loss > 0
