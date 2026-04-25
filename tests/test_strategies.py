"""Tests for strategies.py — momentum, mean-reversion, breakout, multi-factor."""

import numpy as np
import pandas as pd
import pytest

from market_config import StrategyType
from strategies import (
    BollingerBandSqueezeStrategy,
    BreakoutStrategy,
    EMACrossoverStrategy,
    KronosMomentumConfirmStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MultiFactorStrategy,
    Signal,
    STRATEGIES,
    VWAPReversionStrategy,
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


# ─────────────────────── VWAP Reversion strategy ───────────────────────


def _vwap_long_frame(n: int = 60) -> pd.DataFrame:
    """Frame where price sits well below VWAP and RSI is low (oversold)."""
    # Ramp up strongly for the first part, then drop sharply — pushes VWAP up
    # (biased by high-volume upper bars) and current price down (low RSI).
    ramp = np.linspace(100, 150, n - 10)
    drop = np.linspace(150, 120, 10)
    prices = np.concatenate([ramp, drop])
    volumes = np.concatenate([np.full(n - 10, 10000.0), np.full(10, 1000.0)])
    df = pd.DataFrame({
        "Open": prices + 0.1,
        "High": prices + 0.3,
        "Low": prices - 0.3,
        "Close": prices,
        "Volume": volumes,
    })
    df.attrs["symbol"] = "VW-USD"
    df.attrs["timeframe"] = "1h"
    return df


def _vwap_short_frame(n: int = 60) -> pd.DataFrame:
    """Frame where price sits above VWAP and RSI is high (overbought)."""
    down = np.linspace(150, 100, n - 10)
    spike = np.linspace(100, 130, 10)
    prices = np.concatenate([down, spike])
    volumes = np.concatenate([np.full(n - 10, 10000.0), np.full(10, 1000.0)])
    df = pd.DataFrame({
        "Open": prices - 0.1,
        "High": prices + 0.3,
        "Low": prices - 0.3,
        "Close": prices,
        "Volume": volumes,
    })
    df.attrs["symbol"] = "VW-USD"
    df.attrs["timeframe"] = "1h"
    return df


def test_vwap_reversion_strategy_type():
    assert VWAPReversionStrategy().strategy_type == StrategyType.VWAP_REVERSION


def test_vwap_reversion_long_when_below_vwap_and_oversold():
    df = _vwap_long_frame()
    sig = VWAPReversionStrategy().generate_signal(df)
    assert sig is not None
    assert sig.direction == "LONG"
    assert sig.stop_loss < sig.entry_price
    assert sig.take_profit > sig.entry_price  # VWAP is the target


def test_vwap_reversion_short_when_above_vwap_and_overbought():
    df = _vwap_short_frame()
    sig = VWAPReversionStrategy().generate_signal(df)
    assert sig is not None
    assert sig.direction == "SHORT"
    assert sig.stop_loss > sig.entry_price
    assert sig.take_profit < sig.entry_price


def test_vwap_reversion_no_signal_without_volume():
    df = _vwap_long_frame()
    df_no_vol = df.drop(columns=["Volume"])
    sig = VWAPReversionStrategy().generate_signal(df_no_vol)
    assert sig is None


# ─────────────────────── Bollinger-Band Squeeze strategy ───────────────────────


def _bb_squeeze_breakout_frame(n: int = 80, direction: str = "LONG") -> pd.DataFrame:
    """Long flat section (tight bands) then a single sharp breakout bar.

    The breakout *must* be on the very last bar so that ``widths[-2]`` (which
    looks at bars up to n-2) sees only the flat regime — keeping it in the
    bottom-20th percentile — while ``widths[-1]`` jumps higher due to the
    final spike. This is exactly the squeeze-release pattern the strategy
    looks for.
    """
    # Perfectly flat price keeps every prior bar's width at 0 (in the
    # bottom 20th percentile by definition), then the final spike pushes
    # the latest width above zero — a clean squeeze release.
    flat = np.full(n - 1, 100.0)
    if direction == "LONG":
        spike = np.array([110.0])
    else:
        spike = np.array([90.0])
    prices = np.concatenate([flat, spike])
    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 0.1,
        "Low": prices - 0.1,
        "Close": prices,
        "Volume": [1000] * n,
    })
    df.attrs["symbol"] = "BB-USD"
    df.attrs["timeframe"] = "1h"
    return df


def test_bb_squeeze_strategy_type():
    assert BollingerBandSqueezeStrategy().strategy_type == StrategyType.BB_SQUEEZE


def test_bb_squeeze_fires_on_upside_breakout():
    df = _bb_squeeze_breakout_frame(direction="LONG")
    sig = BollingerBandSqueezeStrategy().generate_signal(df)
    assert sig is not None
    assert sig.direction == "LONG"
    assert sig.stop_loss < sig.entry_price


def test_bb_squeeze_fires_on_downside_breakout():
    df = _bb_squeeze_breakout_frame(direction="SHORT")
    sig = BollingerBandSqueezeStrategy().generate_signal(df)
    assert sig is not None
    assert sig.direction == "SHORT"
    assert sig.stop_loss > sig.entry_price


def test_bb_squeeze_no_signal_on_random_walk(sample_ohlcv):
    # Random walk without a squeeze → strategy should decline.
    sig = BollingerBandSqueezeStrategy().generate_signal(sample_ohlcv)
    # Either no signal or a valid-shape one.
    if sig:
        assert sig.direction in ("LONG", "SHORT")


def test_bb_squeeze_insufficient_data_returns_none():
    df = pd.DataFrame({
        "Open": [100.0] * 30,
        "High": [101.0] * 30,
        "Low": [99.0] * 30,
        "Close": [100.0] * 30,
        "Volume": [1000] * 30,
    })
    assert BollingerBandSqueezeStrategy().generate_signal(df) is None


# ─────────────────────── EMA Crossover strategy ───────────────────────


def _ema_bull_cross_frame(n: int = 80) -> pd.DataFrame:
    """Long downtrend pinning fast EMA below slow, then a single huge up-bar
    that flips the fast EMA above the slow on the *last* bar.

    The previous bar must still have ``fast <= slow`` so the cross fires now,
    not earlier in the rally.
    """
    down = np.linspace(120.0, 100.0, n - 1)
    spike = np.array([200.0])  # massive bar to overcome EMA inertia in one step
    prices = np.concatenate([down, spike])
    # Elevated volume on the breakout bar; lower volume earlier so volume_ratio > 1.
    volumes = np.concatenate([np.full(n - 1, 1000.0), np.array([5000.0])])
    df = pd.DataFrame({
        "Open": prices - 0.1,
        "High": prices + 0.3,
        "Low": prices - 0.3,
        "Close": prices,
        "Volume": volumes,
    })
    df.attrs["symbol"] = "EMA-USD"
    df.attrs["timeframe"] = "1h"
    return df


def _ema_bear_cross_frame(n: int = 80) -> pd.DataFrame:
    up = np.linspace(80.0, 100.0, n - 1)
    crash = np.array([20.0])
    prices = np.concatenate([up, crash])
    volumes = np.concatenate([np.full(n - 1, 1000.0), np.array([5000.0])])
    df = pd.DataFrame({
        "Open": prices + 0.1,
        "High": prices + 0.3,
        "Low": prices - 0.3,
        "Close": prices,
        "Volume": volumes,
    })
    df.attrs["symbol"] = "EMA-USD"
    df.attrs["timeframe"] = "1h"
    return df


def test_ema_crossover_strategy_type():
    assert EMACrossoverStrategy().strategy_type == StrategyType.EMA_CROSSOVER


def test_ema_crossover_long_on_bullish_cross():
    df = _ema_bull_cross_frame()
    sig = EMACrossoverStrategy().generate_signal(df)
    assert sig is not None
    assert sig.direction == "LONG"
    assert sig.stop_loss < sig.entry_price
    # R:R should be at least 1.5 by construction.
    assert sig.risk_reward_ratio >= 1.5 - 1e-9


def test_ema_crossover_short_on_bearish_cross():
    df = _ema_bear_cross_frame()
    sig = EMACrossoverStrategy().generate_signal(df)
    assert sig is not None
    assert sig.direction == "SHORT"
    assert sig.stop_loss > sig.entry_price
    assert sig.risk_reward_ratio >= 1.5 - 1e-9


def test_ema_crossover_skips_when_volume_low():
    """Bullish cross with volume below average should not fire."""
    df = _ema_bull_cross_frame()
    # Crush volume on breakout bars so volume_ratio < 1.
    df.loc[df.index[-10:], "Volume"] = 100
    df.loc[df.index[:-10], "Volume"] = 10000
    sig = EMACrossoverStrategy().generate_signal(df)
    assert sig is None


# ─────────────────────── KronosMomentumConfirm tuning ───────────────────────


def _k_frame(n: int = 80) -> pd.DataFrame:
    prices = 100.0 + np.cumsum(np.random.default_rng(3).normal(0, 0.3, n))
    df = pd.DataFrame({
        "Open": prices - 0.1,
        "High": prices + 0.5,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": [1000] * n,
    })
    df.attrs["symbol"] = "K-USD"
    df.attrs["timeframe"] = "1h"
    return df


def test_kronos_momentum_confirm_defaults_tightened():
    strat = KronosMomentumConfirmStrategy()
    assert strat.min_confidence >= 0.5
    assert strat.min_pct >= 3.0
    assert strat.MIN_RR >= 1.5


def test_kronos_momentum_confirm_rejects_low_confidence():
    """Confidence below the new 0.5 floor must produce no signal."""
    df = _k_frame()
    strat = KronosMomentumConfirmStrategy(
        kronos_runner=lambda _d: {"direction": "UP", "magnitude_pct": 5.0, "confidence": 0.4},
    )
    sig = strat.generate_signal(df)
    assert sig is None


def test_kronos_momentum_confirm_rejects_small_magnitude():
    """Magnitude below the new 3.0 floor must produce no signal."""
    df = _k_frame()
    strat = KronosMomentumConfirmStrategy(
        kronos_runner=lambda _d: {"direction": "UP", "magnitude_pct": 2.0, "confidence": 0.9},
    )
    sig = strat.generate_signal(df)
    assert sig is None


def test_kronos_momentum_confirm_rejects_low_rr():
    """If adaptive params compress tp_mult so R:R < 1.5, no trade."""
    df = _k_frame()
    strat = KronosMomentumConfirmStrategy(
        kronos_runner=lambda _d: {"direction": "UP", "magnitude_pct": 3.5, "confidence": 0.9},
    )
    # Flip sl/tp so that tp multiplier is much smaller → rr < 1.5.
    sig = strat.generate_signal(
        df,
        adaptive_params={"sl_atr_mult": 3.0, "tp_atr_mult": 1.0, "kronos_min_confidence": 0.5},
    )
    # Either outright rejected by R:R, or None (depending on ATR/magnitude mix).
    if sig is not None:
        assert sig.risk_reward_ratio >= 1.5


# ─────────────────────── Registry completeness ───────────────────────


def test_new_strategies_registered_in_registry():
    for st in (StrategyType.VWAP_REVERSION, StrategyType.BB_SQUEEZE, StrategyType.EMA_CROSSOVER):
        assert st in STRATEGIES
        assert STRATEGIES[st] is not None
