"""
Tests for the self-improvement integration inside MarketScanner and
PaperTradingEngine.close_trade.

Covers:
  • L1: disabled strategies are excluded in execute_signals; weighted ranking
    actually reorders signals.
  • L3: execute_signals skips signals whose meta-model probability is below
    SIGNAL_QUALITY_THRESHOLD.
  • L4: scan_market attaches detected regime to signal metadata and logs it.
  • L5: scan_market logs Kronos predictions, and close_trade records the
    outcome against the logged prediction row.
  • adaptive params from the self-improver flow through run_all_strategies
    into MeanReversionStrategy (RSI thresholds).
  • SelfImprover failures inside MarketScanner never crash the trading loop.

These focus on behavior that spans multiple modules; pure-unit behavior of
each self-improvement submodule is covered in the existing
tests/test_self_improver.py, tests/test_adaptive_params.py, etc.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from market_config import (
    MARKETS,
    MarketCategory,
    MarketConfig,
    StrategyType,
)
from paper_trading import PaperTradingEngine
from position_sizing import PositionSizeResult
from scanner import SIGNAL_QUALITY_THRESHOLD, MarketScanner
from self_improver import WEIGHT_DISABLED, SelfImprover
from strategies import (
    MeanReversionStrategy,
    STRATEGIES,
    Signal,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _mk_signal(
    symbol: str = "BTC-USD",
    strategy: StrategyType = StrategyType.MOMENTUM,
    direction: str = "LONG",
    strength: float = 0.8,
    entry: float = 100.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Signal:
    return Signal(
        direction=direction,
        strength=strength,
        strategy=strategy,
        symbol=symbol,
        timeframe="1h",
        entry_price=entry,
        # Crypto floor (PaperTradingEngine.MIN_STOP_PCT['crypto']) is 3%;
        # use 4% so synthetic signals survive ``execute_trade`` validation.
        stop_loss=entry * (0.96 if direction == "LONG" else 1.04),
        take_profit=entry * (1.08 if direction == "LONG" else 0.92),
        risk_reward_ratio=2.0,
        reasoning="test",
        metadata=dict(metadata or {}),
    )


def _mk_position(size: float = 500.0, qty: float = 5.0) -> PositionSizeResult:
    # Stops widened to 4% to clear the 3% crypto floor in PaperTradingEngine.
    return PositionSizeResult(
        position_size_usd=size,
        quantity=qty,
        risk_per_trade_usd=size * 0.02,
        risk_pct=0.02,
        kelly_fraction=0.2,
        half_kelly=0.1,
        stop_loss=96.0,
        take_profit=108.0,
        reason="test",
    )


@pytest.fixture
def scanner(tmp_db_path) -> MarketScanner:
    return MarketScanner(db_path=tmp_db_path, use_kronos=False, use_agents=False)


# ---------------------------------------------------------------------------
# L1 — strategy weights + disabled strategies
# ---------------------------------------------------------------------------


class _FakeImprover:
    """Tiny self-improver double for deterministic scanner behavior."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        quality_map: Optional[Dict[str, float]] = None,
        regime_weight_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self.weights = weights or {}
        self.quality_map = quality_map or {}
        self.regime_weight_overrides = regime_weight_overrides or {}
        self.log_regime_calls: List[str] = []
        self.log_kronos_calls: List[Dict[str, Any]] = []
        self.record_kronos_calls: List[Dict[str, Any]] = []
        self.improvement_calls: List[Dict[str, Any]] = []

    # L1
    def get_strategy_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def get_disabled_strategies(self) -> List[str]:
        return [s for s, w in self.weights.items() if w == WEIGHT_DISABLED]

    # L2
    def get_adaptive_params(self, strategy: str, symbol: str) -> Dict[str, float]:
        return {}

    # L3
    def predict_signal_quality(self, features, strategy="", category="") -> float:
        return float(self.quality_map.get(strategy, 0.5))

    # L4
    def log_regime(self, symbol, df, timeframe=None) -> str:
        regime = "ranging"
        self.log_regime_calls.append(symbol)
        return regime

    def get_regime_adjusted_weight(self, strategy, regime) -> float:
        if strategy in self.regime_weight_overrides:
            return self.regime_weight_overrides[strategy]
        return self.weights.get(strategy, 1.0)

    # L5
    def log_kronos_prediction(self, **kwargs) -> int:
        self.log_kronos_calls.append(kwargs)
        return 42

    def get_kronos_confidence_adjustment(self, symbol) -> float:
        return 1.0

    def is_kronos_symbol_blocked(self, symbol) -> bool:
        return getattr(self, "blocked_symbols", set()).__contains__(symbol)

    def record_kronos_outcome(self, **kwargs) -> None:
        self.record_kronos_calls.append(kwargs)

    def evaluate_kronos_accuracy(self) -> Dict[str, Any]:
        return {"hit_rate": 0.0, "n": 0}

    # Orchestration
    def run_improvement_cycle(self, total_trades=None, symbol_bars=None) -> Dict[str, Any]:
        call = {"total_trades": total_trades, "symbol_bars": list((symbol_bars or {}).keys())}
        self.improvement_calls.append(call)
        return {"levels_run": ["L4"]}


def test_execute_signals_skips_disabled_strategy(scanner):
    """A strategy with weight==WEIGHT_DISABLED must be filtered out up-front."""
    fake = _FakeImprover(weights={
        StrategyType.MOMENTUM.value: WEIGHT_DISABLED,
        StrategyType.BREAKOUT.value: 1.0,
    })
    scanner.self_improver = fake

    # Both signals must clear MIN_SIGNAL_STRENGTH (0.4) so the test exercises
    # the disabled-strategy filter rather than the raw-strength floor.
    sig_momentum = _mk_signal(strategy=StrategyType.MOMENTUM, symbol="BTC-USD", strength=0.95)
    sig_breakout = _mk_signal(strategy=StrategyType.BREAKOUT, symbol="ETH-USD", strength=0.5)
    trade_ids = scanner.execute_signals([sig_momentum, sig_breakout])

    # Only the breakout signal should have made it to execute_trade.
    assert len(trade_ids) == 1
    open_positions = scanner.engine.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0]["symbol"] == "ETH-USD"
    assert open_positions[0]["strategy"] == StrategyType.BREAKOUT.value


def test_execute_signals_weights_reorder_ranking(scanner):
    """A higher weight must promote a weaker signal above a stronger one."""
    # Only allow one concurrent position so ranking order determines the winner.
    scanner.engine.MAX_POSITIONS = 1

    fake = _FakeImprover(weights={
        StrategyType.MOMENTUM.value: 0.5,     # reduced weight
        StrategyType.BREAKOUT.value: 2.0,     # high-sharpe boost
    })
    scanner.self_improver = fake

    # Momentum raw strength is higher, but breakout gets 2x weight.
    sig_momentum = _mk_signal(strategy=StrategyType.MOMENTUM, symbol="BTC-USD", strength=0.7)
    sig_breakout = _mk_signal(strategy=StrategyType.BREAKOUT, symbol="ETH-USD", strength=0.5)

    trade_ids = scanner.execute_signals([sig_momentum, sig_breakout])

    assert len(trade_ids) == 1
    open_positions = scanner.engine.get_open_positions()
    assert open_positions[0]["strategy"] == StrategyType.BREAKOUT.value


# ---------------------------------------------------------------------------
# L3 — signal-quality filtering
# ---------------------------------------------------------------------------


def test_execute_signals_filters_below_quality_threshold(scanner):
    """A signal whose quality-model prob is below the threshold must be skipped."""
    fake = _FakeImprover(quality_map={
        StrategyType.MOMENTUM.value: SIGNAL_QUALITY_THRESHOLD - 0.05,  # below threshold
        StrategyType.BREAKOUT.value: 0.9,                              # above threshold
    })
    scanner.self_improver = fake

    sig_bad = _mk_signal(strategy=StrategyType.MOMENTUM, symbol="BTC-USD", strength=0.9)
    sig_good = _mk_signal(strategy=StrategyType.BREAKOUT, symbol="ETH-USD", strength=0.5)

    trade_ids = scanner.execute_signals([sig_bad, sig_good])

    assert len(trade_ids) == 1
    positions = scanner.engine.get_open_positions()
    assert [p["symbol"] for p in positions] == ["ETH-USD"]
    # The quality score should be persisted on the chosen signal's metadata too.
    assert "signal_quality" in sig_good.metadata


# ---------------------------------------------------------------------------
# L4 — regime detection + affinity
# ---------------------------------------------------------------------------


def _fake_frame(n: int = 120, base: float = 100.0) -> pd.DataFrame:
    """Deterministic OHLCV frame large enough for regime + strategy checks."""
    rng = np.random.default_rng(7)
    prices = base + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame({
        "Datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h"),
        "Open": prices - 0.05,
        "High": prices + 0.5,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": rng.integers(1000, 5000, n),
    })
    return df


def test_scan_market_attaches_regime_to_signal_metadata(scanner, monkeypatch):
    """scan_market should call log_regime and propagate the result."""
    fake = _FakeImprover()
    scanner.self_improver = fake

    df = _fake_frame(n=150)
    monkeypatch.setattr("scanner.fetch_market_data", lambda symbol, tf, **kw: df.copy())

    # Bypass Kronos and agent paths; use momentum which generates a signal from
    # the df. We build a minimal config with only momentum enabled.
    config = MarketConfig(
        symbol="TEST-USD",
        display_name="Test",
        category=MarketCategory.CRYPTO,
        timeframes=["1h"],
        enabled_strategies=[StrategyType.MOMENTUM],
    )

    signals = scanner.scan_market("TEST-USD", config)
    assert "TEST-USD" in fake.log_regime_calls
    assert scanner._cycle_regimes.get("TEST-USD") == "ranging"
    # Even if the strategy returns no signal, the regime should be logged.
    for sig in signals:
        assert sig.metadata.get("regime") == "ranging"


# ---------------------------------------------------------------------------
# L5 — Kronos prediction logging + outcome recording
# ---------------------------------------------------------------------------


def test_run_kronos_forecast_logs_prediction(tmp_db_path, monkeypatch):
    """_run_kronos_forecast must attach a prediction_id + log to the DB."""
    scanner = MarketScanner(db_path=tmp_db_path, use_kronos=True, use_agents=False)

    # Stub out the expensive Kronos model with a fake that returns a forecast.
    fake_forecast = SimpleNamespace(
        to_dict=lambda: {
            "direction": "UP",
            "magnitude_pct": 2.0,
            "confidence": 0.8,
            "predicted_close": 102.0,
            "horizon": 12,
        },
    )
    stub_agent = SimpleNamespace(predict=lambda df, timeframe=None: fake_forecast)
    scanner._kronos_agent = stub_agent

    df = _fake_frame()
    data = scanner._run_kronos_forecast("BTC-USD", "1h", df)
    assert data is not None
    assert "prediction_id" in data
    # And the row should exist in kronos_predictions.
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM kronos_predictions").fetchone()["n"]
    assert n == 1


def test_run_kronos_forecast_zeros_confidence_for_blocked_symbol(tmp_db_path):
    """_run_kronos_forecast must zero confidence on symbols the improver blocks.

    Acts as a hard guard for chronically miscalibrated markets so downstream
    kronos_* strategies fall through their min_confidence gate and skip
    entirely.
    """
    scanner = MarketScanner(db_path=tmp_db_path, use_kronos=True, use_agents=False)

    fake_forecast = SimpleNamespace(
        to_dict=lambda: {
            "direction": "UP",
            "magnitude_pct": 2.0,
            "confidence": 0.85,
            "predicted_close": 102.0,
            "horizon": 12,
        },
    )
    scanner._kronos_agent = SimpleNamespace(predict=lambda df, timeframe=None: fake_forecast)

    fake = _FakeImprover()
    fake.blocked_symbols = {"DOOM-USD"}
    scanner.self_improver = fake

    df = _fake_frame()

    blocked = scanner._run_kronos_forecast("DOOM-USD", "1h", df)
    assert blocked is not None
    assert blocked["confidence"] == 0.0
    assert blocked.get("kronos_blocked") is True
    assert blocked.get("raw_confidence") == 0.85

    ok = scanner._run_kronos_forecast("BTC-USD", "1h", df)
    assert ok is not None
    assert ok["confidence"] == 0.85
    assert ok.get("kronos_blocked") is None


def test_close_trade_does_not_evaluate_kronos_immediately(tmp_db_path):
    """Closing a trade must NOT mark its Kronos prediction.

    REL-376 deliberately removed the close-time Kronos evaluation: the
    prediction is about price at ``prediction_time + horizon * step``,
    not the trade's exit price. Trades close at unrelated wall-clock
    times (SL/TP/orphan-cleanup), so reading exit_price as the "actual"
    locked predictions to NEUTRAL and broke accuracy reporting. Real
    evaluation now lives in ``evaluate_pending_kronos_predictions``.

    This regression test pins the new contract:
      1. ``close_trade`` leaves ``correct = NULL`` on the prediction.
      2. ``evaluate_pending_kronos_predictions`` resolves it later using
         the price at the correct future timestamp (here, injected via
         ``price_lookup``).
    """
    import pandas as pd
    from db_schema import get_connection

    improver = SelfImprover(db_path=tmp_db_path)
    pred_id = improver.log_kronos_prediction(
        symbol="BTC-USD", timeframe="1h", predicted_direction="UP",
        predicted_magnitude=2.0, confidence=0.8, horizon=12, predicted_price=102.0,
    )

    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(
        symbol="BTC-USD",
        metadata={"kronos_prediction_id": pred_id, "kronos_entry_price": 100.0},
    )
    tid = engine.execute_trade(sig, _mk_position())
    assert tid is not None

    # Close the trade. Per REL-376 this must NOT touch kronos_predictions.
    result = engine.close_trade(tid, exit_price=103.0, reason="manual")
    assert result is not None
    with get_connection(tmp_db_path) as conn:
        row = conn.execute(
            "SELECT correct, actual_direction, actual_magnitude FROM kronos_predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()
    assert row is not None, "prediction row vanished"
    assert row["correct"] is None, "close_trade must defer Kronos evaluation"
    assert row["actual_direction"] is None
    assert row["actual_magnitude"] is None

    # Now the deferred evaluator resolves it correctly with a price_lookup
    # that returns a synthetic close at prediction_time + horizon (UP, +3%).
    pred_time = pd.Timestamp(
        improver._fetch_prediction_time(pred_id) if hasattr(improver, "_fetch_prediction_time")
        else _read_prediction_time(tmp_db_path, pred_id)
    )
    target_time = pred_time + pd.Timedelta(hours=12)

    def fake_price_lookup(symbol, interval, start, end):
        # Anchor bar at prediction_time (close=100), target bar at target_time
        # (close=103 → +3% → UP, matches the prediction).
        idx = pd.DatetimeIndex([pred_time, target_time])
        return pd.DataFrame({"Close": [100.0, 103.0]}, index=idx)

    stats = improver.evaluate_pending_kronos_predictions(
        now=target_time + pd.Timedelta(minutes=1),
        price_lookup=fake_price_lookup,
    )
    assert stats["resolved"] == 1

    with get_connection(tmp_db_path) as conn:
        resolved = conn.execute(
            "SELECT correct, actual_direction, actual_magnitude FROM kronos_predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()
    assert resolved["correct"] == 1
    assert resolved["actual_direction"] == "UP"
    assert abs(resolved["actual_magnitude"] - 3.0) < 0.01


def _read_prediction_time(db_path: str, pred_id: int):
    from db_schema import get_connection
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT prediction_time FROM kronos_predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()
    return row["prediction_time"]


def test_close_trade_without_kronos_metadata_is_safe(tmp_db_path):
    """close_trade must still work for trades that never carried a Kronos prediction."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(symbol="BTC-USD", metadata={})
    tid = engine.execute_trade(sig, _mk_position())
    assert tid is not None

    result = engine.close_trade(tid, exit_price=105.0, reason="manual")
    assert result is not None
    assert result["status"] == "CLOSED"


# ---------------------------------------------------------------------------
# Adaptive params flow through strategies.py
# ---------------------------------------------------------------------------


def test_mean_reversion_honors_adaptive_rsi_thresholds(monkeypatch):
    """When adaptive_params supply tighter thresholds, the strategy uses them."""
    # A frame at RSI ≈ 68. The default threshold (70) won't fire, but a tuned
    # threshold of 60 should.
    strat = MeanReversionStrategy()

    df = _fake_frame(n=80, base=100.0)
    indicators = {
        "rsi": 68.0,
        # STOCH_DEEP_HIGH is 85.0 with a strict ``>``; use 85.5 so the deep-
        # stoch gate passes regardless of float rounding.
        "stoch_k": 85.5,
        "atr": 1.0,
        "sma_20": 95.0,
    }

    # Without tuning → no signal (rsi 68 < 70 threshold).
    sig_default = strat.generate_signal(df, indicator_data=indicators)
    assert sig_default is None

    # With a tuned threshold of 60 → fires short.
    sig_tuned = strat.generate_signal(
        df,
        indicator_data=indicators,
        adaptive_params={"rsi_overbought": 60.0, "rsi_oversold": 30.0, "sl_atr_mult": 1.0},
    )
    assert sig_tuned is not None
    assert sig_tuned.direction == "SHORT"


# ---------------------------------------------------------------------------
# Scanner never crashes when improver raises
# ---------------------------------------------------------------------------


class _ExplodingImprover(_FakeImprover):
    def get_strategy_weights(self) -> Dict[str, float]:
        raise RuntimeError("boom-weights")

    def predict_signal_quality(self, features, strategy="", category="") -> float:
        raise RuntimeError("boom-quality")

    def run_improvement_cycle(self, total_trades=None, symbol_bars=None):
        raise RuntimeError("boom-cycle")


def test_execute_signals_survives_improver_failures(scanner):
    """If SelfImprover methods raise, trading must still proceed normally."""
    scanner.self_improver = _ExplodingImprover()
    sig = _mk_signal(strategy=StrategyType.MOMENTUM, symbol="BTC-USD", strength=0.9)

    # Should not raise.
    trade_ids = scanner.execute_signals([sig])
    assert trade_ids == [1]


def test_run_scan_cycle_survives_improver_failures(scanner, monkeypatch):
    """run_scan_cycle must not propagate self-improver exceptions."""
    scanner.self_improver = _ExplodingImprover()
    # No-op fetch so scan_market doesn't hit the network.
    monkeypatch.setattr("scanner.fetch_market_data", lambda *a, **kw: pd.DataFrame())
    results = scanner.run_scan_cycle(symbols=["BTC-USD"])
    assert "improvement" in results
    # Either succeeded or has an error message — both are acceptable.
    assert results["improvement"] is None or isinstance(results["improvement"], dict)


# ---------------------------------------------------------------------------
# End-to-end: full cycle on a tiny universe
# ---------------------------------------------------------------------------


def test_run_scan_cycle_end_to_end_writes_regime_and_improvement(tmp_db_path, monkeypatch):
    """One full cycle should populate regime_history and flow through L4."""
    df = _fake_frame(n=200, base=100.0)
    monkeypatch.setattr("scanner.fetch_market_data", lambda symbol, tf, **kw: df.copy())

    scanner = MarketScanner(db_path=tmp_db_path, use_kronos=False, use_agents=False)
    results = scanner.run_scan_cycle(symbols=["BTC-USD"])

    assert "improvement" in results
    assert "regimes" in results
    assert "BTC-USD" in results["regimes"]

    # regime_history should have at least one entry for BTC-USD.
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM regime_history WHERE symbol = 'BTC-USD'"
        ).fetchone()["n"]
    assert n >= 1
