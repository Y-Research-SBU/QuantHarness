"""Integration tests — self-improvement interacting with a real paper trading DB."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from db_schema import get_connection, init_db
from market_config import StrategyType
from paper_trading import PaperTradingEngine
from position_sizing import PositionSizeResult
from self_improver import (
    CADENCE_L1,
    WEIGHT_DISABLED,
    WEIGHT_HIGH_SHARPE,
    SelfImprover,
)
from strategies import Signal


# ---------------------------------------------------------------------------
# Helpers to drive the paper trading engine with synthetic signals.
# ---------------------------------------------------------------------------


def _sig(symbol: str, direction: str, strategy: StrategyType, entry: float = 100.0) -> Signal:
    stop = entry * (0.98 if direction == "LONG" else 1.02)
    tp = entry * (1.04 if direction == "LONG" else 0.96)
    return Signal(
        direction=direction,
        strength=0.7,
        strategy=strategy,
        symbol=symbol,
        timeframe="1h",
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        risk_reward_ratio=2.0,
        reasoning="integration-test signal",
        metadata={"atr_pct": 0.02, "rsi": 55, "signal_strength": 0.7, "kronos_confidence": 0.7},
    )


def _pos(size: float = 500.0, qty: float = 5.0, stop: float = 98.0, tp: float = 104.0) -> PositionSizeResult:
    return PositionSizeResult(
        position_size_usd=size,
        quantity=qty,
        risk_per_trade_usd=size * 0.02,
        risk_pct=0.02,
        kelly_fraction=0.2,
        half_kelly=0.1,
        stop_loss=stop,
        take_profit=tp,
        reason="integration-test",
    )


def _seed_closed_trade(
    db_path: str,
    strategy: str,
    pnl: float,
    *,
    symbol: str = "BTC-USD",
    atr_pct: float = 0.02,
    entry_time: str = None,
) -> int:
    """Insert a fully-formed closed trade row."""
    init_db(db_path)
    md = {"atr_pct": atr_pct, "rsi": 55.0, "signal_strength": 0.7, "kronos_confidence": 0.7}
    entry_time = entry_time or datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO trades
                (symbol, timeframe, strategy, direction, entry_price, exit_price,
                 position_size, quantity, pnl, pnl_pct, status, decision_json,
                 entry_time, exit_time)
            VALUES (?, 1, ?, 'LONG', 100.0, 101.0, 1000.0, 10.0, ?, ?, 'CLOSED', ?, ?, ?)
            """,
            (
                symbol,
                strategy,
                pnl,
                pnl / 1000.0,
                json.dumps(md),
                entry_time,
                entry_time,
            ),
        )
        return int(cur.lastrowid)


def _seed_batch(db_path: str, strategy: str, n_win: int, n_loss: int):
    base = datetime.utcnow() - timedelta(hours=n_win + n_loss)
    for i in range(n_win):
        _seed_closed_trade(db_path, strategy, pnl=25.0, entry_time=(base + timedelta(hours=i)).isoformat())
    for i in range(n_loss):
        _seed_closed_trade(db_path, strategy, pnl=-15.0, entry_time=(base + timedelta(hours=n_win + i)).isoformat())


def _osc_bars(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = np.arange(n)
    prices = 100 + 15 * np.sin(x * 0.2) + rng.normal(0, 0.5, size=n)
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


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_self_improvement_schema_compatible_with_trading_schema(tmp_db_path):
    """Applying the self-improvement schema shouldn't break trade inserts."""
    from self_improvement_schema import apply_self_improvement_schema

    apply_self_improvement_schema(tmp_db_path)
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _sig("BTC-USD", "LONG", StrategyType.MOMENTUM)
    pos = _pos()
    trade_id = engine.execute_trade(sig, pos)
    assert trade_id is not None


def test_full_improvement_cycle_200_trades(tmp_db_path):
    """Seed 200 closed trades and run the full cycle — L1, L2, L3 all fire."""
    _seed_batch(tmp_db_path, "momentum", n_win=120, n_loss=80)
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=200, symbol_bars={"BTC-USD": _osc_bars(300)})
    assert {"L1", "L2", "L3", "L4"}.issubset(set(out["levels_run"]))
    assert out["model_trained"] is True
    assert "momentum" in out["strategy_weights"]


def test_strategy_disabled_after_losing_streak(tmp_db_path):
    """60 consistent losers → strategy should be disabled."""
    _seed_batch(tmp_db_path, "breakout", n_win=5, n_loss=60)
    si = SelfImprover(db_path=tmp_db_path)
    si.run_strategy_scoring()
    assert "breakout" in si.get_disabled_strategies()


def test_strategy_weights_reflect_performance_gradient(tmp_db_path):
    """Multiple strategies → weights should reflect their respective sharpe ratios."""
    _seed_batch(tmp_db_path, "momentum", n_win=60, n_loss=5)          # excellent
    _seed_batch(tmp_db_path, "mean_reversion", n_win=5, n_loss=55)     # terrible
    _seed_batch(tmp_db_path, "multi_factor", n_win=30, n_loss=25)      # mediocre
    si = SelfImprover(db_path=tmp_db_path)
    weights = si.get_strategy_weights()
    # Exact weights depend on synthetic-data sharpe, but ordering must hold:
    assert weights["momentum"] > weights["mean_reversion"]
    assert weights["mean_reversion"] == WEIGHT_DISABLED
    assert weights["multi_factor"] > weights["mean_reversion"]


def test_parameter_adaptation_writes_to_db(tmp_db_path):
    """After running L2, adaptive_params table should contain rows."""
    _seed_batch(tmp_db_path, "momentum", n_win=18, n_loss=12)
    si = SelfImprover(db_path=tmp_db_path)
    si.run_mini_optimization(symbol_bars={"BTC-USD": _osc_bars(300)})
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM adaptive_params").fetchall()
    assert len(rows) > 0


def test_regime_detection_changes_weights(tmp_db_path):
    """Regime affinity should shift the effective weight for a strategy."""
    _seed_batch(tmp_db_path, "momentum", n_win=40, n_loss=10)
    si = SelfImprover(db_path=tmp_db_path)
    # Momentum has affinity 1.5 in trending_up and 0.4 in ranging.
    up_w = si.get_regime_adjusted_weight("momentum", "trending_up")
    range_w = si.get_regime_adjusted_weight("momentum", "ranging")
    assert up_w > range_w


def test_signal_scorer_filters_bad_trades(tmp_db_path):
    """After training, the scorer should rate good-feature signals higher than bad ones."""
    # Seed trades where kronos_confidence drives the win flag.
    for i in range(120):
        good = i < 80
        pnl = 25.0 if good else -15.0
        md = {
            "kronos_confidence": 0.9 if good else 0.3,
            "signal_strength": 0.9 if good else 0.1,
            "rsi": 55.0,
            "macd_hist": 0.1 if good else -0.1,
            "atr_pct": 0.02,
            "volume_ratio": 1.2,
        }
        with get_connection(tmp_db_path) as conn:
            conn.execute(
                """INSERT INTO trades
                   (symbol, timeframe, strategy, direction, entry_price, exit_price,
                    position_size, quantity, pnl, pnl_pct, status, decision_json,
                    entry_time, exit_time)
                   VALUES (?, '1h', 'momentum', 'LONG', 100, 101, 1000, 10, ?, ?, 'CLOSED', ?, ?, ?)""",
                (
                    "BTC-USD",
                    pnl,
                    pnl / 1000.0,
                    json.dumps(md),
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
    si = SelfImprover(db_path=tmp_db_path)
    si.train_signal_model()
    high = si.predict_signal_quality(
        {"kronos_confidence": 0.9, "signal_strength": 0.9, "rsi": 55, "atr_pct": 0.02, "volume_ratio": 1.2},
        strategy="momentum", category="crypto",
    )
    low = si.predict_signal_quality(
        {"kronos_confidence": 0.3, "signal_strength": 0.1, "rsi": 55, "atr_pct": 0.02, "volume_ratio": 1.2},
        strategy="momentum", category="crypto",
    )
    assert high > low


def test_live_trade_then_improvement_cycle(tmp_db_path):
    """Executing a real trade and then running the improvement cycle should coexist."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    # Open & close a single trade so trades table has at least one row
    sig = _sig("BTC-USD", "LONG", StrategyType.MOMENTUM)
    pos = _pos()
    trade_id = engine.execute_trade(sig, pos)
    assert trade_id is not None
    engine.close_trade(trade_id, exit_price=102.0, reason="test")
    # Now run improvement cycle — should not raise.
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=1)
    assert "L1" not in out["levels_run"]  # cadence not hit yet


def test_trade_accumulation_triggers_l1_at_boundary(tmp_db_path):
    """As total_trades grows past 50, L1 should fire exactly once per boundary."""
    _seed_batch(tmp_db_path, "momentum", n_win=30, n_loss=20)
    si = SelfImprover(db_path=tmp_db_path)
    # Below cadence → no L1
    out_49 = si.run_improvement_cycle(total_trades=49)
    assert "L1" not in out_49["levels_run"]
    # Crossing 50 → L1 fires
    out_50 = si.run_improvement_cycle(total_trades=50)
    assert "L1" in out_50["levels_run"]
    # Calling again at 50 → no re-fire
    out_repeat = si.run_improvement_cycle(total_trades=50)
    assert "L1" not in out_repeat["levels_run"]
    # Crossing 100 → L1 fires again
    out_100 = si.run_improvement_cycle(total_trades=100)
    assert "L1" in out_100["levels_run"]


def test_regime_history_persisted_via_log_regime(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    bars = _osc_bars(300)
    for sym in ("BTC-USD", "ETH-USD", "SOL-USD"):
        si.log_regime(sym, bars, timeframe="1h")
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT DISTINCT symbol FROM regime_history").fetchall()
    syms = {r["symbol"] for r in rows}
    assert {"BTC-USD", "ETH-USD", "SOL-USD"}.issubset(syms)


def test_kronos_prediction_and_trade_schema_independent(tmp_db_path):
    """Kronos predictions table should not collide with trades table."""
    si = SelfImprover(db_path=tmp_db_path)
    engine = PaperTradingEngine(db_path=tmp_db_path)

    sig = _sig("BTC-USD", "LONG", StrategyType.MOMENTUM)
    pos = _pos()
    tid = engine.execute_trade(sig, pos)
    pid = si.log_kronos_prediction("BTC-USD", "1h", "UP", 0.02, 0.8, 12)
    assert tid is not None and pid > 0
    with get_connection(tmp_db_path) as conn:
        n_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        n_preds = conn.execute("SELECT COUNT(*) FROM kronos_predictions").fetchone()[0]
    assert n_trades == 1 and n_preds == 1
