"""End-to-end tests for the self-improvement system.

These exercise the full loop: a mock trading environment generates trades,
the SelfImprover observes outcomes, and we verify that subsequent cycles
actually change behavior (disable bad strategies, adjust params, etc.).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from db_schema import get_connection, init_db
from market_config import StrategyType
from paper_trading import PaperTradingEngine
from position_sizing import PositionSizeResult
from self_improver import SelfImprover, WEIGHT_DISABLED
from strategies import Signal


# ---------------------------------------------------------------------------
# A minimal mock scanner that emits synthetic trade outcomes on each "tick".
# ---------------------------------------------------------------------------


class MockScanner:
    """Writes closed trades directly to the DB to simulate a scanning cycle."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        init_db(db_path)
        self.trade_count = 0

    def emit_trade(
        self,
        strategy: str,
        symbol: str = "BTC-USD",
        is_win: bool = True,
        atr_pct: float = 0.02,
        kronos_conf: float = 0.7,
        signal_strength: float = 0.7,
    ) -> int:
        pnl = 25.0 if is_win else -15.0
        md = {
            "atr_pct": atr_pct,
            "rsi": 55.0,
            "signal_strength": signal_strength,
            "kronos_confidence": kronos_conf,
            "macd_hist": 0.05 if is_win else -0.05,
            "volume_ratio": 1.2,
        }
        now = (datetime.utcnow() + timedelta(seconds=self.trade_count)).isoformat()
        self.trade_count += 1
        with get_connection(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO trades
                    (symbol, timeframe, strategy, direction, entry_price, exit_price,
                     position_size, quantity, pnl, pnl_pct, status, decision_json,
                     entry_time, exit_time)
                VALUES (?, '1h', ?, 'LONG', 100, 101, 1000, 10, ?, ?, 'CLOSED', ?, ?, ?)
                """,
                (
                    symbol,
                    strategy,
                    pnl,
                    pnl / 1000.0,
                    json.dumps(md),
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def burst(
        self,
        strategy: str,
        n_win: int,
        n_loss: int,
        symbol: str = "BTC-USD",
        kronos_conf_win: float = 0.85,
        kronos_conf_loss: float = 0.35,
    ) -> None:
        for _ in range(n_win):
            self.emit_trade(strategy, symbol, is_win=True, kronos_conf=kronos_conf_win)
        for _ in range(n_loss):
            self.emit_trade(strategy, symbol, is_win=False, kronos_conf=kronos_conf_loss)


def _osc_bars(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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
# E2E tests
# ---------------------------------------------------------------------------


def test_improvement_loop_with_mock_scanner(tmp_db_path):
    """Run 5 scanner→improvement cycles and verify the system adapts."""
    scanner = MockScanner(tmp_db_path)
    si = SelfImprover(db_path=tmp_db_path)

    levels_fired_total: set = set()

    # Simulate 5 bursts of 50 trades each → 250 trades total.
    for cycle in range(5):
        # Each cycle: momentum wins, mean_reversion loses.
        scanner.burst("momentum", n_win=40, n_loss=10)
        scanner.burst("mean_reversion", n_win=5, n_loss=45)
        total = scanner.trade_count
        result = si.run_improvement_cycle(
            total_trades=total, symbol_bars={"BTC-USD": _osc_bars(300, seed=cycle)}
        )
        levels_fired_total.update(result["levels_run"])

    # Over 250 trades every level should have fired at least once.
    assert {"L1", "L2", "L3", "L4"}.issubset(levels_fired_total)
    # Mean_reversion should be disabled by the end.
    assert "mean_reversion" in si.get_disabled_strategies()
    # Momentum should NOT be disabled.
    assert "momentum" not in si.get_disabled_strategies()


def test_improvement_makes_system_smarter(tmp_db_path):
    """After training, the scorer should discriminate better than random guessing."""
    scanner = MockScanner(tmp_db_path)
    # Seed a strong signal: kronos_conf drives pnl.
    scanner.burst("momentum", n_win=100, n_loss=100,
                  kronos_conf_win=0.9, kronos_conf_loss=0.3)
    si = SelfImprover(db_path=tmp_db_path)
    trained = si.train_signal_model()
    assert trained is not None
    # High confidence should score higher than low confidence.
    high = si.predict_signal_quality(
        {"kronos_confidence": 0.9, "signal_strength": 0.9, "macd_hist": 0.1,
         "atr_pct": 0.02, "volume_ratio": 1.2, "rsi": 55},
        strategy="momentum", category="crypto",
    )
    low = si.predict_signal_quality(
        {"kronos_confidence": 0.3, "signal_strength": 0.1, "macd_hist": -0.1,
         "atr_pct": 0.02, "volume_ratio": 1.2, "rsi": 55},
        strategy="momentum", category="crypto",
    )
    assert high > low
    assert trained.accuracy >= 0.6  # meaningfully better than coin flip


def test_no_interference_with_trading_while_trades_open(tmp_db_path):
    """Running the improvement cycle with open trades must not touch them."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    # Open two trades — stay within correlation-group limits (max 2 per group).
    open_ids = []
    for sym, entry in (("BTC-USD", 100), ("ETH-USD", 50)):
        # 4% stop / 8% TP — clears the 3% crypto floor in PaperTradingEngine.
        sig = Signal(
            direction="LONG", strength=0.7, strategy=StrategyType.MOMENTUM,
            symbol=sym, timeframe="1h", entry_price=float(entry),
            stop_loss=entry * 0.96, take_profit=entry * 1.08,
            risk_reward_ratio=2.0, reasoning="open trade", metadata={"atr_pct": 0.02},
        )
        pos = PositionSizeResult(
            position_size_usd=500, quantity=5, risk_per_trade_usd=10, risk_pct=0.02,
            kelly_fraction=0.2, half_kelly=0.1, stop_loss=sig.stop_loss,
            take_profit=sig.take_profit, reason="t",
        )
        tid = engine.execute_trade(sig, pos)
        assert tid is not None
        open_ids.append(tid)

    # Snapshot open-trade data before running improvement cycle.
    with get_connection(tmp_db_path) as conn:
        before = {
            r["id"]: dict(r)
            for r in conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN'"
            ).fetchall()
        }

    si = SelfImprover(db_path=tmp_db_path)
    si.run_improvement_cycle(total_trades=50, symbol_bars={"BTC-USD": _osc_bars(300)})

    with get_connection(tmp_db_path) as conn:
        after = {
            r["id"]: dict(r)
            for r in conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN'"
            ).fetchall()
        }

    # No open trade row should have been modified (same rows, same field values).
    assert set(before.keys()) == set(after.keys())
    for tid in before:
        for key in ("status", "exit_price", "pnl", "position_size", "stop_loss"):
            assert before[tid][key] == after[tid][key], f"open trade {tid} field {key} was mutated"


def test_improvement_data_exposed_for_dashboard(tmp_db_path):
    """All improvement tables should be queryable for dashboard integration."""
    scanner = MockScanner(tmp_db_path)
    scanner.burst("momentum", n_win=30, n_loss=20)
    si = SelfImprover(db_path=tmp_db_path)
    si.run_strategy_scoring()
    si.log_regime("BTC-USD", _osc_bars(300))
    si.log_kronos_prediction("BTC-USD", "1h", "UP", 0.02, 0.8, 12)

    with get_connection(tmp_db_path) as conn:
        evolution = conn.execute("SELECT * FROM strategy_evolution").fetchall()
        regime = conn.execute("SELECT * FROM regime_history").fetchall()
        kronos = conn.execute("SELECT * FROM kronos_predictions").fetchall()
    assert len(evolution) >= 1
    assert len(regime) >= 1
    assert len(kronos) >= 1


def test_five_cycles_converge_on_stable_weights(tmp_db_path):
    """After enough trades of steady performance, strategy weights should stabilize."""
    scanner = MockScanner(tmp_db_path)
    si = SelfImprover(db_path=tmp_db_path)
    weights_history: List[Dict[str, float]] = []
    for _ in range(5):
        # Steady 80% win rate for momentum every cycle.
        scanner.burst("momentum", n_win=40, n_loss=10)
        si.run_improvement_cycle(total_trades=scanner.trade_count)
        weights_history.append(si.get_strategy_weights())
    # Last three cycles should all assign the same weight.
    last_three_weights = {w.get("momentum") for w in weights_history[-3:]}
    assert len(last_three_weights) == 1, f"weights unstable: {weights_history}"


def test_adaptive_params_used_across_cycles(tmp_db_path):
    """Optimized params should persist between improvement cycles."""
    scanner = MockScanner(tmp_db_path)
    # Need at least 100 trades for L2 cadence to trigger automatically.
    scanner.burst("momentum", n_win=70, n_loss=30)
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=scanner.trade_count,
                                    symbol_bars={"BTC-USD": _osc_bars(300)})
    assert "L2" in out["levels_run"]
    # After cycle 1, adaptive_params table has rows
    with get_connection(tmp_db_path) as conn:
        n_rows_1 = conn.execute("SELECT COUNT(*) FROM adaptive_params").fetchone()[0]
    assert n_rows_1 > 0

    # Run another cycle; params should still be retrievable.
    fresh_si = SelfImprover(db_path=tmp_db_path)
    params = fresh_si.get_adaptive_params("momentum", "BTC-USD")
    assert "sl_atr_mult" in params


def test_kronos_tracking_across_multiple_predictions(tmp_db_path):
    """Log predictions for multiple markets, evaluate, and verify stats."""
    si = SelfImprover(db_path=tmp_db_path)
    for sym in ("BTC-USD", "ETH-USD"):
        for i in range(15):
            pid = si.log_kronos_prediction(sym, "1h", "UP", 0.02, 0.7, 12)
            actual = "UP" if i % 2 == 0 else "DOWN"
            si.record_kronos_outcome(pid, actual, 0.015)
    stats = si.evaluate_kronos_accuracy()
    assert stats["n"] == 30
    assert 0.0 <= stats["hit_rate"] <= 1.0
    assert "BTC-USD" in stats["per_market_accuracy"]
    assert "ETH-USD" in stats["per_market_accuracy"]


def test_full_levels_all_fire_over_500_trades(tmp_db_path):
    """Run enough trades that L5 fires."""
    scanner = MockScanner(tmp_db_path)
    si = SelfImprover(db_path=tmp_db_path)

    # Seed 500+ kronos predictions first so L5 has data
    for i in range(30):
        pid = si.log_kronos_prediction("BTC-USD", "1h", "UP", 0.02, 0.75, 12)
        si.record_kronos_outcome(pid, "UP" if i % 3 != 0 else "DOWN", 0.018)

    # Now simulate cycles up to 500 trades
    all_levels: set = set()
    for _ in range(10):
        scanner.burst("momentum", n_win=40, n_loss=10)
        out = si.run_improvement_cycle(
            total_trades=scanner.trade_count, symbol_bars={"BTC-USD": _osc_bars(300)}
        )
        all_levels.update(out["levels_run"])

    assert {"L1", "L2", "L3", "L4", "L5"}.issubset(all_levels), f"Missing levels: {all_levels}"
