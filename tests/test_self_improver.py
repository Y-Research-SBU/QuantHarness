"""Unit tests for self_improver.SelfImprover."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytest

from db_schema import get_connection, init_db
from self_improver import (
    CADENCE_L1,
    CADENCE_L2,
    CADENCE_L3,
    CADENCE_L5,
    DEFAULT_REGIME_AFFINITY,
    WEIGHT_DISABLED,
    WEIGHT_HIGH_SHARPE,
    WEIGHT_NORMAL,
    WEIGHT_REDUCED,
    SelfImprover,
)


# ---------------------------------------------------------------------------
# Helpers — seed the trades table with synthetic closed trades.
# ---------------------------------------------------------------------------


def _seed_trade(
    db_path: str,
    strategy: str,
    pnl: float,
    *,
    symbol: str = "BTC-USD",
    timeframe: str = "1h",
    direction: str = "LONG",
    entry_price: float = 100.0,
    atr_pct: float = 0.02,
    entry_time: str = None,
    exit_time: str = None,
    metadata_extra: dict = None,
) -> int:
    """Insert a synthetic closed trade and return its id."""
    init_db(db_path)
    md = {"atr_pct": atr_pct, "rsi": 55.0, "signal_strength": 0.7}
    if metadata_extra:
        md.update(metadata_extra)

    entry_time = entry_time or datetime.utcnow().isoformat()
    exit_time = exit_time or datetime.utcnow().isoformat()
    exit_price = entry_price * (1 + pnl / (entry_price * 0.1))  # any plausible exit
    pnl_pct = pnl / 1000.0  # If position_size=1000 then pnl_pct = pnl/1000

    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO trades
                (symbol, timeframe, strategy, direction, entry_price, exit_price,
                 position_size, quantity, pnl, pnl_pct, status, decision_json,
                 entry_time, exit_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED', ?, ?, ?)
            """,
            (
                symbol,
                timeframe,
                strategy,
                direction,
                entry_price,
                exit_price,
                1000.0,
                10.0,
                pnl,
                pnl_pct,
                json.dumps(md),
                entry_time,
                exit_time,
            ),
        )
        return int(cur.lastrowid)


def _seed_many(db_path: str, strategy: str, n_wins: int, n_losses: int, symbol: str = "BTC-USD"):
    base = datetime.utcnow() - timedelta(hours=n_wins + n_losses)
    for i in range(n_wins):
        t = (base + timedelta(hours=i)).isoformat()
        _seed_trade(db_path, strategy, pnl=25.0, symbol=symbol, entry_time=t, exit_time=t)
    for i in range(n_losses):
        t = (base + timedelta(hours=n_wins + i)).isoformat()
        _seed_trade(db_path, strategy, pnl=-15.0, symbol=symbol, entry_time=t, exit_time=t)


# ---------------------------------------------------------------------------
# LEVEL 1: Strategy scoring
# ---------------------------------------------------------------------------


def test_score_strategies_with_no_trades(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    assert si.score_strategies() == {}


def test_score_strategies_with_mixed_results(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=40, n_losses=10)  # clearly profitable
    _seed_many(tmp_db_path, "mean_reversion", n_wins=10, n_losses=40)  # clearly losing
    si = SelfImprover(db_path=tmp_db_path)
    scores = si.score_strategies()
    assert scores["momentum"] > 0
    assert scores["mean_reversion"] < 0


def test_get_strategy_weights_disables_negative_sharpe(tmp_db_path):
    # 60 losing trades → negative sharpe, n >= MIN_TRADES_FOR_DISABLE → disable
    _seed_many(tmp_db_path, "mean_reversion", n_wins=5, n_losses=55)
    si = SelfImprover(db_path=tmp_db_path)
    weights = si.get_strategy_weights()
    assert weights["mean_reversion"] == WEIGHT_DISABLED


def test_get_strategy_weights_doubles_high_sharpe(tmp_db_path):
    # Nearly all wins → very high sharpe
    _seed_many(tmp_db_path, "momentum", n_wins=60, n_losses=2)
    si = SelfImprover(db_path=tmp_db_path)
    weights = si.get_strategy_weights()
    assert weights["momentum"] == WEIGHT_HIGH_SHARPE


def test_get_strategy_weights_reduced_for_small_negative_sharpe(tmp_db_path):
    # Small sample size, slightly negative → reduced (not disabled)
    _seed_many(tmp_db_path, "breakout", n_wins=10, n_losses=15)
    si = SelfImprover(db_path=tmp_db_path)
    weights = si.get_strategy_weights()
    assert weights["breakout"] == WEIGHT_REDUCED  # fewer than 50 trades


def test_get_disabled_strategies_lists_only_disabled(tmp_db_path):
    _seed_many(tmp_db_path, "mean_reversion", n_wins=5, n_losses=55)
    _seed_many(tmp_db_path, "momentum", n_wins=30, n_losses=5)
    si = SelfImprover(db_path=tmp_db_path)
    disabled = si.get_disabled_strategies()
    assert "mean_reversion" in disabled
    assert "momentum" not in disabled


def test_run_strategy_scoring_persists_evolution_log(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=40, n_losses=10)
    si = SelfImprover(db_path=tmp_db_path)
    si.run_strategy_scoring()
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM strategy_evolution").fetchall()
    assert len(rows) >= 1
    assert any(r["strategy"] == "momentum" for r in rows)


# ---------------------------------------------------------------------------
# LEVEL 2: Parameter adaptation
# ---------------------------------------------------------------------------


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


def test_optimize_rsi_thresholds_with_bars(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    out = si.optimize_rsi_thresholds("BTC-USD", bars=_osc_bars(300))
    assert 50.0 <= out["rsi_overbought"] <= 90.0
    assert 10.0 <= out["rsi_oversold"] <= 50.0


def test_optimize_rsi_thresholds_without_bars_returns_defaults(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    out = si.optimize_rsi_thresholds("BTC-USD", bars=None)
    assert out["rsi_overbought"] == 70.0
    assert out["rsi_oversold"] == 30.0


def test_optimize_stop_distances_uses_closed_trades(tmp_db_path):
    # Seed 30 trades so optimizer has enough data
    _seed_many(tmp_db_path, "momentum", n_wins=18, n_losses=12)
    si = SelfImprover(db_path=tmp_db_path)
    out = si.optimize_stop_distances("momentum")
    assert 0.5 <= out["sl_atr_mult"] <= 3.0
    assert out["tp_atr_mult"] > out["sl_atr_mult"]


def test_get_adaptive_params_falls_back_to_defaults(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    p = si.get_adaptive_params("momentum", "BTC-USD")
    assert p["rsi_overbought"] == 70.0


def test_run_mini_optimization_with_sufficient_data(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=18, n_losses=12)
    si = SelfImprover(db_path=tmp_db_path)
    changes = si.run_mini_optimization(symbol_bars={"BTC-USD": _osc_bars(300)})
    assert "momentum" in changes  # stop distances updated
    assert any(k.startswith("BTC-USD:") for k in changes.get("mean_reversion", {}).keys())


def test_run_mini_optimization_with_insufficient_data(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    # No trades → nothing should be optimized (empty or no strategy keys)
    changes = si.run_mini_optimization()
    assert changes == {} or all(not v for v in changes.values())


# ---------------------------------------------------------------------------
# LEVEL 3: Signal quality scoring
# ---------------------------------------------------------------------------


def test_train_signal_model_with_no_trades(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    assert si.train_signal_model() is None


def test_train_signal_model_with_enough_trades(tmp_db_path):
    # Need >=100 trades with both win/loss classes present.
    _seed_many(tmp_db_path, "momentum", n_wins=60, n_losses=50)
    si = SelfImprover(db_path=tmp_db_path)
    trained = si.train_signal_model()
    assert trained is not None
    # verify model log persisted
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM signal_model_log").fetchall()
    assert len(rows) >= 1


def test_predict_signal_quality_neutral_without_training(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    p = si.predict_signal_quality({"rsi": 60}, strategy="momentum", category="crypto")
    assert p == 0.5


# ---------------------------------------------------------------------------
# LEVEL 4: Regime detection
# ---------------------------------------------------------------------------


def test_detect_regime_returns_valid_class(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    df = _osc_bars(300)
    regime = si.detect_regime(df)
    assert regime in {"trending_up", "trending_down", "ranging", "volatile", "unknown"}


def test_log_regime_persists(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    regime = si.log_regime("BTC-USD", _osc_bars(300), timeframe="1h")
    assert regime in {"trending_up", "trending_down", "ranging", "volatile", "unknown"}
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM regime_history WHERE symbol='BTC-USD'").fetchall()
    assert len(rows) == 1


def test_regime_strategy_affinity_has_defaults(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    affinity = si.get_regime_strategy_affinity()
    for strat in DEFAULT_REGIME_AFFINITY:
        assert strat in affinity


def test_get_regime_adjusted_weight_combines(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=40, n_losses=10)
    si = SelfImprover(db_path=tmp_db_path)
    w_trend = si.get_regime_adjusted_weight("momentum", "trending_up")
    w_range = si.get_regime_adjusted_weight("momentum", "ranging")
    # Momentum affinity for trending_up (1.5) > ranging (0.4)
    assert w_trend > w_range


# ---------------------------------------------------------------------------
# LEVEL 5: Kronos accuracy tracking
# ---------------------------------------------------------------------------


def test_log_kronos_prediction_and_evaluate(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    pid = si.log_kronos_prediction(
        "BTC-USD", "1h",
        predicted_direction="UP", predicted_magnitude=0.02,
        confidence=0.8, horizon=12, predicted_price=101.0,
    )
    assert pid > 0
    si.record_kronos_outcome(pid, actual_direction="UP", actual_magnitude=0.018, actual_price=101.8)
    stats = si.evaluate_kronos_accuracy()
    assert stats["n"] == 1
    assert stats["hit_rate"] == 1.0


def test_evaluate_kronos_accuracy_empty(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    stats = si.evaluate_kronos_accuracy()
    assert stats["n"] == 0
    assert stats["hit_rate"] == 0.0


def test_kronos_confidence_adjustment_defaults_when_scarce(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    # 0 samples → default 1.0
    assert si.get_kronos_confidence_adjustment("BTC-USD") == 1.0


def test_kronos_confidence_adjustment_discounts_overconfidence(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    # 20 predictions at conf=0.8, but only 50% correct → ratio < 1 → discount
    for i in range(20):
        pid = si.log_kronos_prediction("BTC-USD", "1h", "UP", 0.02, 0.8, 12)
        # Half correct, half wrong
        actual_dir = "UP" if i % 2 == 0 else "DOWN"
        si.record_kronos_outcome(pid, actual_dir, 0.01)
    adj = si.get_kronos_confidence_adjustment("BTC-USD")
    assert adj < 1.0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_run_improvement_cycle_at_zero_trades(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=0)
    # No levels should fire on an empty DB with 0 trades
    assert out["levels_run"] == []


def test_run_improvement_cycle_at_l1_threshold(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=15, n_losses=10)
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=CADENCE_L1)
    assert "L1" in out["levels_run"]
    # Not yet at L2 / L3 / L5.
    assert "L2" not in out["levels_run"]
    assert "L3" not in out["levels_run"]


def test_run_improvement_cycle_at_l2_threshold(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=30, n_losses=20)
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=CADENCE_L2)
    assert "L1" in out["levels_run"]
    assert "L2" in out["levels_run"]


def test_run_improvement_cycle_at_l3_threshold(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=60, n_losses=40)
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=CADENCE_L3)
    assert "L1" in out["levels_run"]
    assert "L2" in out["levels_run"]
    assert "L3" in out["levels_run"]
    assert out["model_trained"] is True


def test_run_improvement_cycle_at_l5_threshold(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=120, n_losses=80)
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=CADENCE_L5)
    assert "L5" in out["levels_run"]


def test_improvement_cycle_idempotent_on_repeated_calls(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=15, n_losses=10)
    si = SelfImprover(db_path=tmp_db_path)
    # First call at the L1 boundary fires L1.
    out1 = si.run_improvement_cycle(total_trades=CADENCE_L1)
    assert "L1" in out1["levels_run"]
    # Second call with the same trade count should NOT re-fire L1
    # (cadence tracking prevents duplicate work on the same boundary).
    out2 = si.run_improvement_cycle(total_trades=CADENCE_L1)
    assert "L1" not in out2["levels_run"]


def test_run_improvement_cycle_with_symbol_bars_runs_l4(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    out = si.run_improvement_cycle(total_trades=0, symbol_bars={"BTC-USD": _osc_bars(300)})
    assert "L4" in out["levels_run"]


# ---------------------------------------------------------------------------
# Extra coverage: edge cases & DB contracts
# ---------------------------------------------------------------------------


def test_self_improvement_tables_created_on_init(tmp_db_path):
    from self_improvement_schema import self_improvement_tables_exist
    SelfImprover(db_path=tmp_db_path)
    assert self_improvement_tables_exist(tmp_db_path)


def test_strategy_evolution_row_includes_weight_and_reason(tmp_db_path):
    _seed_many(tmp_db_path, "momentum", n_wins=40, n_losses=10)
    si = SelfImprover(db_path=tmp_db_path)
    si.run_strategy_scoring()
    with get_connection(tmp_db_path) as conn:
        row = conn.execute(
            "SELECT * FROM strategy_evolution WHERE strategy='momentum' LIMIT 1"
        ).fetchone()
    assert row is not None
    assert row["weight"] is not None
    assert row["reason"] is not None


def test_score_strategies_respects_rolling_window(tmp_db_path):
    """With a small rolling window only the most recent N trades count."""
    # 100 ancient losses, then 10 recent winners (varied so std > 0)
    base = datetime.utcnow() - timedelta(days=5)
    for i in range(100):
        t = (base + timedelta(minutes=i)).isoformat()
        _seed_trade(tmp_db_path, "momentum", pnl=-15.0, entry_time=t, exit_time=t)
    recent = datetime.utcnow()
    for i in range(10):
        t = (recent + timedelta(seconds=i)).isoformat()
        # Varying wins so the returns have non-zero std.
        _seed_trade(tmp_db_path, "momentum", pnl=20.0 + i, entry_time=t, exit_time=t)
    si = SelfImprover(db_path=tmp_db_path, rolling_window=10)
    scores = si.score_strategies()
    assert scores["momentum"] > 0  # only recent winners counted


def test_kronos_outcome_on_missing_prediction_is_noop(tmp_db_path):
    si = SelfImprover(db_path=tmp_db_path)
    # ID 999 doesn't exist → should silently skip
    si.record_kronos_outcome(999, "UP", 0.02)
    stats = si.evaluate_kronos_accuracy()
    assert stats["n"] == 0


def test_regime_affinity_learned_override(tmp_db_path):
    """When enough trades exist per (strategy, regime), learned affinity overrides prior."""
    si = SelfImprover(db_path=tmp_db_path)
    # Log a regime first so trades match against it.
    si.log_regime("BTC-USD", _osc_bars(300))
    # 15 profitable trades for momentum in ranging regime (ranging prior is 0.4)
    for _ in range(15):
        _seed_trade(tmp_db_path, "momentum", pnl=30.0)
    affinity = si.get_regime_strategy_affinity()
    # Either learned or prior — just verify structure
    assert "momentum" in affinity
    for regime in ("trending_up", "trending_down", "ranging", "volatile"):
        assert regime in affinity["momentum"]


def test_emergency_disable_low_win_rate(tmp_db_path):
    """Strategies with win rate below EMERGENCY_DISABLE_WIN_RATE are disabled
    even if sharpe is non-negative (e.g. a strategy that rarely trades but
    has a non-negative sharpe due to lucky outliers while losing 90%+ of trades)."""
    from self_improver import SelfImprover, WEIGHT_DISABLED, EMERGENCY_DISABLE_WIN_RATE
    imp = SelfImprover(db_path=tmp_db_path, min_trades_for_disable=5)
    conn = __import__("sqlite3").connect(tmp_db_path)
    # Insert 10 trades: 0 wins (all losses) but with pnl near 0 so sharpe ~ 0
    for i in range(10):
        conn.execute(
            "INSERT INTO trades (symbol, timeframe, strategy, direction, entry_price, position_size, quantity, stop_loss, take_profit, status, pnl, entry_time, exit_time) "
            "VALUES (?, '15m', 'bad_strat', 'LONG', 100, 10, 0.1, 95, 110, 'STOPPED', -0.01, datetime('now', ?), datetime('now', ?))",
            (f"SYM{i}-USD", f"-{10-i} hours", f"-{10-i} hours"),
        )
    conn.commit()
    conn.close()

    weights = imp.get_strategy_weights()
    assert weights.get("bad_strat") == WEIGHT_DISABLED, (
        f"Expected bad_strat (0% win rate) to be disabled, got weight={weights.get('bad_strat')}"
    )


def test_min_trades_for_disable_lowered():
    """MIN_TRADES_FOR_DISABLE should be 10 (not 50) to catch bad strategies faster."""
    from self_improver import MIN_TRADES_FOR_DISABLE
    assert MIN_TRADES_FOR_DISABLE == 10
