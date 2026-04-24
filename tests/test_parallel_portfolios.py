"""Tests for parallel_portfolios.py."""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest import mock

import pytest

import parallel_portfolios
from db_schema import get_connection, init_db
from parallel_portfolios import (
    PortfolioPerformance,
    PortfolioPreset,
    _max_drawdown_pct,
    _seed_adaptive_params,
    _sharpe,
    default_presets,
    measure_performance,
    merge_best_into_live,
    pick_best,
    render_comparison_table,
)
from self_improvement_schema import apply_self_improvement_schema


def test_default_presets_returns_five():
    presets = default_presets(db_dir="/tmp/quantagent_test")
    assert len(presets) == 5
    names = [p.name for p in presets]
    assert names == ["A", "B", "C", "D", "E"]
    # Each preset has its own DB path.
    paths = [p.db_path for p in presets]
    assert len(set(paths)) == 5


def test_preset_a_has_aggressive_rsi():
    presets = default_presets(db_dir="/tmp")
    a = next(p for p in presets if p.name == "A")
    assert a.seed_params["rsi_overbought"] == 60.0
    assert a.seed_params["rsi_oversold"] == 40.0
    assert a.seed_params["sl_atr_mult"] == 0.75


def test_preset_b_has_conservative_rsi():
    presets = default_presets(db_dir="/tmp")
    b = next(p for p in presets if p.name == "B")
    assert b.seed_params["rsi_overbought"] == 80.0
    assert b.seed_params["sl_atr_mult"] == 2.0


def test_preset_c_only_kronos_strategies():
    from market_config import StrategyType
    presets = default_presets(db_dir="/tmp")
    c = next(p for p in presets if p.name == "C")
    assert c.allowed_strategies is not None
    for st in c.allowed_strategies:
        assert "kronos" in st.value.lower() or "multi_timeframe" in st.value.lower()


def test_preset_d_no_kronos():
    from market_config import StrategyType
    presets = default_presets(db_dir="/tmp")
    d = next(p for p in presets if p.name == "D")
    assert d.use_kronos is False
    for st in d.allowed_strategies or ():
        assert "kronos" not in st.value.lower()


def test_preset_e_uses_self_improvement():
    presets = default_presets(db_dir="/tmp")
    e = next(p for p in presets if p.name == "E")
    assert e.use_self_improvement is True


def test_seed_adaptive_params_writes_rows(tmp_path):
    db = str(tmp_path / "p.db")
    init_db(db)
    apply_self_improvement_schema(db)
    preset = PortfolioPreset(
        name="X",
        db_path=db,
        description="test",
        seed_params={"sl_atr_mult": 0.9, "tp_atr_mult": 1.8},
    )
    _seed_adaptive_params(preset, ["BTC-USD"])
    with get_connection(db) as conn:
        rows = conn.execute(
            "SELECT param_name, param_value FROM adaptive_params WHERE symbol='BTC-USD'"
        ).fetchall()
    names = {r["param_name"] for r in rows}
    assert "sl_atr_mult" in names
    assert "tp_atr_mult" in names


def test_sharpe_zero_for_short_series():
    assert _sharpe([]) == 0.0
    assert _sharpe([0.01]) == 0.0


def test_sharpe_positive_for_consistent_wins():
    s = _sharpe([0.01, 0.012, 0.011, 0.013, 0.009, 0.012])
    assert s > 0


def test_max_drawdown_basic():
    curve = [100, 110, 105, 120, 90, 100]
    dd = _max_drawdown_pct(curve)
    # Peak at 120, trough at 90 → (120-90)/120 = 0.25
    assert dd == pytest.approx(0.25, abs=1e-3)


def test_max_drawdown_empty():
    assert _max_drawdown_pct([]) == 0.0


def test_measure_performance_returns_zeros_for_missing_db(tmp_path):
    preset = PortfolioPreset(
        name="X",
        db_path=str(tmp_path / "nope.db"),
        description="missing",
    )
    perf = measure_performance(preset)
    assert perf.closed_trades == 0
    assert perf.total_pnl == 0.0


def test_measure_performance_reads_master_balance(tmp_path):
    db = str(tmp_path / "p.db")
    init_db(db)
    with get_connection(db) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO portfolios
               (symbol, initial_balance, current_balance, peak_balance,
                total_pnl, total_trades, winning_trades, losing_trades)
               VALUES ('__MASTER__', 10000, 10500, 10500, 500, 5, 3, 2)"""
        )
    preset = PortfolioPreset(name="X", db_path=db, description="x")
    perf = measure_performance(preset)
    assert perf.total_trades == 5
    assert perf.winning_trades == 3
    assert perf.total_pnl == pytest.approx(500.0)
    assert perf.final_equity == pytest.approx(10500.0)


def test_pick_best_picks_highest_sharpe():
    perfs = {
        "A": PortfolioPerformance(name="A", description="", db_path="", closed_trades=10, sharpe_ratio=0.5, total_pnl=100),
        "B": PortfolioPerformance(name="B", description="", db_path="", closed_trades=10, sharpe_ratio=1.5, total_pnl=200),
        "C": PortfolioPerformance(name="C", description="", db_path="", closed_trades=0, sharpe_ratio=99.0, total_pnl=0),
    }
    best = pick_best(perfs)
    # C has 0 trades — must be skipped.
    assert best is not None
    assert best.name == "B"


def test_pick_best_returns_none_when_all_empty():
    perfs = {
        "A": PortfolioPerformance(name="A", description="", db_path="", closed_trades=0),
    }
    assert pick_best(perfs) is None


def test_render_comparison_table_lists_each_portfolio():
    perfs = {
        "A": PortfolioPerformance(name="A", description="aggressive", db_path="", closed_trades=5,
                                   win_rate=0.6, sharpe_ratio=1.2, max_drawdown_pct=0.05,
                                   total_pnl=120, final_equity=10120),
        "B": PortfolioPerformance(name="B", description="conservative", db_path="", closed_trades=4,
                                   win_rate=0.5, sharpe_ratio=0.8, max_drawdown_pct=0.04,
                                   total_pnl=80, final_equity=10080),
    }
    table = render_comparison_table(perfs)
    assert "A" in table and "B" in table
    assert "aggressive" in table
    assert "Sharpe" in table


def test_merge_best_into_live_copies_adaptive_params(tmp_path):
    src_db = str(tmp_path / "winner.db")
    live_db = str(tmp_path / "live.db")
    init_db(src_db)
    apply_self_improvement_schema(src_db)
    apply_self_improvement_schema(live_db)
    with get_connection(src_db) as conn:
        conn.execute(
            """INSERT INTO adaptive_params
               (strategy, symbol, param_name, param_value, sample_size, improvement_pct)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("momentum", "BTC-USD", "sl_atr_mult", 1.7, 200, 0.1),
        )
    best = PortfolioPerformance(name="A", description="", db_path=src_db, closed_trades=10, sharpe_ratio=1.5)
    counts = merge_best_into_live(best, live_db=live_db)
    assert counts["adaptive_params"] >= 1
    with get_connection(live_db) as conn:
        row = conn.execute(
            "SELECT param_value FROM adaptive_params WHERE strategy='momentum'"
        ).fetchone()
    assert row is not None
    assert row["param_value"] == pytest.approx(1.7)


def test_main_with_no_cycles_runs(tmp_path, monkeypatch):
    """The CLI should run end-to-end on an empty universe without crashing."""
    captured = {}

    class FakeScanner:
        def __init__(self, *args, **kwargs):
            captured.setdefault("constructed", 0)
            captured["constructed"] += 1

        def run_scan_cycle(self, symbols=None):
            return {"signals_found": 0, "trades_opened": 0, "stops_triggered": 0}

        scan_market = None  # unused but exists

    # Force MarketScanner used inside _build_scanner to be the fake.
    monkeypatch.setattr(parallel_portfolios, "MarketScanner", FakeScanner)
    rc = parallel_portfolios.main([
        "--symbols", "BTC-USD",
        "--trades", "1",
        "--max-cycles", "1",
        "--db-dir", str(tmp_path),
        "--live-db", str(tmp_path / "live.db"),
    ])
    assert rc == 0
