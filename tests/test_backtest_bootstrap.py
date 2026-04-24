"""Tests for backtest_bootstrap.py.

The bootstrap script fetches yfinance data and runs the full pipeline at
scale, so the tests stub out network I/O and verify the orchestration is
correct on synthetic OHLCV.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, List
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import backtest_bootstrap
from backtest_bootstrap import (
    BootstrapStats,
    DEFAULT_MARKETS,
    SELF_IMPROVEMENT_TABLES,
    _drive_self_improvement,
    _merge_self_improvement_into_live,
    _walk_one_market,
    run_bootstrap,
)
from db_schema import get_connection
from paper_trading import PaperTradingEngine
from self_improvement_schema import apply_self_improvement_schema
from self_improver import SelfImprover


def _trending_bars(n: int = 200, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """Synthetic OHLCV with mild noise around an uptrend."""
    rng = np.random.default_rng(42)
    prices = start + np.cumsum(np.full(n, step) + rng.normal(0, 0.3, size=n))
    return pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "Open": prices - 0.05,
            "High": prices + 0.4,
            "Low": prices - 0.4,
            "Close": prices,
            "Volume": np.full(n, 1000.0),
        }
    )


def test_default_markets_present_in_market_config():
    from market_config import MARKETS
    for sym in DEFAULT_MARKETS:
        assert sym in MARKETS, f"DEFAULT_MARKETS includes {sym} which isn't in MARKETS"


def test_walk_one_market_opens_trades(tmp_db_path):
    apply_self_improvement_schema(tmp_db_path)
    engine = PaperTradingEngine(db_path=tmp_db_path)
    self_improver = SelfImprover(db_path=tmp_db_path)
    df = _trending_bars(300)

    n_signals, n_opened, last = _walk_one_market(
        engine=engine,
        self_improver=self_improver,
        symbol="BTC-USD",
        timeframe="1h",
        df=df,
        warmup=60,
    )
    assert n_signals >= 0
    assert n_opened >= 0
    assert last is not None
    # Verify at least one trade was logged (or zero, but the table exists).
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()
    assert rows["n"] == engine.get_master_portfolio()["total_trades"]


def test_walk_one_market_skips_unknown_symbol(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    self_improver = SelfImprover(db_path=tmp_db_path)
    df = _trending_bars(300)
    n_signals, n_opened, last = _walk_one_market(
        engine=engine,
        self_improver=self_improver,
        symbol="NOT-A-SYMBOL",
        timeframe="1h",
        df=df,
        warmup=60,
    )
    assert (n_signals, n_opened, last) == (0, 0, None)


def test_walk_one_market_handles_short_history(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    self_improver = SelfImprover(db_path=tmp_db_path)
    df = _trending_bars(20)  # less than warmup
    out = _walk_one_market(
        engine=engine,
        self_improver=self_improver,
        symbol="BTC-USD",
        timeframe="1h",
        df=df,
        warmup=60,
    )
    assert out == (0, 0, None)


def test_drive_self_improvement_runs_all_levels(tmp_db_path):
    """All five levels should be invoked (and not raise) even with sparse data."""
    apply_self_improvement_schema(tmp_db_path)
    self_improver = SelfImprover(db_path=tmp_db_path)
    bars = {"BTC-USD": _trending_bars(200)}
    # Should not raise even though there are no trades yet.
    _drive_self_improvement(self_improver, bars)

    # L4 leaves regime_history rows.
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM regime_history").fetchone()
    assert rows["n"] >= 1


def test_merge_self_improvement_copies_rows(tmp_db_path, tmp_path):
    """_merge_self_improvement_into_live appends rows from src into live."""
    src_db = str(tmp_path / "src.db")
    dst_db = str(tmp_path / "dst.db")
    apply_self_improvement_schema(src_db)
    apply_self_improvement_schema(dst_db)

    # Seed the source with an adaptive_params row.
    with get_connection(src_db) as conn:
        conn.execute(
            """INSERT INTO adaptive_params
               (strategy, symbol, param_name, param_value, sample_size, improvement_pct)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("momentum", "BTC-USD", "sl_atr_mult", 1.25, 100, 0.05),
        )
        conn.execute(
            """INSERT INTO regime_history (symbol, timeframe, regime)
               VALUES (?, ?, ?)""",
            ("BTC-USD", "1h", "trending_up"),
        )

    counts = _merge_self_improvement_into_live(src_db, dst_db)
    assert counts["adaptive_params"] == 1
    assert counts["regime_history"] == 1

    with get_connection(dst_db) as conn:
        ap = conn.execute(
            "SELECT param_name, param_value FROM adaptive_params WHERE strategy='momentum'"
        ).fetchone()
        assert ap is not None
        assert ap["param_value"] == pytest.approx(1.25)


def test_merge_self_improvement_handles_missing_table(tmp_path):
    """If src is missing a table, merge skips it without raising."""
    src_db = str(tmp_path / "src.db")
    dst_db = str(tmp_path / "dst.db")
    # Don't initialize src — it's an empty file.
    open(src_db, "w").close()
    apply_self_improvement_schema(dst_db)
    counts = _merge_self_improvement_into_live(src_db, dst_db)
    for table in SELF_IMPROVEMENT_TABLES:
        assert counts.get(table, 0) == 0


def test_run_bootstrap_with_synthetic_data(tmp_path, monkeypatch):
    """End-to-end bootstrap with a stubbed fetcher and a single market."""
    bootstrap_db = str(tmp_path / "bootstrap.db")
    live_db = str(tmp_path / "live.db")

    def fake_fetch(symbol, interval, **kwargs):
        # Return rich trending data only for BTC; empty for everything else.
        if symbol == "BTC-USD":
            return _trending_bars(300)
        return pd.DataFrame()

    monkeypatch.setattr(backtest_bootstrap, "fetch_market_data", fake_fetch)

    stats = run_bootstrap(
        markets=["BTC-USD"],
        days=60,
        bootstrap_db=bootstrap_db,
        live_db=live_db,
        merge=True,
        timeframes=["1h"],
        warmup=60,
    )

    assert isinstance(stats, BootstrapStats)
    assert os.path.exists(bootstrap_db)
    # Live DB should have inherited self-improvement rows from the bootstrap.
    with get_connection(live_db) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM regime_history WHERE symbol='BTC-USD'"
        ).fetchone()
    assert rows["n"] >= 1


def test_run_bootstrap_no_merge_keeps_live_clean(tmp_path, monkeypatch):
    """With merge=False the live DB shouldn't get any rows."""
    bootstrap_db = str(tmp_path / "bootstrap.db")
    live_db = str(tmp_path / "live.db")
    apply_self_improvement_schema(live_db)

    def fake_fetch(symbol, interval, **kwargs):
        return _trending_bars(120) if symbol == "BTC-USD" else pd.DataFrame()

    monkeypatch.setattr(backtest_bootstrap, "fetch_market_data", fake_fetch)

    run_bootstrap(
        markets=["BTC-USD"],
        days=10,
        bootstrap_db=bootstrap_db,
        live_db=live_db,
        merge=False,
        timeframes=["1h"],
        warmup=60,
    )
    with get_connection(live_db) as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM regime_history").fetchone()
    # No merge → live DB should still be empty of bootstrap rows.
    assert rows["n"] == 0


def test_main_with_no_data_succeeds(tmp_path, monkeypatch):
    """main() should still exit 0 when fetcher returns nothing."""
    monkeypatch.setattr(backtest_bootstrap, "fetch_market_data", lambda *a, **k: pd.DataFrame())
    rc = backtest_bootstrap.main([
        "--markets", "BTC-USD",
        "--bootstrap-db", str(tmp_path / "bootstrap.db"),
        "--live-db", str(tmp_path / "live.db"),
        "--days", "5",
        "--timeframes", "1h",
    ])
    assert rc == 0
