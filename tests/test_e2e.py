"""End-to-end tests — full system lifecycle from fresh DB to dashboard.

These tests stitch together ContinuousRunner + MarketScanner + PaperTradingEngine
+ Flask dashboard against an in-memory set of mocked markets and verify the
whole stack produces consistent output.

No network calls. yfinance and Kronos are replaced with deterministic stubs.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import dashboard
from db_schema import get_connection, init_db
from market_config import MARKETS, StrategyType
from paper_trading import PaperTradingEngine
from run_continuous import ContinuousRunner
from scanner import MarketScanner
from strategies import Signal


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _uptrend_df(symbol: str, timeframe: str, n: int = 120) -> pd.DataFrame:
    np.random.seed(1)
    close = np.linspace(100, 130, n) + np.random.randn(n) * 0.1
    df = pd.DataFrame({
        "Datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h"),
        "Open": close - 0.1,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": np.full(n, 5000),
    })
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe
    return df


@pytest.fixture
def mocked_scanner_deps(monkeypatch):
    """Patch scanner.fetch_market_data to return deterministic uptrend data."""
    def fake_fetch(symbol, interval, **kwargs):
        bars = kwargs.get("bars")
        df = _uptrend_df(symbol, interval, n=(bars or 120))
        return df

    monkeypatch.setattr("scanner.fetch_market_data", fake_fetch)
    return fake_fetch


# ══════════════════════════════════════════════════════════════════════
# ContinuousRunner E2E
# ══════════════════════════════════════════════════════════════════════


class TestContinuousRunnerE2E:
    def test_two_cycles_write_snapshots(self, tmp_db_path, mocked_scanner_deps):
        """Run the runner for 2 scan cycles on a single symbol with a fake clock."""
        scanner = MarketScanner(db_path=tmp_db_path, use_kronos=False)

        # Fake clock that does not advance (prevents infinite loop).
        now = [1000.0]
        runner = ContinuousRunner(
            scanner=scanner,
            symbols=["BTC-USD"],
            summary_every_seconds=999999,
            sleep_seconds=0.0,
            clock=lambda: now[0],
            sleeper=lambda _s: None,
        )
        schedules = runner.run(max_cycles=2)
        assert schedules["BTC-USD"].total_cycles >= 1

        with get_connection(tmp_db_path) as conn:
            snap_count = conn.execute(
                "SELECT COUNT(*) AS n FROM portfolio_snapshots WHERE symbol='BTC-USD'"
            ).fetchone()["n"]
        assert snap_count >= 1

    def test_run_once_touches_every_market(self, tmp_db_path, mocked_scanner_deps):
        scanner = MarketScanner(db_path=tmp_db_path, use_kronos=False)
        runner = ContinuousRunner(
            scanner=scanner,
            symbols=["BTC-USD", "ETH-USD"],
            summary_every_seconds=999999,
            sleep_seconds=0.0,
        )
        schedules = runner.run_once()
        assert set(schedules.keys()) == {"BTC-USD", "ETH-USD"}

        with get_connection(tmp_db_path) as conn:
            syms = {
                r["symbol"]
                for r in conn.execute(
                    "SELECT DISTINCT symbol FROM portfolio_snapshots"
                ).fetchall()
            }
        assert "BTC-USD" in syms
        assert "ETH-USD" in syms


# ══════════════════════════════════════════════════════════════════════
# Full lifecycle: fresh DB → trades → closes → dashboard
# ══════════════════════════════════════════════════════════════════════


def _run_trade_session(db_path: str):
    """Simulate a trading session: open, close, open an unclosed position."""
    engine = PaperTradingEngine(db_path=db_path)

    def mk_signal(symbol, direction="LONG", entry=100.0, stop=95.0, tp=115.0):
        return Signal(
            direction=direction,
            strength=0.7,
            strategy=StrategyType.MOMENTUM,
            symbol=symbol,
            timeframe="1h",
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            risk_reward_ratio=1.5,
            reasoning="e2e",
            metadata={},
        )

    from position_sizing import PositionSizeResult
    def mk_pos(size=500.0, qty=5.0, stop=95.0, tp=115.0):
        return PositionSizeResult(
            position_size_usd=size,
            quantity=qty,
            risk_per_trade_usd=size * 0.02,
            risk_pct=0.02,
            kelly_fraction=0.1,
            half_kelly=0.05,
            stop_loss=stop,
            take_profit=tp,
            reason="e2e",
        )

    # Winning BTC trade
    tid = engine.execute_trade(mk_signal("BTC-USD"), mk_pos(size=500.0, qty=5.0))
    engine.close_trade(tid, exit_price=110.0)  # +$50

    # Losing ETH trade
    tid = engine.execute_trade(mk_signal("ETH-USD"), mk_pos(size=500.0, qty=5.0))
    engine.close_trade(tid, exit_price=95.0)  # -$25

    # Open SPY trade still live → unrealized P&L should show up.
    engine.execute_trade(mk_signal("SPY"), mk_pos(size=500.0, qty=5.0))

    # Snapshots
    for sym in ("BTC-USD", "ETH-USD", "SPY"):
        engine.take_snapshot(sym)

    return engine


class TestFullLifecycle:
    def test_fresh_db_to_dashboard(self, tmp_db_path, tmp_path):
        """Fresh DB → trade session → dashboard reports consistent numbers."""
        _run_trade_session(tmp_db_path)

        backtest_dir = tmp_path / "backtests"
        backtest_dir.mkdir()
        app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(backtest_dir))
        client = app.test_client()

        # /health
        r = client.get("/health")
        assert r.status_code == 200
        assert r.get_json()["db_exists"] is True

        # /api/overview
        r = client.get("/api/overview")
        overview = r.get_json()
        # total_balance is equity — should equal cash + open positions summed
        # across all portfolios.
        assert overview["total_balance"] > 0
        assert "realized_pnl" in overview
        assert "unrealized_pnl" in overview
        # Realised P&L = +50 - 25 = +25
        assert overview["realized_pnl"] == pytest.approx(25.0)
        # SPY trade is still open → unrealized_pnl is computed (could be 0
        # if no exit-price data yet, which is fine).
        assert isinstance(overview["unrealized_pnl"], float)
        assert overview["open_positions"] == 1

        # /api/markets
        r = client.get("/api/markets")
        cards = r.get_json()
        assert len(cards) == len(MARKETS)
        btc = next(c for c in cards if c["symbol"] == "BTC-USD")
        assert btc["realized_pnl"] == pytest.approx(50.0)
        eth = next(c for c in cards if c["symbol"] == "ETH-USD")
        assert eth["realized_pnl"] == pytest.approx(-25.0)

        # /api/trades
        r = client.get("/api/trades")
        trades = r.get_json()
        assert len(trades) >= 3
        statuses = {t["status"] for t in trades}
        assert "CLOSED" in statuses
        assert "OPEN" in statuses

        # /api/strategies
        r = client.get("/api/strategies")
        strategies = r.get_json()
        # Only momentum was used.
        strat_names = {s["strategy"] for s in strategies}
        assert "momentum" in strat_names

    def test_equity_consistent_between_overview_and_markets(self, tmp_db_path, tmp_path):
        """Accounting identity: total_balance (cash + allocated at cost)
        plus unrealized P&L equals initial capital + total P&L."""
        _run_trade_session(tmp_db_path)

        backtest_dir = tmp_path / "bt"
        backtest_dir.mkdir()
        app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(backtest_dir))
        client = app.test_client()

        overview = client.get("/api/overview").get_json()
        # total_balance = cash + open_position_sizes (at cost, no MTM)
        # total_pnl = realized + unrealized
        # Identity: total_balance + unrealized = initial + total_pnl
        #   because total_balance = initial + realized - allocated + allocated
        #                         = initial + realized
        #   and total_pnl = realized + unrealized
        #   so total_balance + unrealized = initial + realized + unrealized = initial + total_pnl ✓
        total_balance = float(overview["total_balance"])
        unrealized = float(overview.get("unrealized_pnl", 0))
        initial = float(overview["total_initial"])
        total_pnl = float(overview["total_pnl"])
        assert total_balance + unrealized == pytest.approx(initial + total_pnl, rel=1e-4)


# ══════════════════════════════════════════════════════════════════════
# Dashboard API shape & status codes
# ══════════════════════════════════════════════════════════════════════


class TestDashboardAPI:
    def test_health_endpoint(self, dashboard_client):
        r = dashboard_client.get("/health")
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    def test_api_overview_on_empty_db(self, dashboard_client):
        r = dashboard_client.get("/api/overview")
        assert r.status_code == 200
        data = r.get_json()
        # Empty DB → zero totals.
        assert data["total_balance"] == 0
        assert data["total_pnl"] == 0

    def test_api_markets_on_empty_db(self, dashboard_client):
        r = dashboard_client.get("/api/markets")
        assert r.status_code == 200
        cards = r.get_json()
        assert len(cards) == len(MARKETS)
        for c in cards:
            # With no data, the card falls back to initial_balance.
            assert c["current_balance"] >= 0

    def test_api_trades_on_empty_db(self, dashboard_client):
        r = dashboard_client.get("/api/trades")
        assert r.status_code == 200
        assert r.get_json() == []

    def test_api_trades_respects_limit(self, tmp_db_path, dashboard_client):
        # Seed a few trades.
        _run_trade_session(tmp_db_path)
        r = dashboard_client.get("/api/trades?limit=1")
        assert r.status_code == 200
        assert len(r.get_json()) <= 1

    def test_api_scanner_on_empty_db(self, dashboard_client):
        r = dashboard_client.get("/api/scanner")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_snapshots" in data
        assert data["total_snapshots"] == 0

    def test_api_backtests_empty(self, dashboard_client):
        r = dashboard_client.get("/api/backtests")
        assert r.status_code == 200
        assert r.get_json() == []


# ══════════════════════════════════════════════════════════════════════
# DB sync endpoint
# ══════════════════════════════════════════════════════════════════════


class TestDBSyncEndpoint:
    def test_sync_endpoint_uploads_db_and_serves_data(self, tmp_db_path, tmp_path):
        """POST /api/sync with a DB file → new data is served by subsequent GETs."""
        # Prepare a source DB populated with trades.
        source_db = tmp_path / "source.db"
        _run_trade_session(str(source_db))

        # Target DB (empty) for the dashboard.
        target_db = tmp_path / "target.db"
        init_db(str(target_db))

        backtest_dir = tmp_path / "bt"
        backtest_dir.mkdir()
        app = dashboard.create_app(db_path=str(target_db), backtest_dir=str(backtest_dir))
        client = app.test_client()

        # Before sync: no trades.
        r = client.get("/api/trades")
        assert r.get_json() == []

        # Upload.
        with open(source_db, "rb") as fh:
            payload = fh.read()
        r = client.post(
            "/api/sync",
            data={"db": (io.BytesIO(payload), "upload.db")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

        # After sync: trades show up.
        r = client.get("/api/trades")
        assert len(r.get_json()) >= 3

    def test_sync_rejects_missing_file(self, dashboard_client):
        r = dashboard_client.post("/api/sync", data={})
        assert r.status_code == 400

    def test_sync_respects_auth_token(self, tmp_db_path, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNC_TOKEN", "secret")

        backtest_dir = tmp_path / "bt"
        backtest_dir.mkdir()
        app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(backtest_dir))
        client = app.test_client()

        # No token → unauthorized.
        r = client.post(
            "/api/sync",
            data={"db": (io.BytesIO(b"dummy"), "x.db")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 401

        # Correct token → ok.
        r = client.post(
            "/api/sync",
            data={"db": (io.BytesIO(b"dummy"), "x.db")},
            content_type="multipart/form-data",
            headers={"X-Sync-Token": "secret"},
        )
        assert r.status_code == 200

    def test_sync_writes_per_instance_db_and_state(self, tmp_db_path, tmp_path):
        """POST /api/sync with instance_db_<name> + instance_state_<name> →
        dashboard's tournament leaderboard reflects the synced heartbeat.
        """
        # Prepare a fake instance dir with profile.py so ProfileLoader picks it up.
        instance_root = tmp_path / "isolated"
        instance_root.mkdir()
        (instance_root / "instances").mkdir()
        baseline_dir = instance_root / "instances" / "baseline"
        baseline_dir.mkdir()
        # Minimal profile.py that re-uses the real Profile dataclass.
        (baseline_dir / "profile.py").write_text(
            "from instances.profile import Profile\n"
            "PROFILE = Profile(name='baseline', db_path='paper_trades_baseline.db')\n"
        )

        backtest_dir = tmp_path / "bt"
        backtest_dir.mkdir()
        app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(backtest_dir))
        app.config["INSTANCE_ROOT"] = str(instance_root)
        client = app.test_client()

        # Build a dummy DB file (raw bytes are fine — we only check it lands).
        db_payload = b"SQLite format 3\x00" + b"x" * 64
        state_payload = json.dumps({
            "pid": 12345,
            "last_heartbeat": "2026-04-26T07:00:00",
            "equity": 9999.5,
            "open_positions": 4,
            "realized_pnl": -0.5,
            "total_trades": 7,
            "db_path": "paper_trades_baseline.db",
        }).encode()

        r = client.post(
            "/api/sync",
            data={
                "instance_db_baseline": (io.BytesIO(db_payload), "x.db"),
                "instance_state_baseline": (io.BytesIO(state_payload), "state.json"),
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body["status"] == "ok"
        assert "instance_db_baseline" in body["synced"]
        assert "instance_state_baseline" in body["synced"]

        # Files actually landed on disk under the redirected root.
        landed_db = instance_root / "paper_trades_baseline.db"
        landed_state = instance_root / "instances" / "baseline" / "state.json"
        assert landed_db.exists()
        assert landed_state.exists()
        state = json.loads(landed_state.read_text())
        assert state["pid"] == 12345
        assert state["equity"] == 9999.5

    def test_sync_rejects_unknown_instance(self, tmp_db_path, tmp_path):
        """Unknown instance names must be silently dropped (no path traversal)."""
        instance_root = tmp_path / "isolated"
        (instance_root / "instances").mkdir(parents=True)
        # Note: no profile.py created → ProfileLoader sees zero valid instances.

        backtest_dir = tmp_path / "bt"
        backtest_dir.mkdir()
        app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(backtest_dir))
        app.config["INSTANCE_ROOT"] = str(instance_root)
        client = app.test_client()

        r = client.post(
            "/api/sync",
            data={
                "instance_db_evil": (io.BytesIO(b"x"), "x.db"),
                "instance_state_../../etc/passwd": (io.BytesIO(b"x"), "x.json"),
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 400  # nothing recognized landed
        # Critically: no file should have been written.
        assert not (instance_root / "paper_trades_evil.db").exists()


# ══════════════════════════════════════════════════════════════════════
# Cross-module: scanner → engine → dashboard
# ══════════════════════════════════════════════════════════════════════


class TestScannerToDashboard:
    def test_scan_cycle_produces_dashboard_visible_snapshots(
        self, tmp_db_path, tmp_path, mocked_scanner_deps
    ):
        scanner = MarketScanner(db_path=tmp_db_path, use_kronos=False)
        scanner.run_scan_cycle(symbols=["BTC-USD"])

        backtest_dir = tmp_path / "bt"
        backtest_dir.mkdir()
        app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(backtest_dir))
        client = app.test_client()

        scanner_status = client.get("/api/scanner").get_json()
        assert scanner_status["total_snapshots"] >= 1


# ══════════════════════════════════════════════════════════════════════
# Data consistency after sequence of operations
# ══════════════════════════════════════════════════════════════════════


class TestDataConsistency:
    def test_no_trades_means_no_negative_balances(self, tmp_db_path):
        """Freshly created portfolios must have non-negative balances."""
        engine = PaperTradingEngine(db_path=tmp_db_path)
        for p in engine.get_all_portfolios():
            assert p["current_balance"] >= 0
            assert p["peak_balance"] >= 0

    def test_dashboard_never_returns_nan_or_inf(self, tmp_db_path, tmp_path):
        """After a trade session, no dashboard number is NaN/Inf."""
        _run_trade_session(tmp_db_path)

        backtest_dir = tmp_path / "bt"
        backtest_dir.mkdir()
        app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(backtest_dir))
        client = app.test_client()

        for endpoint in ("/api/overview", "/api/markets", "/api/trades",
                         "/api/strategies", "/api/scanner"):
            data = client.get(endpoint).get_json()
            _assert_no_nan(data)


def _assert_no_nan(obj):
    """Recursively verify no NaN/Inf in a JSON-like structure."""
    if isinstance(obj, float):
        assert obj == obj  # NaN != NaN
        assert obj != float("inf")
        assert obj != float("-inf")
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_nan(item)
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_nan(v)


# ══════════════════════════════════════════════════════════════════════
# Runner performance summary
# ══════════════════════════════════════════════════════════════════════


class TestRunnerPerformanceSummary:
    def test_performance_summary_structure(self, tmp_db_path):
        scanner = MarketScanner(db_path=tmp_db_path, use_kronos=False)
        runner = ContinuousRunner(scanner=scanner, symbols=["BTC-USD"])
        summary = runner.performance_summary()
        assert "generated_at" in summary
        assert "portfolios" in summary
        assert "totals" in summary
        assert isinstance(summary["portfolios"], list)
