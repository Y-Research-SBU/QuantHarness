"""Tests for scanner_controller.py — background scanner lifecycle."""

import time

import pandas as pd
import pytest

from scanner_controller import ScannerController, get_controller, reset_controller


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_controller()
    yield
    reset_controller()


@pytest.fixture
def no_network(monkeypatch):
    """Patch fetch_market_data so the scanner doesn't hit yfinance during tests."""
    monkeypatch.setattr("scanner.fetch_market_data", lambda *a, **k: pd.DataFrame())


def test_controller_init_not_running(tmp_db_path):
    c = ScannerController(db_path=tmp_db_path)
    assert c.is_running() is False


def test_controller_start_and_stop(tmp_db_path, no_network):
    c = ScannerController(db_path=tmp_db_path)
    result = c.start(interval_seconds=3600, use_agents=False)
    assert result["started"] is True
    assert c.is_running() is True

    result = c.stop()
    assert result["stopped"] is True
    # Give the thread a moment to exit cleanly.
    time.sleep(0.2)
    assert c.is_running() is False


def test_controller_double_start(tmp_db_path, no_network):
    c = ScannerController(db_path=tmp_db_path)
    c.start(interval_seconds=3600)
    result2 = c.start(interval_seconds=3600)
    assert result2["started"] is False
    assert result2["reason"] == "already_running"
    c.stop()


def test_controller_stop_when_not_running(tmp_db_path):
    c = ScannerController(db_path=tmp_db_path)
    result = c.stop()
    assert result["stopped"] is False


def test_controller_status_structure(tmp_db_path):
    c = ScannerController(db_path=tmp_db_path)
    status = c.status()
    assert "scanner" in status
    assert "summary" in status
    assert "running" in status["scanner"]


def test_controller_status_summary_has_portfolios(tmp_db_path):
    c = ScannerController(db_path=tmp_db_path)
    status = c.status()
    summary = status["summary"]
    # 12 configured markets → 12 portfolios
    assert summary["portfolios"] >= 1
    assert summary["total_initial"] > 0


def test_run_once_synchronous(tmp_db_path, no_network):
    c = ScannerController(db_path=tmp_db_path)
    result = c.run_once()
    assert "markets_scanned" in result
    assert "signals_found" in result


def test_get_controller_returns_singleton(tmp_db_path):
    c1 = get_controller(db_path=tmp_db_path)
    c2 = get_controller()
    assert c1 is c2


def test_reset_controller_releases_singleton(tmp_db_path):
    c1 = get_controller(db_path=tmp_db_path)
    reset_controller()
    c2 = get_controller(db_path=tmp_db_path)
    assert c1 is not c2


# ---------------------------------------------------------------------------
# Regression: status summary must report master capital, not per-symbol zeros.
# (REL-379) ``get_all_portfolios`` excludes the master row and per-symbol
# rows are seeded with ``initial_balance=0.0``. Summing only those rows would
# report total capital as $0 in the dashboard.
# ---------------------------------------------------------------------------


def test_status_summary_reports_master_capital(tmp_db_path):
    """total_initial / total_balance must come from the master portfolio."""
    from paper_trading import PaperTradingEngine

    c = ScannerController(db_path=tmp_db_path)
    summary = c.status()["summary"]

    assert summary["total_initial"] == pytest.approx(
        PaperTradingEngine.MASTER_INITIAL_BALANCE
    )
    assert summary["total_balance"] == pytest.approx(
        PaperTradingEngine.MASTER_INITIAL_BALANCE
    )
    # Per-symbol analytics rows still reported as portfolios for breakdown.
    assert summary["portfolios"] >= 1
    assert summary["total_trades"] == 0
    assert summary["total_pnl"] == 0.0


def test_status_summary_does_not_double_count_trades(tmp_db_path):
    """After one closed trade, totals must equal the master row — not 2×.

    ``execute_trade`` and ``close_trade`` bump both the master row and the
    per-symbol analytics row in lockstep. The status summary previously
    summed the per-symbol rows; combining that with the master row would
    double-count. We assert the summary equals the master row exactly.
    """
    from paper_trading import PaperTradingEngine
    from strategies import Signal
    from market_config import StrategyType
    from position_sizing import PositionSizeResult

    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = Signal(
        direction="LONG", strength=0.7, strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="1h", entry_price=100.0,
        stop_loss=96.0, take_profit=108.0, risk_reward_ratio=2.0,
        reasoning="regression", metadata={"atr_pct": 0.04},
    )
    pos = PositionSizeResult(
        position_size_usd=500.0, quantity=5.0, risk_per_trade_usd=10.0,
        risk_pct=0.02, kelly_fraction=0.2, half_kelly=0.1,
        stop_loss=96.0, take_profit=108.0, reason="regression",
    )
    tid = engine.execute_trade(sig, pos)
    assert tid is not None
    engine.close_trade(tid, exit_price=108.0)  # +$40 win

    c = ScannerController(db_path=tmp_db_path)
    summary = c.status()["summary"]
    master = engine.get_master_portfolio()

    assert summary["total_trades"] == master["total_trades"] == 1
    assert summary["winning_trades"] == master["winning_trades"] == 1
    assert summary["total_pnl"] == pytest.approx(master["total_pnl"])
    assert summary["total_balance"] == pytest.approx(master["current_balance"])
