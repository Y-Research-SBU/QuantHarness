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
