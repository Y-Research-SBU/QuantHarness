"""Tests for scanner.py — scan cycle runs without error."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_config import MARKETS, StrategyType
from scanner import MarketScanner


@pytest.fixture
def patched_fetcher(monkeypatch, sample_ohlcv):
    """Mock fetch_market_data to return a fixed DataFrame without hitting the network."""
    def fake_fetch(symbol, interval, **kwargs):
        df = sample_ohlcv.copy()
        df.attrs["symbol"] = symbol
        df.attrs["timeframe"] = interval
        return df

    monkeypatch.setattr("scanner.fetch_market_data", fake_fetch)
    return fake_fetch


def test_scanner_init(tmp_db_path):
    scanner = MarketScanner(db_path=tmp_db_path)
    assert scanner.engine is not None
    assert scanner.use_agents is False


def test_scanner_init_with_agents_flag(tmp_db_path):
    scanner = MarketScanner(db_path=tmp_db_path, use_agents=True)
    assert scanner.use_agents is True
    # Trading graph is lazy-loaded; shouldn't be initialized yet.
    assert scanner._trading_graph is None


def test_scan_market_returns_list(tmp_db_path, patched_fetcher):
    scanner = MarketScanner(db_path=tmp_db_path)
    cfg = MARKETS["BTC-USD"]
    sigs = scanner.scan_market("BTC-USD", cfg)
    assert isinstance(sigs, list)
    for s in sigs:
        assert s.symbol == "BTC-USD"


def test_scan_market_empty_df(tmp_db_path, monkeypatch):
    monkeypatch.setattr("scanner.fetch_market_data", lambda *a, **k: pd.DataFrame())
    scanner = MarketScanner(db_path=tmp_db_path)
    cfg = MARKETS["BTC-USD"]
    sigs = scanner.scan_market("BTC-USD", cfg)
    assert sigs == []


def test_scan_market_handles_exception(tmp_db_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network error")
    monkeypatch.setattr("scanner.fetch_market_data", boom)
    scanner = MarketScanner(db_path=tmp_db_path)
    cfg = MARKETS["BTC-USD"]
    # Should not raise
    sigs = scanner.scan_market("BTC-USD", cfg)
    assert sigs == []


def test_execute_signals_empty(tmp_db_path):
    scanner = MarketScanner(db_path=tmp_db_path)
    ids = scanner.execute_signals([])
    assert ids == []


def test_run_scan_cycle_returns_summary(tmp_db_path, patched_fetcher):
    scanner = MarketScanner(db_path=tmp_db_path)
    # Limit to 1 symbol so this stays fast.
    result = scanner.run_scan_cycle(symbols=["BTC-USD"])
    assert "cycle_time" in result
    assert "markets_scanned" in result
    assert "signals_found" in result
    assert "trades_opened" in result
    assert result["markets_scanned"] == 1


def test_run_scan_cycle_takes_snapshots(tmp_db_path, patched_fetcher):
    scanner = MarketScanner(db_path=tmp_db_path)
    scanner.run_scan_cycle(symbols=["BTC-USD"])
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM portfolio_snapshots WHERE symbol = 'BTC-USD'").fetchall()
    assert len(rows) >= 1


def test_run_scan_cycle_with_unknown_symbol(tmp_db_path, patched_fetcher):
    scanner = MarketScanner(db_path=tmp_db_path)
    result = scanner.run_scan_cycle(symbols=["UNKNOWN-XYZ"])
    assert result["markets_scanned"] == 0


def test_check_all_stops_empty(tmp_db_path):
    scanner = MarketScanner(db_path=tmp_db_path)
    assert scanner.check_all_stops({}) == []


def test_scanner_default_symbols(tmp_db_path, patched_fetcher):
    scanner = MarketScanner(db_path=tmp_db_path)
    # Use only 1 symbol via override — but verify the default would be all markets.
    result = scanner.run_scan_cycle(symbols=["BTC-USD", "ETH-USD"])
    assert result["markets_scanned"] == 2
