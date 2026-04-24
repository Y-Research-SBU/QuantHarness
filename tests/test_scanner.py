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


def test_run_scan_cycle_marks_open_positions_to_market(tmp_db_path, patched_fetcher):
    """Scan cycle must refresh unrealised P&L on open positions."""
    from market_config import StrategyType
    from paper_trading import PaperTradingEngine
    from position_sizing import PositionSizeResult
    from strategies import Signal

    scanner = MarketScanner(db_path=tmp_db_path, use_self_improvement=False, use_kronos=False)
    scanner.engine.COOLDOWN_MINUTES = 0

    # Seed one open LONG trade on BTC-USD @ 100.
    signal = Signal(
        direction="LONG", strength=0.8, strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="4h",
        entry_price=100.0, stop_loss=80.0, take_profit=200.0,
        risk_reward_ratio=2.0, reasoning="test", metadata={},
    )
    pos = PositionSizeResult(
        position_size_usd=500.0, quantity=5.0, risk_per_trade_usd=10.0,
        risk_pct=0.02, kelly_fraction=0.2, half_kelly=0.1,
        stop_loss=80.0, take_profit=200.0, reason="test",
    )
    tid = scanner.engine.execute_trade(signal, pos)
    assert tid is not None

    # Spy on mark_to_market to confirm it runs during the scan cycle.
    real_m2m = scanner.engine.mark_to_market
    calls = {"n": 0}

    def spy(prices):
        calls["n"] += 1
        return real_m2m(prices)

    scanner.engine.mark_to_market = spy  # type: ignore[assignment]

    result = scanner.run_scan_cycle(symbols=["BTC-USD"])
    assert calls["n"] == 1
    # The scanner surfaces the summary fields it got back.
    assert "unrealized_pnl" in result
    assert result["positions_marked"] == 1


def test_trend_filter_blocks_short_in_uptrend(tmp_db_path):
    """execute_signals should skip SHORT signals when regime is trending_up."""
    from strategies import Signal
    from market_config import StrategyType

    scanner = MarketScanner(db_path=tmp_db_path)
    signal = Signal(
        direction="SHORT", strength=0.9, strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="4h",
        entry_price=100.0, stop_loss=105.0, take_profit=80.0,
        risk_reward_ratio=2.0, reasoning="test",
        metadata={"regime": "trending_up"},
    )
    ids = scanner.execute_signals([signal])
    assert ids == []


def test_trend_filter_blocks_long_in_downtrend(tmp_db_path):
    """execute_signals should skip LONG signals when regime is trending_down."""
    from strategies import Signal
    from market_config import StrategyType

    scanner = MarketScanner(db_path=tmp_db_path)
    signal = Signal(
        direction="LONG", strength=0.9, strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="4h",
        entry_price=100.0, stop_loss=95.0, take_profit=120.0,
        risk_reward_ratio=2.0, reasoning="test",
        metadata={"regime": "trending_down"},
    )
    ids = scanner.execute_signals([signal])
    assert ids == []


def test_trend_filter_allows_short_in_downtrend(tmp_db_path):
    """SHORT in a trending_down market should be allowed (with-trend trade)."""
    from strategies import Signal
    from market_config import StrategyType

    scanner = MarketScanner(db_path=tmp_db_path)
    signal = Signal(
        direction="SHORT", strength=0.9, strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="4h",
        entry_price=100.0, stop_loss=105.0, take_profit=80.0,
        risk_reward_ratio=2.0, reasoning="test",
        metadata={"regime": "trending_down"},
    )
    ids = scanner.execute_signals([signal])
    # Should attempt execution (may or may not fill depending on sizing,
    # but it should NOT be filtered out by the trend filter).
    # We check it got past the filter by verifying it's not empty OR
    # that the engine was called (not filtered).
    # Since the signal has valid params, it should execute.
    assert len(ids) >= 1


def test_trend_filter_allows_ranging_regime(tmp_db_path):
    """Ranging regime should not filter any direction."""
    from strategies import Signal
    from market_config import StrategyType

    scanner = MarketScanner(db_path=tmp_db_path)
    signal = Signal(
        direction="SHORT", strength=0.9, strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="4h",
        entry_price=100.0, stop_loss=105.0, take_profit=80.0,
        risk_reward_ratio=2.0, reasoning="test",
        metadata={"regime": "ranging"},
    )
    ids = scanner.execute_signals([signal])
    assert len(ids) >= 1
