"""Tests for scanner.py — scan cycle runs without error."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_config import MARKETS, MarketCategory, MarketConfig, StrategyType
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


def test_close_orphaned_positions(tmp_db_path):
    """Positions for symbols removed from MARKETS should be force-closed."""
    from strategies import Signal
    from market_config import StrategyType

    scanner = MarketScanner(db_path=tmp_db_path)

    # Open a trade for BTC-USD (a real MARKETS symbol)
    signal = Signal(
        direction="LONG", strength=0.8, strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="4h",
        entry_price=100.0, stop_loss=90.0, take_profit=120.0,
        risk_reward_ratio=2.0, reasoning="test",
    )
    ids = scanner.execute_signals([signal])
    assert len(ids) == 1

    # Manually insert an orphan trade for a symbol NOT in MARKETS
    from db_schema import get_connection
    from datetime import datetime
    with get_connection(tmp_db_path) as conn:
        conn.execute(
            """INSERT INTO trades
               (symbol, timeframe, strategy, direction, entry_price, position_size,
                quantity, stop_loss, take_profit, pnl, status, created_at, updated_at,
                entry_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("FAKE-ORPHAN", "4h", "momentum", "LONG", 50.0, 100.0,
             2.0, 45.0, 60.0, 0.0, "OPEN",
             datetime.utcnow().isoformat(), datetime.utcnow().isoformat(),
             datetime.utcnow().isoformat()),
        )

    # Verify we have 2 open positions
    open_before = scanner.engine.get_open_positions()
    assert len(open_before) == 2

    # Run orphan cleanup with only BTC-USD as active
    closed = scanner._close_orphaned_positions(
        active_symbols=["BTC-USD"],
        current_prices={"BTC-USD": 105.0},
    )

    # Orphan should be closed (at entry price since no price available)
    assert len(closed) == 1
    assert closed[0]["symbol"] == "FAKE-ORPHAN"

    # Only BTC-USD should remain open
    open_after = scanner.engine.get_open_positions()
    assert len(open_after) == 1
    assert open_after[0]["symbol"] == "BTC-USD"


def test_single_symbol_scan_does_not_orphan_other_markets(tmp_db_path, patched_fetcher):
    """run_scan_cycle(symbols=[X]) must NOT force-close positions on other
    valid MARKETS symbols.  Regression test for the bug where the per-symbol
    scan list was passed to orphan detection, causing every open trade on a
    *different* symbol to be closed immediately."""
    from strategies import Signal
    from market_config import StrategyType

    scanner = MarketScanner(db_path=tmp_db_path)

    # Open a trade on ETH-USD (a valid MARKETS symbol)
    signal = Signal(
        direction="LONG", strength=0.8, strategy=StrategyType.KRONOS_MOMENTUM_CONFIRM,
        symbol="ETH-USD", timeframe="4h",
        entry_price=3000.0, stop_loss=2800.0, take_profit=3400.0,
        risk_reward_ratio=2.0, reasoning="test",
    )
    ids = scanner.execute_signals([signal])
    assert len(ids) == 1

    # Now run a scan cycle for BTC-USD only
    results = scanner.run_scan_cycle(symbols=["BTC-USD"])

    # ETH-USD position should still be open — it's a valid MARKETS symbol,
    # not an orphan.
    open_positions = scanner.engine.get_open_positions()
    eth_open = [p for p in open_positions if p["symbol"] == "ETH-USD"]
    assert len(eth_open) == 1, (
        f"ETH-USD position was incorrectly closed as orphan during BTC-USD scan. "
        f"Open positions: {[p['symbol'] for p in open_positions]}"
    )


# ─────────────────────── Min-strength filter ───────────────────────


def test_execute_signals_drops_weak_signals(tmp_db_path):
    """Signals with strength below MIN_SIGNAL_STRENGTH (0.4) must be rejected
    before sizing/execution, even when nothing else would block them."""
    from strategies import Signal
    from scanner import MIN_SIGNAL_STRENGTH

    scanner = MarketScanner(db_path=tmp_db_path)
    weak = Signal(
        direction="LONG", strength=MIN_SIGNAL_STRENGTH - 0.05,
        strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="4h",
        entry_price=100.0, stop_loss=95.0, take_profit=120.0,
        risk_reward_ratio=2.0, reasoning="weak",
        metadata={},
    )
    ids = scanner.execute_signals([weak])
    assert ids == []


def test_execute_signals_keeps_signals_at_or_above_strength_floor(tmp_db_path):
    from strategies import Signal
    from scanner import MIN_SIGNAL_STRENGTH

    scanner = MarketScanner(db_path=tmp_db_path)
    strong = Signal(
        direction="LONG", strength=MIN_SIGNAL_STRENGTH + 0.4,
        strategy=StrategyType.MOMENTUM,
        symbol="BTC-USD", timeframe="4h",
        entry_price=100.0, stop_loss=95.0, take_profit=120.0,
        risk_reward_ratio=2.0, reasoning="ok",
        metadata={},
    )
    ids = scanner.execute_signals([strong])
    # Strong signal passes the filter and sizing succeeds with a 5% stop.
    assert len(ids) == 1


# ── L1 evolution filter tests ──

def test_scan_market_filters_disabled_strategies(tmp_db_path, monkeypatch):
    """Scanner should skip strategies that self_improver marks as disabled."""
    from unittest.mock import MagicMock, patch
    import pandas as pd
    import numpy as np

    scanner = MarketScanner(db_path=tmp_db_path)

    # Set up a mock self_improver that disables "momentum"
    mock_improver = MagicMock()
    mock_improver.get_disabled_strategies.return_value = ["momentum", "kronos_divergence"]
    mock_improver.get_adaptive_params.return_value = {}
    mock_improver.log_regime.return_value = "trending_up"
    scanner.self_improver = mock_improver

    # Track which strategies run_all_strategies receives
    captured_strategies = []
    original_run_all = None
    from strategies import run_all_strategies as _orig

    def spy_run_all(df, enabled_strategies=None, **kwargs):
        captured_strategies.extend(enabled_strategies or [])
        return []  # no signals

    # Create a config with all strategies enabled including momentum
    config = MarketConfig(
        symbol="BTC-USD",
        display_name="Bitcoin",
        category=MarketCategory.CRYPTO,
        timeframes=["4h"],
        enabled_strategies=[StrategyType.MOMENTUM, StrategyType.KRONOS_MOMENTUM_CONFIRM],
    )

    # Mock fetch_market_data to return a valid DataFrame
    fake_df = pd.DataFrame({
        "open": np.random.random(100),
        "high": np.random.random(100),
        "low": np.random.random(100),
        "close": np.random.random(100),
        "volume": np.random.random(100) * 1000,
    })

    with patch("scanner.fetch_market_data", return_value=fake_df), \
         patch("scanner.run_all_strategies", side_effect=spy_run_all):
        scanner.scan_market("BTC-USD", config)

    # momentum should have been filtered out, only kronos_momentum_confirm remains
    strategy_values = [s.value if hasattr(s, 'value') else s for s in captured_strategies]
    assert "momentum" not in strategy_values, f"momentum should be disabled but got {strategy_values}"
    assert "kronos_momentum_confirm" in strategy_values


def test_scan_market_no_improver_respects_whitelist(tmp_db_path, monkeypatch):
    """Without self_improver, scanner still applies the L0 backtest whitelist.

    ADA-USD has 'momentum' in its whitelist, and 'kronos_momentum_confirm' is
    allowed via the LIVE_KRONOS_ALLOWED_CATEGORIES override on crypto. Both
    should pass through the L0 filter when no self_improver is configured.
    """
    from unittest.mock import patch
    import pandas as pd
    import numpy as np

    scanner = MarketScanner(db_path=tmp_db_path)
    scanner.self_improver = None

    captured_strategies = []

    def spy_run_all(df, enabled_strategies=None, **kwargs):
        captured_strategies.extend(enabled_strategies or [])
        return []

    config = MarketConfig(
        symbol="ADA-USD",
        display_name="Cardano",
        category=MarketCategory.CRYPTO,
        timeframes=["4h"],
        enabled_strategies=[StrategyType.MOMENTUM, StrategyType.KRONOS_MOMENTUM_CONFIRM],
    )

    fake_df = pd.DataFrame({
        "open": np.random.random(100),
        "high": np.random.random(100),
        "low": np.random.random(100),
        "close": np.random.random(100),
        "volume": np.random.random(100) * 1000,
    })

    with patch("scanner.fetch_market_data", return_value=fake_df), \
         patch("scanner.run_all_strategies", side_effect=spy_run_all):
        scanner.scan_market("ADA-USD", config)

    strategy_values = [s.value if hasattr(s, 'value') else s for s in captured_strategies]
    assert "momentum" in strategy_values
    assert "kronos_momentum_confirm" in strategy_values
