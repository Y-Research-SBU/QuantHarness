"""Integration tests — exercise full pipelines end-to-end with mocked data.

These drive multiple subsystems at once (signal → sizing → execution →
stop/TP → close → P&L → snapshot) to catch issues the per-module unit tests
would miss.

No network or external APIs are used; yfinance and Kronos are stubbed out.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from db_schema import get_connection
from market_config import MARKETS, StrategyType
from paper_trading import PaperTradingEngine
from position_sizing import PositionSizeResult, calculate_position_size
from risk_manager import RiskManager
from scanner import MarketScanner
from strategies import Signal


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _mk_signal(
    symbol="BTC-USD",
    direction="LONG",
    entry=100.0,
    stop=95.0,
    tp=115.0,
    strategy=StrategyType.MOMENTUM,
    timeframe="1h",
) -> Signal:
    return Signal(
        direction=direction,
        strength=0.7,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        risk_reward_ratio=1.5,
        reasoning="integration test",
        metadata={},
    )


# ══════════════════════════════════════════════════════════════════════
# Full single-trade pipeline
# ══════════════════════════════════════════════════════════════════════


class TestFullTradePipeline:
    """Signal → position sizing → execute → stops → close → P&L."""

    def test_happy_path_long_trade_takes_profit(self, engine):
        """LONG trade reaches TP via check_stops → master balance increases by pnl."""
        signal = _mk_signal(entry=100.0, stop=90.0, tp=115.0)
        pos = calculate_position_size(
            portfolio_balance=10000.0,
            entry_price=100.0,
            stop_loss_price=90.0,
            direction="LONG",
        )
        assert pos.position_size_usd > 0

        start = engine.get_master_portfolio()["current_balance"]
        trade_id = engine.execute_trade(signal, pos)
        assert trade_id is not None

        # Price spikes through TP → check_stops closes at TP.
        closed = engine.check_stops({"BTC-USD": 120.0})
        assert len(closed) == 1
        assert closed[0]["exit_price"] == pytest.approx(pos.take_profit)
        assert closed[0]["pnl"] > 0
        assert closed[0]["status"] == "CLOSED"

        master = engine.get_master_portfolio()
        assert master["current_balance"] > start
        assert master["winning_trades"] == 1

    def test_long_trade_stops_out(self, engine, make_signal, make_position):
        """LONG trade hits stop → status STOPPED, loss recorded."""
        sig = make_signal(entry=100.0, stop=95.0, tp=110.0)
        pos = make_position(size=500.0, qty=5.0, stop=95.0, tp=110.0)
        trade_id = engine.execute_trade(sig, pos)

        closed = engine.check_stops({"BTC-USD": 94.0})
        assert len(closed) == 1
        assert closed[0]["status"] == "STOPPED"
        assert closed[0]["exit_price"] == pytest.approx(95.0)
        assert closed[0]["pnl"] < 0

        master = engine.get_master_portfolio()
        assert master["losing_trades"] == 1
        assert master["consecutive_losses"] == 1

    def test_short_trade_takes_profit(self, engine, make_signal, make_position):
        """SHORT TP: price must fall below tp, not above."""
        sig = make_signal(direction="SHORT", entry=100.0, stop=105.0, tp=90.0)
        pos = make_position(size=500.0, qty=5.0, stop=105.0, tp=90.0)
        engine.execute_trade(sig, pos)

        closed = engine.check_stops({"BTC-USD": 85.0})
        assert len(closed) == 1
        assert closed[0]["exit_price"] == pytest.approx(90.0)
        assert closed[0]["pnl"] > 0
        assert engine.get_master_portfolio()["winning_trades"] == 1

    def test_short_trade_stops_out_when_price_rises(self, engine, make_signal, make_position):
        """SHORT stop triggers when price RISES above stop."""
        sig = make_signal(direction="SHORT", entry=100.0, stop=105.0, tp=90.0)
        pos = make_position(size=500.0, qty=5.0, stop=105.0, tp=90.0)
        engine.execute_trade(sig, pos)

        closed = engine.check_stops({"BTC-USD": 106.0})
        assert len(closed) == 1
        assert closed[0]["status"] == "STOPPED"
        assert closed[0]["pnl"] < 0

    def test_check_stops_does_not_close_mid_range(self, engine, make_signal, make_position):
        """Price in the middle of stop/TP band: position stays open."""
        sig = make_signal(entry=100.0, stop=90.0, tp=115.0)
        pos = make_position(size=500.0, qty=5.0, stop=90.0, tp=115.0)
        engine.execute_trade(sig, pos)

        closed = engine.check_stops({"BTC-USD": 100.0})
        assert closed == []
        assert len(engine.get_open_positions("BTC-USD")) == 1

    def test_check_stops_ignores_unknown_symbols(self, engine, make_signal, make_position):
        """Position for a symbol with no current price stays open."""
        sig = make_signal(entry=100.0, stop=90.0, tp=115.0)
        pos = make_position(size=500.0, qty=5.0, stop=90.0, tp=115.0)
        engine.execute_trade(sig, pos)

        closed = engine.check_stops({"ETH-USD": 5000.0})
        assert closed == []
        assert len(engine.get_open_positions("BTC-USD")) == 1


# ══════════════════════════════════════════════════════════════════════
# Position stacking across multiple trades
# ══════════════════════════════════════════════════════════════════════


class TestPositionStacking:
    def test_multiple_trades_same_symbol_blocked(self, engine, make_signal, make_position):
        """Three attempts → first succeeds, rest blocked."""
        ids = [
            engine.execute_trade(make_signal(), make_position())
            for _ in range(3)
        ]
        assert ids[0] is not None
        assert ids[1] is None
        assert ids[2] is None
        assert len(engine.get_open_positions("BTC-USD")) == 1

    def test_stacking_across_different_symbols_allowed(self, engine, make_signal, make_position):
        """Different symbols → both trades open."""
        t1 = engine.execute_trade(
            make_signal(symbol="BTC-USD"), make_position(size=200.0, qty=2.0)
        )
        t2 = engine.execute_trade(
            make_signal(symbol="ETH-USD"), make_position(size=200.0, qty=2.0)
        )
        assert t1 is not None and t2 is not None
        assert len(engine.get_open_positions()) == 2

    def test_stacking_after_close_reopens(self, engine, make_signal, make_position):
        t1 = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
        engine.close_trade(t1, exit_price=105.0)
        t2 = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
        assert t2 is not None


# ══════════════════════════════════════════════════════════════════════
# Correlation block across crypto group
# ══════════════════════════════════════════════════════════════════════


class TestCorrelationAcrossGroup:
    def test_two_crypto_longs_allowed_third_blocked(self, engine, make_signal, make_position):
        """BTC long + ETH long allowed; SOL long blocked."""
        assert engine.execute_trade(
            make_signal(symbol="BTC-USD"), make_position(size=200.0, qty=2.0)
        ) is not None
        assert engine.execute_trade(
            make_signal(symbol="ETH-USD"), make_position(size=200.0, qty=2.0)
        ) is not None
        # Third same-direction crypto long should be blocked.
        assert engine.execute_trade(
            make_signal(symbol="SOL-USD"), make_position(size=200.0, qty=2.0)
        ) is None

    def test_closing_one_reopens_third_slot(self, engine, make_signal, make_position):
        t1 = engine.execute_trade(
            make_signal(symbol="BTC-USD"), make_position(size=200.0, qty=2.0)
        )
        engine.execute_trade(
            make_signal(symbol="ETH-USD"), make_position(size=200.0, qty=2.0)
        )
        # Close BTC → SOL should now be allowed.
        engine.close_trade(t1, exit_price=100.0)
        sol = engine.execute_trade(
            make_signal(symbol="SOL-USD"), make_position(size=200.0, qty=2.0)
        )
        assert sol is not None

    def test_opposite_directions_do_not_count_toward_group_limit(self, engine, make_signal, make_position):
        """BTC long + ETH short should leave room for SOL long."""
        engine.execute_trade(
            make_signal(symbol="BTC-USD", direction="LONG", entry=100.0, stop=95.0, tp=110.0),
            make_position(size=200.0, qty=2.0, stop=95.0, tp=110.0),
        )
        engine.execute_trade(
            make_signal(symbol="ETH-USD", direction="SHORT", entry=100.0, stop=105.0, tp=90.0),
            make_position(size=200.0, qty=2.0, stop=105.0, tp=90.0),
        )
        sol = engine.execute_trade(
            make_signal(symbol="SOL-USD", direction="LONG", entry=100.0, stop=95.0, tp=110.0),
            make_position(size=200.0, qty=2.0, stop=95.0, tp=110.0),
        )
        assert sol is not None


# ══════════════════════════════════════════════════════════════════════
# Equity tracking across multiple open/close cycles
# ══════════════════════════════════════════════════════════════════════


class TestEquityTracking:
    def test_equity_invariant_across_many_trades(self, engine, make_signal, make_position):
        """For many open/close cycles, master cash ≈ initial + realised_pnl."""
        initial = engine.get_master_portfolio()["initial_balance"]
        total_pnl = 0.0

        trades = [
            (100.0, 105.0),
            (100.0, 98.0),
            (100.0, 102.0),
            (100.0, 110.0),
            (100.0, 95.0),
        ]
        for entry, exit_ in trades:
            tid = engine.execute_trade(
                make_signal(entry=entry, stop=entry * 0.9, tp=entry * 1.2),
                make_position(size=500.0, qty=5.0, stop=entry * 0.9, tp=entry * 1.2),
            )
            result = engine.close_trade(tid, exit_price=exit_)
            total_pnl += result["pnl"]

        master = engine.get_master_portfolio()
        assert master["current_balance"] == pytest.approx(initial + total_pnl)
        assert master["total_pnl"] == pytest.approx(total_pnl)

    def test_peak_balance_tracks_high_water_mark(self, engine, make_signal, make_position):
        """Master peak_balance monotonically increases, never decreases."""
        t1 = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
        engine.close_trade(t1, exit_price=110.0)  # +$50
        p1 = engine.get_master_portfolio()["peak_balance"]
        assert p1 > 10000.0

        t2 = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
        engine.close_trade(t2, exit_price=90.0)  # -$50
        p2 = engine.get_master_portfolio()["peak_balance"]
        assert p2 == pytest.approx(p1)

    def test_max_drawdown_updated_from_peak(self, engine, make_signal, make_position):
        """Master max_drawdown reflects cumulative losses relative to high-water mark."""
        t1 = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
        engine.close_trade(t1, exit_price=115.0)  # +$75 → peak

        t2 = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
        engine.close_trade(t2, exit_price=85.0)  # -$75

        assert engine.get_master_portfolio()["max_drawdown"] > 0


# ══════════════════════════════════════════════════════════════════════
# Circuit breaker flow
# ══════════════════════════════════════════════════════════════════════


class TestCircuitBreakerFlow:
    def test_breaker_activates_after_large_loss_then_blocks_new_trade(
        self, engine, make_signal, make_position
    ):
        # Force a ~$1,200 loss: open size=$500 qty=5, close at price=-140 → pnl = (-140-100)*5 = -1200
        tid = engine.execute_trade(
            make_signal(entry=100.0, stop=90.0, tp=110.0),
            make_position(size=500.0, qty=5.0, stop=90.0, tp=110.0),
        )
        closed = engine.close_trade(tid, exit_price=-140.0)
        assert closed["pnl"] < -1000.0

        assert engine.get_master_portfolio()["is_circuit_breaker_active"] == 1

        # Trying to open a new trade should now fail at risk-check.
        new_id = engine.execute_trade(
            make_signal(entry=100.0, stop=95.0, tp=110.0),
            make_position(size=100.0, qty=1.0, stop=95.0, tp=110.0),
        )
        assert new_id is None

    def test_allocation_does_not_activate_breaker(self, engine, make_signal, make_position):
        """Opening a position (even one that gets clamped) does NOT flag the breaker."""
        engine.execute_trade(
            make_signal(),
            make_position(size=2500.0, qty=25.0),  # clamped to MAX_POSITION_SIZE
        )
        assert engine.get_master_portfolio()["is_circuit_breaker_active"] == 0


# ══════════════════════════════════════════════════════════════════════
# Snapshots & equity curve
# ══════════════════════════════════════════════════════════════════════


class TestSnapshots:
    def test_master_snapshot_uses_equity_not_cash(self, engine, make_signal, make_position):
        engine.execute_trade(make_signal(), make_position(size=400.0, qty=4.0))
        engine.take_snapshot(engine.MASTER_SYMBOL)

        with get_connection(engine.db_path) as conn:
            row = conn.execute(
                "SELECT balance FROM portfolio_snapshots WHERE symbol = ?",
                (engine.MASTER_SYMBOL,),
            ).fetchone()
        # Master cash (9600) + allocated (400) = equity 10000
        assert row["balance"] == pytest.approx(10000.0)

    def test_multiple_snapshots_accumulate(self, engine, make_signal, make_position):
        for _ in range(3):
            engine.take_snapshot("BTC-USD")

        with get_connection(engine.db_path) as conn:
            rows = conn.execute(
                "SELECT COUNT(*) as n FROM portfolio_snapshots WHERE symbol = 'BTC-USD'"
            ).fetchone()
        assert rows["n"] == 3


# ══════════════════════════════════════════════════════════════════════
# Full scan cycle
# ══════════════════════════════════════════════════════════════════════


def _uptrend_df(symbol: str, timeframe: str, n: int = 100) -> pd.DataFrame:
    """Deterministic uptrend data that should produce momentum signals."""
    np.random.seed(0)
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


class TestFullScanCycle:
    def test_full_scan_cycle_single_symbol(self, tmp_db_path, monkeypatch):
        """Scan → execute → snapshot, all against a mocked data fetcher."""
        def fake_fetch(symbol, interval, **kwargs):
            return _uptrend_df(symbol, interval)

        monkeypatch.setattr("scanner.fetch_market_data", fake_fetch)
        # Ensure Kronos isn't loaded (heavy import). Disable it on the scanner.
        scanner = MarketScanner(db_path=tmp_db_path, use_kronos=False)
        results = scanner.run_scan_cycle(symbols=["BTC-USD"])

        assert results["markets_scanned"] == 1
        assert "signals_found" in results
        # Snapshots should have been written for the scanned symbol.
        with get_connection(tmp_db_path) as conn:
            rows = conn.execute(
                "SELECT COUNT(*) AS n FROM portfolio_snapshots WHERE symbol = 'BTC-USD'"
            ).fetchone()
        assert rows["n"] >= 1

    def test_scan_cycle_with_no_data_is_safe(self, tmp_db_path, monkeypatch):
        """Empty DataFrames must not break the cycle or throw."""
        monkeypatch.setattr("scanner.fetch_market_data", lambda *a, **k: pd.DataFrame())
        scanner = MarketScanner(db_path=tmp_db_path, use_kronos=False)
        results = scanner.run_scan_cycle(symbols=["BTC-USD"])
        assert results["signals_found"] == 0
        assert results["trades_opened"] == 0

    def test_scan_cycle_handles_fetcher_exception(self, tmp_db_path, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("simulated outage")
        monkeypatch.setattr("scanner.fetch_market_data", boom)
        scanner = MarketScanner(db_path=tmp_db_path, use_kronos=False)
        # Should not raise.
        results = scanner.run_scan_cycle(symbols=["BTC-USD"])
        assert results["markets_scanned"] == 1


# ══════════════════════════════════════════════════════════════════════
# Position sizing ⇄ trade execution integration
# ══════════════════════════════════════════════════════════════════════


class TestPositionSizingWithExecution:
    def test_position_size_respects_max_position_pct(self, engine):
        """calculate_position_size should cap at max_position_pct; execute_trade
        can then consume it without exceeding cash.

        Stop is set to 97.0 (3%) — the minimum allowed for crypto by
        PaperTradingEngine.MIN_STOP_PCT. Kelly with a 3% stop and 0.8
        win rate / 3.0 RR still wants a position far above the 25% cap, so
        the cap is what binds (which is what this test exists to prove).
        """
        start = engine.get_master_portfolio()["current_balance"]
        pos = calculate_position_size(
            portfolio_balance=start,
            entry_price=100.0,
            stop_loss_price=97.0,  # Tight stop (at the crypto floor) → Kelly wants huge size.
            direction="LONG",
            win_rate=0.8,
            avg_win_loss_ratio=3.0,
            max_position_pct=0.25,
        )
        assert pos.position_size_usd <= start * 0.25 + 1e-6

        tid = engine.execute_trade(
            _mk_signal(entry=100.0, stop=97.0, tp=109.0), pos
        )
        assert tid is not None
        assert engine.get_master_portfolio()["current_balance"] >= 0

    def test_reduced_size_after_consecutive_losses(self, engine, make_signal, make_position):
        """After 3 consecutive losses, execute_trade must halve the position."""
        for _ in range(3):
            tid = engine.execute_trade(
                make_signal(entry=100.0, stop=95.0, tp=110.0),
                make_position(size=200.0, qty=2.0, stop=95.0, tp=110.0),
            )
            engine.close_trade(tid, exit_price=90.0)  # -$20

        assert engine.get_master_portfolio()["consecutive_losses"] == 3

        # Next trade: multiplier = 0.5 → $500 requested becomes $250.
        tid = engine.execute_trade(
            make_signal(entry=100.0, stop=95.0, tp=110.0),
            make_position(size=500.0, qty=5.0, stop=95.0, tp=110.0),
        )
        assert tid is not None
        open_pos = engine.get_open_positions("BTC-USD")[0]
        assert open_pos["position_size"] == pytest.approx(250.0)

    def test_streak_resets_after_win(self, engine, make_signal, make_position):
        """A single winning trade resets consecutive_losses to 0."""
        for _ in range(2):
            tid = engine.execute_trade(make_signal(), make_position(size=200.0, qty=2.0))
            engine.close_trade(tid, exit_price=90.0)
        assert engine.get_master_portfolio()["consecutive_losses"] == 2

        tid = engine.execute_trade(make_signal(), make_position(size=200.0, qty=2.0))
        engine.close_trade(tid, exit_price=110.0)
        assert engine.get_master_portfolio()["consecutive_losses"] == 0


# ══════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_close_nonexistent_trade_returns_none(self, engine):
        assert engine.close_trade(99999, exit_price=100.0) is None

    def test_cannot_close_already_closed_trade(self, engine, make_signal, make_position):
        tid = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
        engine.close_trade(tid, exit_price=110.0)
        # Second close attempt returns None.
        assert engine.close_trade(tid, exit_price=115.0) is None

    def test_execute_trade_clamps_oversize_down_to_max(self, engine, make_signal, make_position):
        """A requested $20k position is clamped down to MAX_POSITION_SIZE, not rejected."""
        tid = engine.execute_trade(
            make_signal(),
            make_position(size=20000.0, qty=200.0),
        )
        assert tid is not None
        open_pos = engine.get_open_positions("BTC-USD")[0]
        assert open_pos["position_size"] == pytest.approx(engine.MAX_POSITION_SIZE)

    def test_execute_trade_unknown_symbol_returns_none(self, engine, make_signal, make_position):
        tid = engine.execute_trade(
            make_signal(symbol="UNKNOWN-XYZ"),
            make_position(size=500.0, qty=5.0),
        )
        assert tid is None

    def test_empty_db_portfolios(self, tmp_db_path):
        """New engine creates portfolios for all known markets."""
        engine = PaperTradingEngine(db_path=tmp_db_path)
        assert len(engine.get_all_portfolios()) == len(MARKETS)

    def test_daily_pnl_reset(self, engine, make_signal, make_position):
        """reset_daily_pnl zeros the master-level daily counter."""
        tid = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
        engine.close_trade(tid, exit_price=110.0)
        assert engine.get_master_portfolio()["daily_pnl"] > 0

        engine.reset_daily_pnl()
        assert engine.get_master_portfolio()["daily_pnl"] == 0.0

    def test_zero_balance_position_sizing_returns_zero(self):
        pos = calculate_position_size(
            portfolio_balance=0.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            direction="LONG",
        )
        assert pos.position_size_usd == 0
        assert pos.quantity == 0

    def test_negative_pnl_from_long_when_price_below_entry(self, engine, make_signal, make_position):
        """A LONG closed below entry must have negative pnl."""
        tid = engine.execute_trade(
            make_signal(entry=100.0),
            make_position(size=500.0, qty=5.0),
        )
        result = engine.close_trade(tid, exit_price=80.0)
        assert result["pnl"] == pytest.approx((80.0 - 100.0) * 5.0)
        assert result["pnl"] < 0

    def test_positive_pnl_from_short_when_price_below_entry(self, engine, make_signal, make_position):
        """SHORT sign rule: pnl = (entry - exit) * qty."""
        tid = engine.execute_trade(
            make_signal(direction="SHORT", entry=100.0, stop=105.0, tp=90.0),
            make_position(size=500.0, qty=5.0, stop=105.0, tp=90.0),
        )
        result = engine.close_trade(tid, exit_price=80.0)
        assert result["pnl"] == pytest.approx((100.0 - 80.0) * 5.0)
        assert result["pnl"] > 0


# ══════════════════════════════════════════════════════════════════════
# Risk manager integrated with engine
# ══════════════════════════════════════════════════════════════════════


class TestRiskManagerIntegrated:
    def test_daily_loss_limit_halts_further_trades(self, engine, make_signal, make_position):
        """Once daily loss breaches 3%, no more trades are accepted."""
        tid = engine.execute_trade(
            make_signal(entry=100.0),
            make_position(size=500.0, qty=5.0),
        )
        engine.close_trade(tid, exit_price=30.0)  # -350
        assert engine.get_master_portfolio()["daily_pnl"] < -300.0

        # Next trade must be rejected.
        assert engine.execute_trade(
            make_signal(),
            make_position(size=100.0, qty=1.0),
        ) is None

    def test_drawdown_warning_does_not_block(self, engine, make_signal, make_position):
        """Drawdown in the warning band (8-10%) still allows trading."""
        tid = engine.execute_trade(
            make_signal(entry=100.0),
            make_position(size=500.0, qty=5.0),
        )
        engine.close_trade(tid, exit_price=-70.0)  # pnl = (-70-100)*5 = -850
        assert engine.get_master_portfolio()["is_circuit_breaker_active"] == 0

        engine.reset_daily_pnl()

        tid2 = engine.execute_trade(
            make_signal(),
            make_position(size=200.0, qty=2.0),
        )
        assert tid2 is not None


# ══════════════════════════════════════════════════════════════════════
# Strategy performance tracking
# ══════════════════════════════════════════════════════════════════════


class TestStrategyPerformance:
    def test_strategy_perf_tracks_wins_losses(self, engine, make_signal, make_position):
        """Multiple trades of the same strategy → aggregated in strategy_performance."""
        for exit_price in (110.0, 90.0, 115.0):  # win, loss, win
            tid = engine.execute_trade(
                make_signal(strategy=StrategyType.MOMENTUM),
                make_position(size=500.0, qty=5.0),
            )
            engine.close_trade(tid, exit_price=exit_price)

        with get_connection(engine.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM strategy_performance "
                "WHERE strategy='momentum' AND symbol='BTC-USD' AND timeframe='1h'"
            ).fetchone()
        assert row is not None
        assert row["total_trades"] == 3
        assert row["winning_trades"] == 2
        assert row["losing_trades"] == 1

    def test_strategies_are_isolated_per_strategy_row(self, engine, make_signal, make_position):
        """Different strategies get their own strategy_performance rows."""
        t1 = engine.execute_trade(
            make_signal(strategy=StrategyType.MOMENTUM),
            make_position(size=500.0, qty=5.0),
        )
        engine.close_trade(t1, exit_price=110.0)

        # Need to use a different symbol since same-symbol stacking is blocked.
        t2 = engine.execute_trade(
            make_signal(symbol="ETH-USD", strategy=StrategyType.BREAKOUT),
            make_position(size=500.0, qty=5.0),
        )
        engine.close_trade(t2, exit_price=90.0)

        with get_connection(engine.db_path) as conn:
            rows = conn.execute("SELECT strategy FROM strategy_performance").fetchall()
        strategies = {r["strategy"] for r in rows}
        assert strategies == {"momentum", "breakout"}
