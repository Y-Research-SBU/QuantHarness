"""Tests for paper_trading.py — open/close trades, P&L, snapshots."""

import pytest

from market_config import StrategyType
from paper_trading import PaperTradingEngine
from position_sizing import PositionSizeResult
from strategies import Signal


def _mk_signal(
    symbol: str = "BTC-USD",
    direction: str = "LONG",
    entry: float = 100.0,
    stop: float = 95.0,
    tp: float = 110.0,
    timeframe: str = "4h",
    strategy: StrategyType = StrategyType.MOMENTUM,
) -> Signal:
    return Signal(
        direction=direction,
        strength=0.8,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        risk_reward_ratio=2.0,
        reasoning="test",
        metadata={},
    )


def _mk_pos(size: float = 500.0, qty: float = 5.0, stop: float = 95.0, tp: float = 110.0) -> PositionSizeResult:
    return PositionSizeResult(
        position_size_usd=size,
        quantity=qty,
        risk_per_trade_usd=size * 0.02,
        risk_pct=0.02,
        kelly_fraction=0.2,
        half_kelly=0.1,
        stop_loss=stop,
        take_profit=tp,
        reason="test",
    )


# ─────────────────────── Initialization ───────────────────────


def test_engine_creates_portfolios_for_all_markets(tmp_db_path):
    from market_config import MARKETS
    engine = PaperTradingEngine(db_path=tmp_db_path)
    # get_all_portfolios excludes the master row; one row per configured market.
    pfs = engine.get_all_portfolios()
    assert len(pfs) == len(MARKETS)


def test_master_portfolio_has_initial_balance(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    master = engine.get_master_portfolio()
    assert master["symbol"] == PaperTradingEngine.MASTER_SYMBOL
    assert master["initial_balance"] == PaperTradingEngine.MASTER_INITIAL_BALANCE
    assert master["current_balance"] == PaperTradingEngine.MASTER_INITIAL_BALANCE


def test_per_symbol_rows_hold_no_capital(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    for p in engine.get_all_portfolios():
        assert p["initial_balance"] == 0.0
        assert p["current_balance"] == 0.0


def test_get_portfolio_by_symbol(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    p = engine.get_portfolio("BTC-USD")
    assert p is not None
    assert p["symbol"] == "BTC-USD"


def test_get_portfolio_unknown_returns_none(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    assert engine.get_portfolio("XYZ-UNKNOWN") is None


# ─────────────────────── Execute trade ───────────────────────


def test_execute_trade_opens_position(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    trade_id = engine.execute_trade(_mk_signal(), _mk_pos())
    assert trade_id is not None
    open_pos = engine.get_open_positions("BTC-USD")
    assert len(open_pos) == 1
    assert open_pos[0]["id"] == trade_id
    assert open_pos[0]["status"] == "OPEN"


def test_execute_trade_deducts_master_balance(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    before = engine.get_master_portfolio()["current_balance"]
    engine.execute_trade(_mk_signal(), _mk_pos(size=500))
    after = engine.get_master_portfolio()["current_balance"]
    assert after == pytest.approx(before - 500)


def test_execute_trade_increments_total_trades(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_mk_signal(), _mk_pos())
    # Per-symbol analytics track the trade count.
    assert engine.get_portfolio("BTC-USD")["total_trades"] == 1
    # Master tracks the global count too.
    assert engine.get_master_portfolio()["total_trades"] == 1


def test_execute_trade_unknown_symbol(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(symbol="UNKNOWN-XYZ")
    trade_id = engine.execute_trade(sig, _mk_pos())
    assert trade_id is None


def test_execute_trade_blocked_by_correlation(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    # First trade: LONG BTC
    engine.execute_trade(_mk_signal(symbol="BTC-USD", direction="LONG"), _mk_pos())
    # Second trade: LONG ETH — same correlation group, allowed (limit=2).
    t2 = engine.execute_trade(_mk_signal(symbol="ETH-USD", direction="LONG"), _mk_pos())
    assert t2 is not None
    # Third trade: LONG SOL — same group, NOW blocked (3rd same-direction).
    t3 = engine.execute_trade(_mk_signal(symbol="SOL-USD", direction="LONG"), _mk_pos())
    assert t3 is None


def test_execute_trade_clamps_oversize_to_max(tmp_db_path):
    """Position sizes above MAX_POSITION_SIZE are clamped down rather than rejected."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    t = engine.execute_trade(_mk_signal(), _mk_pos(size=15000, qty=150))
    assert t is not None
    open_pos = engine.get_open_positions("BTC-USD")[0]
    assert open_pos["position_size"] == pytest.approx(PaperTradingEngine.MAX_POSITION_SIZE)


# ─────────────────────── Close trade ───────────────────────


def test_close_trade_winning_long(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    result = engine.close_trade(tid, exit_price=110)
    assert result is not None
    # LONG: (110 - 100) * 5 = 50
    assert result["pnl"] == pytest.approx(50)
    assert result["status"] == "CLOSED"


def test_close_trade_losing_long(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    result = engine.close_trade(tid, exit_price=95, reason="stop_loss")
    # (95 - 100) * 5 = -25
    assert result["pnl"] == pytest.approx(-25)
    assert result["status"] == "STOPPED"


def test_close_trade_winning_short(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(direction="SHORT", entry=100, stop=105, tp=90)
    tid = engine.execute_trade(sig, _mk_pos(size=500, qty=5, stop=105, tp=90))
    result = engine.close_trade(tid, exit_price=90)
    # SHORT: (100 - 90) * 5 = 50
    assert result["pnl"] == pytest.approx(50)


def test_close_trade_updates_master_balance(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    initial = engine.get_master_portfolio()["current_balance"]
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=110)
    after = engine.get_master_portfolio()["current_balance"]
    # initial - 500 (entry) + 500 (refund) + 50 (pnl) = initial + 50
    assert after == pytest.approx(initial + 50)


def test_close_trade_updates_win_count(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=110)
    # Both master and per-symbol rows track wins/losses.
    for p in (engine.get_portfolio("BTC-USD"), engine.get_master_portfolio()):
        assert p["winning_trades"] == 1
        assert p["losing_trades"] == 0


def test_close_trade_updates_loss_count(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=95, reason="stop_loss")
    for p in (engine.get_portfolio("BTC-USD"), engine.get_master_portfolio()):
        assert p["winning_trades"] == 0
        assert p["losing_trades"] == 1
        assert p["consecutive_losses"] == 1


def test_close_trade_resets_consecutive_losses_on_win(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.COOLDOWN_MINUTES = 0  # test rapid same-symbol streak
    # Lose, lose, win
    t1 = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(t1, exit_price=95, reason="stop_loss")
    t2 = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(t2, exit_price=95, reason="stop_loss")
    assert engine.get_master_portfolio()["consecutive_losses"] == 2

    t3 = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(t3, exit_price=110)
    assert engine.get_master_portfolio()["consecutive_losses"] == 0


def test_close_nonexistent_trade_returns_none(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    assert engine.close_trade(trade_id=99999, exit_price=100) is None


def test_close_already_closed_trade_returns_none(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(), _mk_pos())
    engine.close_trade(tid, exit_price=110)
    # Second close → None
    assert engine.close_trade(tid, exit_price=105) is None


# ─────────────────────── Check stops ───────────────────────


def test_check_stops_triggers_stop_loss_long(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5, stop=95, tp=110))
    closed = engine.check_stops({"BTC-USD": 94.0})
    assert len(closed) == 1
    assert closed[0]["status"] == "STOPPED"


def test_check_stops_triggers_take_profit_long(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5, stop=95, tp=110))
    closed = engine.check_stops({"BTC-USD": 111.0})
    assert len(closed) == 1
    assert closed[0]["status"] == "CLOSED"


def test_check_stops_triggers_stop_loss_short(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(direction="SHORT", entry=100, stop=105, tp=90)
    engine.execute_trade(sig, _mk_pos(size=500, qty=5, stop=105, tp=90))
    closed = engine.check_stops({"BTC-USD": 106.0})
    assert len(closed) == 1
    assert closed[0]["status"] == "STOPPED"


def test_check_stops_ignores_price_in_range(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5, stop=95, tp=110))
    closed = engine.check_stops({"BTC-USD": 102.0})
    assert closed == []


def test_check_stops_handles_missing_price(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_mk_signal(), _mk_pos())
    # No price for BTC → no crash
    closed = engine.check_stops({"ETH-USD": 3000.0})
    assert closed == []


# ─────────────────────── Snapshots ───────────────────────


def test_take_snapshot_creates_row(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.take_snapshot("BTC-USD")
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM portfolio_snapshots WHERE symbol = ?", ("BTC-USD",)).fetchall()
    assert len(rows) == 1


def test_take_snapshot_unknown_symbol_noop(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.take_snapshot("UNKNOWN")  # should not crash
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM portfolio_snapshots WHERE symbol = ?", ("UNKNOWN",)).fetchall()
    assert len(rows) == 0


# ─────────────────────── Daily P&L reset ───────────────────────


def test_reset_daily_pnl(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=95, reason="stop_loss")
    assert engine.get_master_portfolio()["daily_pnl"] < 0

    engine.reset_daily_pnl()
    assert engine.get_master_portfolio()["daily_pnl"] == 0.0


# ─────────────────────── Trade history ───────────────────────


def test_trade_history_empty(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    assert engine.get_trade_history() == []


def test_trade_history_after_close(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(), _mk_pos())
    engine.close_trade(tid, exit_price=110)
    history = engine.get_trade_history()
    assert len(history) == 1


def test_trade_history_filter_by_symbol(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    t = engine.execute_trade(_mk_signal(symbol="BTC-USD"), _mk_pos())
    engine.close_trade(t, exit_price=110)
    h = engine.get_trade_history(symbol="ETH-USD")
    assert h == []
    h = engine.get_trade_history(symbol="BTC-USD")
    assert len(h) == 1


# ─────────────────────── API cost logging ───────────────────────


def test_log_api_cost(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.log_api_cost(
        symbol="BTC-USD",
        timeframe="4h",
        model="test-model",
        prompt_tokens=1000,
        completion_tokens=500,
        operation="test",
    )
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM api_costs").fetchall()
    assert len(rows) == 1
    assert rows[0]["total_tokens"] == 1500
    assert rows[0]["estimated_cost_usd"] > 0


# ─────────────────────── Mark-to-market (Bug 1) ───────────────────────


def test_mark_to_market_updates_unrealized_pnl_long(tmp_db_path):
    """Open LONG at 100, mark at 105 → pnl = (105-100)*qty on the open row."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.COOLDOWN_MINUTES = 0
    tid = engine.execute_trade(
        _mk_signal(entry=100, stop=95, tp=120),
        _mk_pos(size=500, qty=5, stop=95, tp=120),
    )
    summary = engine.mark_to_market({"BTC-USD": 105.0})
    assert summary["positions_marked"] == 1
    assert summary["total_unrealized_pnl"] == pytest.approx(25.0)

    open_trade = engine.get_open_positions("BTC-USD")[0]
    assert open_trade["status"] == "OPEN"  # still open
    assert open_trade["pnl"] == pytest.approx(25.0)
    assert open_trade["pnl_pct"] == pytest.approx(25.0 / 500)


def test_mark_to_market_updates_unrealized_pnl_short(tmp_db_path):
    """SHORT at 100, mark at 95 → pnl = (100-95)*qty on the open row."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.COOLDOWN_MINUTES = 0
    tid = engine.execute_trade(
        _mk_signal(direction="SHORT", entry=100, stop=105, tp=80),
        _mk_pos(size=500, qty=5, stop=105, tp=80),
    )
    summary = engine.mark_to_market({"BTC-USD": 95.0})
    assert summary["total_unrealized_pnl"] == pytest.approx(25.0)

    open_trade = engine.get_open_positions("BTC-USD")[0]
    assert open_trade["pnl"] == pytest.approx(25.0)
    assert open_trade["status"] == "OPEN"


def test_mark_to_market_does_not_close_trade(tmp_db_path):
    """mark_to_market must never mutate status or close positions."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(), _mk_pos())
    engine.mark_to_market({"BTC-USD": 50.0})  # huge paper loss, but open
    open_pos = engine.get_open_positions("BTC-USD")
    assert len(open_pos) == 1
    assert open_pos[0]["status"] == "OPEN"
    # master balance is unchanged — no realised P&L flowed through.
    master = engine.get_master_portfolio()
    assert master["total_pnl"] == 0.0


def test_mark_to_market_skips_missing_prices(tmp_db_path):
    """Positions whose symbol isn't in the price map are left untouched."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.COOLDOWN_MINUTES = 0
    tid_btc = engine.execute_trade(_mk_signal(symbol="BTC-USD"), _mk_pos())
    tid_eth = engine.execute_trade(_mk_signal(symbol="ETH-USD"), _mk_pos())
    summary = engine.mark_to_market({"BTC-USD": 105.0})  # no ETH price
    assert summary["positions_marked"] == 1
    assert summary["positions_skipped"] == 1


def test_mark_to_market_skips_nonpositive_prices(tmp_db_path):
    """A zero/negative price must not overwrite pnl with a nonsense value."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_mk_signal(), _mk_pos())
    summary = engine.mark_to_market({"BTC-USD": 0.0})
    assert summary["positions_marked"] == 0
    assert summary["positions_skipped"] == 1


def test_mark_to_market_empty_when_no_open_positions(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    summary = engine.mark_to_market({"BTC-USD": 100.0})
    assert summary == {
        "total_unrealized_pnl": 0.0,
        "positions_marked": 0,
        "positions_skipped": 0,
    }


def test_mark_to_market_totals_across_multiple_positions(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_mk_signal(symbol="BTC-USD"), _mk_pos(size=500, qty=5))
    engine.execute_trade(_mk_signal(symbol="ETH-USD"), _mk_pos(size=500, qty=5))
    summary = engine.mark_to_market({"BTC-USD": 110.0, "ETH-USD": 90.0})
    # BTC: +10*5=50 win; ETH: -10*5=-50 loss → net 0
    assert summary["positions_marked"] == 2
    assert summary["total_unrealized_pnl"] == pytest.approx(0.0)


# ─────────────────────── Cooldown (Bug 2) ───────────────────────


def test_cooldown_blocks_reentry_after_stop_loss(tmp_db_path):
    """Stop-loss → re-entry on same symbol is blocked while cooldown is active."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    assert engine.COOLDOWN_MINUTES > 0  # default is active
    tid = engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=95, reason="stop_loss")
    second = engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5))
    assert second is None


def test_cooldown_blocks_reentry_after_winning_close(tmp_db_path):
    """Any recent trade on a symbol — wins included — blocks immediate re-entry.

    This matches the Bug 4 requirement of a robust 'any recent trade' check.
    """
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=110)
    assert engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5)) is None


def test_cooldown_does_not_block_other_symbols(tmp_db_path):
    """Cooldown is scoped to the symbol — other markets remain tradable."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(symbol="BTC-USD"), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=95, reason="stop_loss")
    eth = engine.execute_trade(_mk_signal(symbol="ETH-USD"), _mk_pos(size=500, qty=5))
    assert eth is not None


def test_cooldown_disabled_allows_immediate_reentry(tmp_db_path):
    """COOLDOWN_MINUTES = 0 reverts to the old just-block-while-open behavior."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.COOLDOWN_MINUTES = 0
    tid = engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=95, reason="stop_loss")
    # Immediately reopen succeeds.
    assert engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5)) is not None


def test_cooldown_expires_after_window(tmp_db_path):
    """Once the cooldown window has elapsed, re-entry is allowed again."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.COOLDOWN_MINUTES = 30
    tid = engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=95, reason="stop_loss")

    # Rewind the exit_time so it's outside the cooldown window.
    from datetime import datetime, timedelta
    from db_schema import get_connection
    old = (datetime.utcnow() - timedelta(minutes=45)).isoformat()
    with get_connection(tmp_db_path) as conn:
        conn.execute("UPDATE trades SET exit_time = ? WHERE id = ?", (old, tid))

    assert engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5)) is not None


# ─────────────────── Price validation (Bug 3) ──────────────────────


def test_execute_trade_rejects_zero_entry_price(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(entry=0.0, stop=0.0, tp=0.0)
    assert engine.execute_trade(sig, _mk_pos(size=500, qty=5, stop=0.95, tp=1.1)) is None


def test_execute_trade_rejects_zero_stop_loss(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(entry=100.0, stop=95.0, tp=110.0)
    pos = _mk_pos(size=500, qty=5, stop=0.0, tp=110.0)
    assert engine.execute_trade(sig, pos) is None


def test_execute_trade_rejects_zero_take_profit(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(entry=100.0, stop=95.0, tp=110.0)
    pos = _mk_pos(size=500, qty=5, stop=95.0, tp=0.0)
    assert engine.execute_trade(sig, pos) is None


def test_execute_trade_rejects_negative_prices(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(entry=-1.0, stop=95.0, tp=110.0)
    pos = _mk_pos(size=500, qty=5, stop=95.0, tp=110.0)
    assert engine.execute_trade(sig, pos) is None


def test_execute_trade_accepts_legitimate_sub_cent_prices(tmp_db_path):
    """Low-cap tokens with tiny-but-real prices (e.g. SHIB ~ $1e-5) must still trade."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(entry=1e-5, stop=0.9e-5, tp=1.2e-5)
    pos = _mk_pos(size=500, qty=5e7, stop=0.9e-5, tp=1.2e-5)
    assert engine.execute_trade(sig, pos) is not None


# ───────────────── Enhanced stacking guard (Bug 4) ─────────────────


def test_stacking_guard_blocks_while_open(tmp_db_path):
    """With cooldown off, an open trade still blocks a duplicate on same symbol."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.COOLDOWN_MINUTES = 0
    first = engine.execute_trade(_mk_signal(), _mk_pos())
    second = engine.execute_trade(_mk_signal(), _mk_pos())
    assert first is not None
    assert second is None


def test_stacking_guard_blocks_across_open_and_recent_closed(tmp_db_path):
    """Even if no OPEN trade remains, a recently-closed trade blocks re-entry
    within the cooldown window."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    # Use a short but nonzero cooldown so the test is deterministic.
    engine.COOLDOWN_MINUTES = 30
    tid = engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=110)  # CLOSED, not OPEN
    # No open positions but the closed trade is within cooldown → blocked.
    assert engine.get_open_positions("BTC-USD") == []
    assert engine.execute_trade(_mk_signal(), _mk_pos(size=500, qty=5)) is None
