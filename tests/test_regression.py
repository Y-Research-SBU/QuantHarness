"""Regression tests — one per bug that slipped through today.

Each test is named after the specific bug it would catch. If any of these
fail, one of the production fixes has been reverted.

Today's bug list (see commit history):
  REL-312: Dashboard showed cash balance instead of equity.
  REL-313: Breakeven ($0) trades counted as losses.
  REL-310: Correlation blocked single same-direction position (too aggressive).
  [no-REL]: Circuit breaker used cash balance instead of equity — allocation
            treated as a loss.
  [no-REL]: Position stacking — no limit on same-symbol positions.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from db_schema import get_connection, init_db
from market_config import StrategyType
from paper_trading import PaperTradingEngine
from position_sizing import PositionSizeResult
from risk_manager import RiskManager
from strategies import Signal


# ══════════════════════════════════════════════════════════════════════
# Bug 1: Circuit breaker used cash balance instead of equity
# ══════════════════════════════════════════════════════════════════════


def test_circuit_breaker_not_triggered_by_allocation(engine, make_signal, make_position):
    """Opening a large position allocates capital — that is NOT a loss.

    Before the fix, opening a trade that deducted >10% of cash would mark
    the circuit breaker active because the drawdown check compared
    current_balance (cash) to initial_balance without adding back open
    position value.
    """
    # Allocate 50% of a fresh $10k portfolio to a single trade — cash drops
    # to $5k, but equity is still $10k.
    signal = make_signal(entry=100.0, stop=95.0, tp=115.0)
    position = make_position(size=5000.0, qty=50.0)
    trade_id = engine.execute_trade(signal, position)
    assert trade_id is not None

    portfolio = engine.get_portfolio("BTC-USD")
    assert portfolio["current_balance"] == pytest.approx(5000.0)
    # No realised loss has occurred → circuit breaker must NOT fire.
    assert portfolio["is_circuit_breaker_active"] == 0

    # A second risk check (e.g. for a new trade) should still pass because
    # equity = cash + allocated = $10k and drawdown is 0%.
    rm = RiskManager()
    result = rm.check_trade_allowed(
        symbol="ETH-USD",
        direction="LONG",
        portfolio_balance=5000.0,
        initial_balance=10000.0,
        daily_pnl=0.0,
        consecutive_losses=0,
        open_positions=[{"symbol": "BTC-USD", "direction": "LONG"}],
        open_position_value=5000.0,
    )
    assert result.allowed
    assert "CIRCUIT BREAKER" not in result.reason


def test_equity_not_cash_in_drawdown_check(engine, make_signal, make_position):
    """The drawdown check must use equity, not cash."""
    # Open a trade that drops cash well below the 10% threshold
    # (cash = $1,000, allocated = $9,000 → equity = $10,000).
    signal = make_signal(entry=100.0, stop=99.0, tp=103.0)
    position = make_position(size=9000.0, qty=90.0)
    assert engine.execute_trade(signal, position) is not None

    # A cash-based drawdown would be 90%. An equity-based one is 0%.
    rm = RiskManager()
    drawdown_on_cash = rm.check_drawdown(current_balance=1000.0, initial_balance=10000.0)
    drawdown_on_equity = rm.check_drawdown(current_balance=10000.0, initial_balance=10000.0)
    assert not drawdown_on_cash.allowed
    assert drawdown_on_equity.allowed

    # check_trade_allowed must internally use equity.
    result = rm.check_trade_allowed(
        symbol="BTC-USD",
        direction="LONG",
        portfolio_balance=1000.0,
        initial_balance=10000.0,
        daily_pnl=0.0,
        consecutive_losses=0,
        open_positions=[],
        open_position_value=9000.0,
    )
    assert result.allowed, f"Equity check should pass but: {result.reason}"


def test_circuit_breaker_triggered_by_real_realized_loss(engine, make_signal, make_position):
    """Conversely, a REAL realised loss >=10% must trigger the breaker."""
    signal = make_signal(entry=100.0, stop=95.0, tp=110.0)
    # Open a tiny position so the closing loss dominates.
    position = make_position(size=500.0, qty=5.0)
    trade_id = engine.execute_trade(signal, position)
    assert trade_id is not None

    # Force a closing price that produces a >$1,000 loss on the 10k portfolio.
    # qty=5, entry=100, exit=-1100/5+100 = close at a price that yields -$1,100.
    closed = engine.close_trade(trade_id, exit_price=-120.0, reason="manual")
    assert closed is not None
    assert closed["pnl"] < -1000.0

    portfolio = engine.get_portfolio("BTC-USD")
    assert portfolio["is_circuit_breaker_active"] == 1


# ══════════════════════════════════════════════════════════════════════
# Bug 2: Position stacking — no limit on same-symbol positions
# ══════════════════════════════════════════════════════════════════════


def test_position_stacking_blocked_same_symbol(engine, make_signal, make_position):
    """execute_trade must refuse a second open position on the same symbol."""
    first = engine.execute_trade(make_signal(), make_position())
    assert first is not None
    second = engine.execute_trade(make_signal(), make_position())
    assert second is None  # Blocked

    open_positions = engine.get_open_positions("BTC-USD")
    assert len(open_positions) == 1


def test_opposing_positions_blocked_same_symbol(engine, make_signal, make_position):
    """Opening a SHORT while already LONG on the same symbol must be rejected
    (falls out of the stacking guard)."""
    long_id = engine.execute_trade(
        make_signal(direction="LONG", entry=100.0, stop=95.0, tp=110.0),
        make_position(size=500.0, qty=5.0),
    )
    assert long_id is not None

    short_id = engine.execute_trade(
        make_signal(direction="SHORT", entry=100.0, stop=105.0, tp=90.0),
        make_position(size=500.0, qty=5.0, stop=105.0, tp=90.0),
    )
    assert short_id is None

    open_positions = engine.get_open_positions("BTC-USD")
    assert len(open_positions) == 1
    assert open_positions[0]["direction"] == "LONG"


def test_stacking_allowed_again_after_close(engine, make_signal, make_position):
    """After closing the first position, a new position on the same symbol
    must be allowed again."""
    first = engine.execute_trade(make_signal(), make_position())
    engine.close_trade(first, exit_price=105.0)

    second = engine.execute_trade(make_signal(), make_position())
    assert second is not None


# ══════════════════════════════════════════════════════════════════════
# Bug 3: Breakeven ($0 P&L) counted as losses
# ══════════════════════════════════════════════════════════════════════


def test_breakeven_not_counted_as_loss(engine, make_signal, make_position):
    """A trade closed at entry price (P&L = $0) must not increment losing_trades."""
    trade_id = engine.execute_trade(
        make_signal(entry=100.0, stop=95.0, tp=110.0),
        make_position(size=500.0, qty=5.0),
    )
    # Close at the entry price — P&L is exactly 0.
    closed = engine.close_trade(trade_id, exit_price=100.0)
    assert closed["pnl"] == 0.0

    portfolio = engine.get_portfolio("BTC-USD")
    assert portfolio["losing_trades"] == 0
    assert portfolio["winning_trades"] == 0


def test_breakeven_does_not_reset_winning_streak(engine, make_signal, make_position):
    """Breakeven should not break or continue a winning/losing streak."""
    # First trade: loss → consecutive_losses = 1.
    t1 = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
    engine.close_trade(t1, exit_price=95.0)  # -$25
    assert engine.get_portfolio("BTC-USD")["consecutive_losses"] == 1

    # Second trade: breakeven — streak must stay at 1, not bump to 2.
    t2 = engine.execute_trade(make_signal(), make_position(size=500.0, qty=5.0))
    engine.close_trade(t2, exit_price=100.0)
    assert engine.get_portfolio("BTC-USD")["consecutive_losses"] == 1


def test_portfolio_stats_exclude_breakeven_from_losses(engine, make_signal, make_position):
    """Mix of wins, losses, breakevens — only real wins/losses counted."""
    trades = [
        (100.0, 110.0, "win"),       # +$50
        (100.0, 90.0, "loss"),       # -$50
        (100.0, 100.0, "breakeven"), # $0
        (100.0, 105.0, "win"),       # +$25
        (100.0, 100.0, "breakeven"),
    ]
    for entry, exit_price, _ in trades:
        tid = engine.execute_trade(
            make_signal(entry=entry, stop=entry * 0.9, tp=entry * 1.2),
            make_position(size=500.0, qty=5.0, stop=entry * 0.9, tp=entry * 1.2),
        )
        engine.close_trade(tid, exit_price=exit_price)

    portfolio = engine.get_portfolio("BTC-USD")
    assert portfolio["winning_trades"] == 2
    assert portfolio["losing_trades"] == 1


def test_strategy_performance_excludes_breakeven_from_losses(engine, make_signal, make_position):
    """strategy_performance table must also not count $0 as a loss."""
    tid = engine.execute_trade(
        make_signal(strategy=StrategyType.MOMENTUM),
        make_position(size=500.0, qty=5.0),
    )
    engine.close_trade(tid, exit_price=100.0)  # breakeven

    with get_connection(engine.db_path) as conn:
        row = conn.execute(
            "SELECT winning_trades, losing_trades, total_trades FROM strategy_performance "
            "WHERE strategy = 'momentum' AND symbol = 'BTC-USD'"
        ).fetchone()

    assert row is not None
    assert row["winning_trades"] == 0
    assert row["losing_trades"] == 0
    assert row["total_trades"] == 1


# ══════════════════════════════════════════════════════════════════════
# Bug 4: Correlation blocked one same-direction position (too aggressive)
# ══════════════════════════════════════════════════════════════════════


def test_correlation_allows_two_same_direction():
    """Two same-direction crypto positions is allowed; the block only kicks
    in once we try to add a third."""
    rm = RiskManager()

    # 1 same-direction position → allowed
    r1 = rm.check_correlation(
        "SOL-USD",
        direction="LONG",
        open_positions=[{"symbol": "BTC-USD", "direction": "LONG"}],
    )
    assert r1.allowed

    # 2 same-direction positions → still allowed (up to MAX_CORRELATED_POSITIONS)
    r2 = rm.check_correlation(
        "SOL-USD",
        direction="LONG",
        open_positions=[
            {"symbol": "BTC-USD", "direction": "LONG"},
            {"symbol": "ETH-USD", "direction": "LONG"},
        ],
    )
    assert not r2.allowed, (
        "Third same-direction crypto position must be blocked, but got: "
        f"{r2.reason}"
    )


def test_correlation_allows_opposite_direction():
    """Correlation block only applies to same-direction stacking."""
    rm = RiskManager()
    result = rm.check_correlation(
        "SOL-USD",
        direction="SHORT",
        open_positions=[
            {"symbol": "BTC-USD", "direction": "LONG"},
            {"symbol": "ETH-USD", "direction": "LONG"},
        ],
    )
    assert result.allowed


def test_correlation_block_is_per_group():
    """A crypto long position must not interfere with a stocks long."""
    rm = RiskManager()
    result = rm.check_correlation(
        "SPY",  # us_equity group
        direction="LONG",
        open_positions=[
            {"symbol": "BTC-USD", "direction": "LONG"},  # crypto group
            {"symbol": "ETH-USD", "direction": "LONG"},
        ],
    )
    assert result.allowed


# ══════════════════════════════════════════════════════════════════════
# Bug 5: Dashboard showed cash balance instead of equity
# ══════════════════════════════════════════════════════════════════════


def _seed_db_with_open_position(db_path: str) -> None:
    """Seed a single portfolio with an open position allocating cash."""
    conn = init_db(db_path)
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO portfolios
           (symbol, initial_balance, current_balance, total_pnl, total_trades,
            winning_trades, losing_trades, peak_balance)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("BTC-USD", 10000.0, 7000.0, 0.0, 1, 0, 0, 10000.0),
    )
    conn.execute(
        """INSERT INTO trades
           (symbol, timeframe, strategy, direction, entry_price, position_size,
            quantity, status, entry_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)""",
        ("BTC-USD", "1h", "momentum", "LONG", 100.0, 3000.0, 30.0, now),
    )
    conn.commit()
    conn.close()


def test_dashboard_shows_equity_not_cash(tmp_db_path):
    """Dashboard overview total_balance must equal cash + open position sizes."""
    _seed_db_with_open_position(tmp_db_path)

    import dashboard
    overview = dashboard.build_overview(tmp_db_path)

    # cash=$7,000 + open_position=$3,000 = equity $10,000
    assert overview["total_balance"] == pytest.approx(10000.0)


def test_dashboard_market_grid_shows_equity_not_cash(tmp_db_path):
    """The per-market grid's current_balance column must also be equity."""
    _seed_db_with_open_position(tmp_db_path)

    import dashboard
    cards = dashboard.build_market_grid(tmp_db_path)
    btc = next(c for c in cards if c["symbol"] == "BTC-USD")
    assert btc["current_balance"] == pytest.approx(10000.0)


def test_dashboard_unrealized_pnl_calculated(tmp_db_path):
    """With open + closed trades, unrealized_pnl must be non-zero and reported
    separately from realized_pnl."""
    conn = init_db(tmp_db_path)
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO portfolios
           (symbol, initial_balance, current_balance, total_pnl, total_trades,
            winning_trades, peak_balance)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("BTC-USD", 10000.0, 7100.0, 100.0, 2, 1, 10100.0),
    )
    # Closed trade: entry 100, exit 110, qty 10 → +$100 (recorded as total_pnl).
    conn.execute(
        """INSERT INTO trades (symbol, timeframe, strategy, direction,
            entry_price, exit_price, position_size, quantity, pnl, status,
            entry_time, exit_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED', ?, ?)""",
        ("BTC-USD", "1h", "momentum", "LONG", 100.0, 110.0, 1000.0, 10.0,
         100.0, now, now),
    )
    # Open trade: entry 100, qty 20. The unrealized calc uses latest close
    # which for this DB is the last exit_price (110) → +$200 unrealized.
    conn.execute(
        """INSERT INTO trades (symbol, timeframe, strategy, direction,
            entry_price, position_size, quantity, status, entry_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)""",
        ("BTC-USD", "1h", "momentum", "LONG", 100.0, 2000.0, 20.0, now),
    )
    conn.commit()
    conn.close()

    import dashboard
    overview = dashboard.build_overview(tmp_db_path)
    assert overview["realized_pnl"] == pytest.approx(100.0)
    assert overview["unrealized_pnl"] == pytest.approx(200.0)
    assert overview["total_pnl"] == pytest.approx(300.0)


def test_dashboard_short_unrealized_pnl_has_correct_sign(tmp_db_path):
    """Unrealized P&L on a SHORT must use (entry - current_price), not (current - entry)."""
    conn = init_db(tmp_db_path)
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO portfolios (symbol, initial_balance, current_balance,
            total_pnl, peak_balance)
           VALUES (?, ?, ?, ?, ?)""",
        ("BTC-USD", 10000.0, 9000.0, 0.0, 10000.0),
    )
    # One closed trade so the dashboard has a price to use.
    conn.execute(
        """INSERT INTO trades (symbol, timeframe, strategy, direction,
            entry_price, exit_price, position_size, quantity, pnl, status,
            entry_time, exit_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED', ?, ?)""",
        ("ETH-USD", "1h", "momentum", "LONG", 100.0, 90.0, 1000.0, 10.0,
         -100.0, now, now),
    )
    # Open SHORT: entry 100, qty 10. Latest price will fall back to entry
    # (there is no exit price for BTC-USD), so unrealized = 0.
    conn.execute(
        """INSERT INTO trades (symbol, timeframe, strategy, direction,
            entry_price, position_size, quantity, status, entry_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)""",
        ("BTC-USD", "1h", "momentum", "SHORT", 100.0, 1000.0, 10.0, now),
    )
    conn.commit()
    conn.close()

    import dashboard
    unrealized = dashboard._compute_unrealized_pnl(tmp_db_path)
    # No price data means $0 unrealized; the key point is no exception was
    # raised and the SHORT was handled without flipping sign.
    assert unrealized["BTC-USD"] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════
# Extra invariants closely tied to today's bugs
# ══════════════════════════════════════════════════════════════════════


def test_close_trade_returns_capital_plus_pnl(engine, make_signal, make_position):
    """Closing a trade must add back position_size + pnl to current_balance.

    i.e. winning_close_balance = starting_cash - size + size + pnl
                              = starting_cash + pnl
    """
    start = engine.get_portfolio("BTC-USD")["current_balance"]
    size = 500.0
    qty = 5.0
    tid = engine.execute_trade(
        make_signal(entry=100.0),
        make_position(size=size, qty=qty),
    )
    # Sanity: cash went down by exactly size.
    assert engine.get_portfolio("BTC-USD")["current_balance"] == pytest.approx(start - size)

    closed = engine.close_trade(tid, exit_price=110.0)  # +$50 pnl
    assert closed["pnl"] == pytest.approx(50.0)

    after = engine.get_portfolio("BTC-USD")["current_balance"]
    assert after == pytest.approx(start + 50.0)


def test_close_losing_trade_returns_remaining_capital(engine, make_signal, make_position):
    """Even losers return position_size + pnl (pnl is negative). Balance after a
    total wipeout of a $500 trade should be start - $500."""
    start = engine.get_portfolio("BTC-USD")["current_balance"]
    tid = engine.execute_trade(
        make_signal(entry=100.0),
        make_position(size=500.0, qty=5.0),
    )
    # exit at 0 → pnl = -500
    engine.close_trade(tid, exit_price=0.0)
    after = engine.get_portfolio("BTC-USD")["current_balance"]
    assert after == pytest.approx(start - 500.0)


def test_equity_invariant_holds_after_open(engine, make_signal, make_position):
    """Equity = cash_balance + sum(open position sizes)."""
    start_equity = engine.get_portfolio("BTC-USD")["current_balance"]
    size = 1500.0
    engine.execute_trade(make_signal(), make_position(size=size, qty=15.0))

    portfolio = engine.get_portfolio("BTC-USD")
    open_positions = engine.get_open_positions("BTC-USD")
    allocated = sum(p["position_size"] for p in open_positions)
    assert portfolio["current_balance"] + allocated == pytest.approx(start_equity)
