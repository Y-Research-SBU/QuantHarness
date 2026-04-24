"""Parametrized invariant tests — sweep across inputs to catch edge cases.

These complement test_regression.py by exploring the parameter space around
each invariant, rather than asserting on one specific instance.
"""

from __future__ import annotations

import math

import pytest

from db_schema import get_connection
from market_config import MARKETS, StrategyType, get_correlation_groups
from paper_trading import PaperTradingEngine
from position_sizing import (
    PositionSizeResult,
    calculate_half_kelly,
    calculate_kelly_fraction,
    calculate_position_size,
    calculate_stop_loss,
)
from risk_manager import RiskManager
from strategies import Signal


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _sig(symbol="BTC-USD", direction="LONG", entry=100.0, stop=95.0, tp=115.0):
    return Signal(
        direction=direction,
        strength=0.5,
        strategy=StrategyType.MOMENTUM,
        symbol=symbol,
        timeframe="1h",
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        risk_reward_ratio=1.5,
        reasoning="inv",
        metadata={},
    )


def _pos(size=500.0, qty=5.0, stop=95.0, tp=115.0):
    return PositionSizeResult(
        position_size_usd=size,
        quantity=qty,
        risk_per_trade_usd=size * 0.02,
        risk_pct=0.02,
        kelly_fraction=0.1,
        half_kelly=0.05,
        stop_loss=stop,
        take_profit=tp,
        reason="inv",
    )


# ══════════════════════════════════════════════════════════════════════
# Drawdown invariants — circuit breaker should only fire when the equity
# drop is REAL (realized) and >= 10%.
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("initial,current,expected_allowed", [
    (10000, 10000, True),   # 0% dd
    (10000, 9999, True),    # 0.01% dd
    (10000, 9500, True),    # 5% dd
    (10000, 9200, True),    # 8% dd (warning band)
    (10000, 9100, True),    # 9% dd
    (10000, 9001, True),    # just under 10%
    (10000, 9000, False),   # exactly 10% → blocked
    (10000, 8999, False),
    (10000, 7000, False),
    (10000, 0, False),
    (10000, -100, False),
])
def test_drawdown_threshold_boundary(initial, current, expected_allowed):
    rm = RiskManager()
    assert rm.check_drawdown(current, initial).allowed is expected_allowed


@pytest.mark.parametrize("initial", [0, -100, -1])
def test_drawdown_rejects_invalid_initial(initial):
    rm = RiskManager()
    r = rm.check_drawdown(current_balance=5000, initial_balance=initial)
    assert not r.allowed


@pytest.mark.parametrize("daily_pnl,initial,expected_allowed", [
    (0, 10000, True),
    (-100, 10000, True),
    (-299, 10000, True),    # below 3% limit
    (-300, 10000, False),   # exactly hits 3% → blocked
    (-500, 10000, False),
    (500, 10000, True),     # positive P&L → allowed
    (-100000, 10000, False),
])
def test_daily_loss_threshold_boundary(daily_pnl, initial, expected_allowed):
    rm = RiskManager()
    assert rm.check_daily_loss(daily_pnl, initial).allowed is expected_allowed


# ══════════════════════════════════════════════════════════════════════
# Correlation invariants
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("existing,direction,expected_allowed", [
    (0, "LONG", True),
    (1, "LONG", True),   # 1 existing → allowed (under limit of 2)
    (2, "LONG", False),  # 2 existing → blocked
    (3, "LONG", False),
    (0, "SHORT", True),
    (1, "SHORT", True),
    (2, "SHORT", False),
])
def test_correlation_threshold(existing, direction, expected_allowed):
    rm = RiskManager()
    # Build `existing` positions in crypto group that match direction.
    group_syms = get_correlation_groups()["crypto"]
    other_syms = [s for s in group_syms if s != "SOL-USD"]
    positions = [
        {"symbol": other_syms[i % len(other_syms)], "direction": direction}
        for i in range(existing)
    ]
    r = rm.check_correlation("SOL-USD", direction, positions)
    assert r.allowed is expected_allowed


def test_correlation_ignores_positions_from_other_groups():
    rm = RiskManager()
    result = rm.check_correlation(
        "BTC-USD",
        direction="LONG",
        open_positions=[
            {"symbol": "SPY", "direction": "LONG"},   # us_equity
            {"symbol": "GC=F", "direction": "LONG"},  # commodities
            {"symbol": "EURUSD=X", "direction": "LONG"},  # forex
        ],
    )
    assert result.allowed


def test_correlation_does_not_count_self():
    """The symbol being checked doesn't count against itself."""
    rm = RiskManager()
    result = rm.check_correlation(
        "BTC-USD",
        direction="LONG",
        open_positions=[{"symbol": "BTC-USD", "direction": "LONG"}],
    )
    # Should pass since BTC-USD is excluded from same-group counting.
    assert result.allowed


# ══════════════════════════════════════════════════════════════════════
# P&L sign invariants
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("direction,entry,exit_price,expected_sign", [
    ("LONG",  100.0, 110.0, +1),
    ("LONG",  100.0, 90.0,  -1),
    ("LONG",  100.0, 100.0, 0),
    ("SHORT", 100.0, 90.0,  +1),
    ("SHORT", 100.0, 110.0, -1),
    ("SHORT", 100.0, 100.0, 0),
])
def test_pnl_sign_per_direction(tmp_db_path, direction, entry, exit_price, expected_sign):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    stop = entry * 0.9 if direction == "LONG" else entry * 1.1
    tp = entry * 1.2 if direction == "LONG" else entry * 0.8
    tid = engine.execute_trade(
        _sig(direction=direction, entry=entry, stop=stop, tp=tp),
        _pos(size=500.0, qty=5.0, stop=stop, tp=tp),
    )
    result = engine.close_trade(tid, exit_price=exit_price)
    if expected_sign > 0:
        assert result["pnl"] > 0
    elif expected_sign < 0:
        assert result["pnl"] < 0
    else:
        assert result["pnl"] == 0


# ══════════════════════════════════════════════════════════════════════
# Breakeven invariant: $0 P&L never counted as a loss
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("entry_prices", [
    [100.0, 200.0, 50.0, 1.5, 99999.0],
    [10.0],
    [1.0, 2.0, 3.0],
])
def test_breakeven_never_counted_as_loss(tmp_db_path, entry_prices):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    for i, entry in enumerate(entry_prices):
        symbol = list(MARKETS.keys())[i % len(MARKETS)]
        stop = entry * 0.9
        tp = entry * 1.2
        tid = engine.execute_trade(
            _sig(symbol=symbol, entry=entry, stop=stop, tp=tp),
            _pos(size=200.0, qty=2.0, stop=stop, tp=tp),
        )
        if tid is not None:
            result = engine.close_trade(tid, exit_price=entry)
            assert result["pnl"] == 0.0

    # No portfolio should have any loss counted.
    for p in engine.get_all_portfolios():
        assert p["losing_trades"] == 0
        assert p["winning_trades"] == 0


# ══════════════════════════════════════════════════════════════════════
# Position-stacking invariant
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("attempts", [1, 2, 3, 5, 10])
def test_only_first_trade_succeeds_per_symbol(tmp_db_path, attempts):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    results = [
        engine.execute_trade(_sig(), _pos(size=200.0, qty=2.0))
        for _ in range(attempts)
    ]
    succeeded = [r for r in results if r is not None]
    assert len(succeeded) == 1
    assert len(engine.get_open_positions("BTC-USD")) == 1


# ══════════════════════════════════════════════════════════════════════
# Position sizing
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("balance", [100, 1000, 10000, 100000])
@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_position_size_never_exceeds_max_pct(balance, direction):
    entry = 100.0
    stop = 99.0 if direction == "LONG" else 101.0
    pos = calculate_position_size(
        portfolio_balance=balance,
        entry_price=entry,
        stop_loss_price=stop,
        direction=direction,
        win_rate=0.9,               # very high
        avg_win_loss_ratio=10.0,    # and high payoff → Kelly wants big size
        max_position_pct=0.25,
    )
    assert pos.position_size_usd <= balance * 0.25 + 1e-6


@pytest.mark.parametrize("risk_pct,max_pct", [
    (0.01, 0.25),
    (0.02, 0.25),
    (0.05, 0.5),
])
def test_position_risk_never_exceeds_max_risk(risk_pct, max_pct):
    balance = 10000.0
    pos = calculate_position_size(
        portfolio_balance=balance,
        entry_price=100.0,
        stop_loss_price=90.0,
        direction="LONG",
        win_rate=0.5,
        avg_win_loss_ratio=1.5,
        max_risk_pct=risk_pct,
        max_position_pct=max_pct,
    )
    # risk_per_trade_usd is capped either by Kelly or by max_risk_pct.
    assert pos.risk_per_trade_usd <= balance * risk_pct + 1e-6


@pytest.mark.parametrize("win_rate,ratio,expected_positive", [
    (0.6, 1.5, True),   # edge → kelly > 0
    (0.5, 1.0, False),  # break-even → kelly ≈ 0
    (0.3, 1.0, False),  # negative edge
])
def test_kelly_sign(win_rate, ratio, expected_positive):
    k = calculate_kelly_fraction(win_rate, ratio)
    if expected_positive:
        assert k > 0
    else:
        assert k <= 0


@pytest.mark.parametrize("win_rate,ratio", [
    (0.6, 1.5),
    (0.7, 2.0),
    (0.9, 3.0),
])
def test_half_kelly_is_half_of_kelly(win_rate, ratio):
    k = calculate_kelly_fraction(win_rate, ratio)
    h = calculate_half_kelly(win_rate, ratio)
    if k <= 0:
        assert h == 0
    else:
        assert h == pytest.approx(k / 2)


@pytest.mark.parametrize("direction,expected_below", [
    ("LONG", True),
    ("SHORT", False),
])
def test_stop_loss_direction(direction, expected_below):
    stop = calculate_stop_loss(entry_price=100.0, direction=direction, default_pct=0.03)
    if expected_below:
        assert stop < 100.0
    else:
        assert stop > 100.0


# ══════════════════════════════════════════════════════════════════════
# Portfolio peak_balance is a high-water mark
# ══════════════════════════════════════════════════════════════════════


def test_peak_balance_only_increases(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    prev_peak = engine.get_portfolio("BTC-USD")["peak_balance"]

    # Sequence of randomish P&L.
    exit_prices = [110.0, 90.0, 120.0, 85.0, 130.0, 70.0]
    for exit in exit_prices:
        tid = engine.execute_trade(_sig(), _pos(size=200.0, qty=2.0))
        engine.close_trade(tid, exit_price=exit)

        new_peak = engine.get_portfolio("BTC-USD")["peak_balance"]
        assert new_peak >= prev_peak - 1e-6
        prev_peak = new_peak


# ══════════════════════════════════════════════════════════════════════
# Circuit breaker toggles
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("dd_pct,expected_active", [
    (0.00, False),
    (0.05, False),
    (0.099, False),
    (0.10, True),
    (0.20, True),
    (0.50, True),
])
def test_is_circuit_breaker_active(dd_pct, expected_active):
    rm = RiskManager()
    initial = 10000.0
    current = initial * (1 - dd_pct)
    assert rm.is_circuit_breaker_active(current, initial) is expected_active


# ══════════════════════════════════════════════════════════════════════
# Equity invariant across many trades
# ══════════════════════════════════════════════════════════════════════


def test_equity_invariant_large_sequence(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    initial = engine.get_portfolio("BTC-USD")["initial_balance"]

    expected_pnl = 0.0
    for exit_price in [105.0, 95.0, 102.0, 98.0, 108.0, 93.0, 106.0]:
        tid = engine.execute_trade(_sig(), _pos(size=200.0, qty=2.0))
        result = engine.close_trade(tid, exit_price=exit_price)
        expected_pnl += result["pnl"]

    portfolio = engine.get_portfolio("BTC-USD")
    assert portfolio["current_balance"] == pytest.approx(initial + expected_pnl)
    assert portfolio["total_pnl"] == pytest.approx(expected_pnl)


# ══════════════════════════════════════════════════════════════════════
# Consecutive losses counter
# ══════════════════════════════════════════════════════════════════════


def test_consecutive_losses_increments_and_resets(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)

    # 3 losses in a row
    for _ in range(3):
        tid = engine.execute_trade(_sig(), _pos(size=200.0, qty=2.0))
        engine.close_trade(tid, exit_price=90.0)
    assert engine.get_portfolio("BTC-USD")["consecutive_losses"] == 3

    # Win → reset to 0
    tid = engine.execute_trade(_sig(), _pos(size=200.0, qty=2.0))
    engine.close_trade(tid, exit_price=110.0)
    assert engine.get_portfolio("BTC-USD")["consecutive_losses"] == 0


# ══════════════════════════════════════════════════════════════════════
# Get all markets exactly matches portfolios
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("symbol", list(MARKETS.keys()))
def test_portfolio_created_for_every_configured_market(tmp_db_path, symbol):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    assert engine.get_portfolio(symbol) is not None


# ══════════════════════════════════════════════════════════════════════
# Dashboard never crashes on odd DB states
# ══════════════════════════════════════════════════════════════════════


def test_dashboard_handles_portfolio_without_trades(tmp_db_path, tmp_path):
    """Seed an engine but take no snapshots / trades — dashboard still works."""
    PaperTradingEngine(db_path=tmp_db_path)

    import dashboard
    app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(tmp_path))
    client = app.test_client()
    for endpoint in ("/api/overview", "/api/markets", "/api/trades",
                     "/api/strategies", "/api/scanner", "/api/backtests"):
        r = client.get(endpoint)
        assert r.status_code == 200, f"{endpoint} failed: {r.data!r}"


def test_dashboard_handles_only_open_positions(tmp_db_path, tmp_path):
    """Never-closed open positions still produce valid dashboard data."""
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_sig(), _pos(size=200.0, qty=2.0))

    import dashboard
    app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(tmp_path))
    client = app.test_client()
    overview = client.get("/api/overview").get_json()
    # Equity = initial since nothing closed with P&L.
    assert overview["open_positions"] == 1
    # unrealized_pnl is numeric (could be 0 due to no price history yet).
    assert isinstance(overview["unrealized_pnl"], (int, float))


# ══════════════════════════════════════════════════════════════════════
# Symbol-level summaries
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("n_trades", [1, 2, 5, 10])
def test_total_trades_matches_count(tmp_db_path, n_trades):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    for i in range(n_trades):
        tid = engine.execute_trade(_sig(), _pos(size=100.0, qty=1.0))
        engine.close_trade(tid, exit_price=105.0)

    portfolio = engine.get_portfolio("BTC-USD")
    assert portfolio["total_trades"] == n_trades


# ══════════════════════════════════════════════════════════════════════
# Stop-loss/Take-profit direction
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("direction,entry,stop,tp,price,expect_close,expect_status", [
    # LONG hits SL when price ≤ stop
    ("LONG",  100.0, 95.0, 115.0, 95.0, True, "STOPPED"),
    ("LONG",  100.0, 95.0, 115.0, 94.0, True, "STOPPED"),
    ("LONG",  100.0, 95.0, 115.0, 95.5, False, None),
    # LONG hits TP when price ≥ tp
    ("LONG",  100.0, 95.0, 115.0, 115.0, True, "CLOSED"),
    ("LONG",  100.0, 95.0, 115.0, 120.0, True, "CLOSED"),
    # SHORT hits SL when price ≥ stop
    ("SHORT", 100.0, 105.0, 90.0, 105.0, True, "STOPPED"),
    ("SHORT", 100.0, 105.0, 90.0, 110.0, True, "STOPPED"),
    # SHORT hits TP when price ≤ tp
    ("SHORT", 100.0, 105.0, 90.0, 90.0, True, "CLOSED"),
    ("SHORT", 100.0, 105.0, 90.0, 85.0, True, "CLOSED"),
    # SHORT mid-range
    ("SHORT", 100.0, 105.0, 90.0, 100.0, False, None),
])
def test_check_stops_boundary(
    tmp_db_path, direction, entry, stop, tp, price, expect_close, expect_status
):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _sig(direction=direction, entry=entry, stop=stop, tp=tp)
    pos = _pos(size=200.0, qty=2.0, stop=stop, tp=tp)
    engine.execute_trade(sig, pos)

    closed = engine.check_stops({"BTC-USD": price})
    if expect_close:
        assert len(closed) == 1
        assert closed[0]["status"] == expect_status
    else:
        assert closed == []
