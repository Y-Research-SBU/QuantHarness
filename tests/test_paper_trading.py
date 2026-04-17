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
    pfs = engine.get_all_portfolios()
    assert len(pfs) == len(MARKETS)


def test_portfolios_have_initial_balance(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    for p in engine.get_all_portfolios():
        assert p["initial_balance"] == 10000.0
        assert p["current_balance"] == 10000.0


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


def test_execute_trade_deducts_balance(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    before = engine.get_portfolio("BTC-USD")["current_balance"]
    engine.execute_trade(_mk_signal(), _mk_pos(size=500))
    after = engine.get_portfolio("BTC-USD")["current_balance"]
    assert after == pytest.approx(before - 500)


def test_execute_trade_increments_total_trades(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.execute_trade(_mk_signal(), _mk_pos())
    p = engine.get_portfolio("BTC-USD")
    assert p["total_trades"] == 1


def test_execute_trade_unknown_symbol(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    sig = _mk_signal(symbol="UNKNOWN-XYZ")
    trade_id = engine.execute_trade(sig, _mk_pos())
    assert trade_id is None


def test_execute_trade_blocked_by_correlation(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    # First trade: LONG BTC
    engine.execute_trade(_mk_signal(symbol="BTC-USD", direction="LONG"), _mk_pos())
    # Second trade: LONG ETH — same correlation group, should be blocked.
    t2 = engine.execute_trade(_mk_signal(symbol="ETH-USD", direction="LONG"), _mk_pos())
    assert t2 is None


def test_execute_trade_insufficient_balance(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    # Huge position > $10k balance
    t = engine.execute_trade(_mk_signal(), _mk_pos(size=15000))
    assert t is None


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


def test_close_trade_updates_portfolio_balance(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    initial = engine.get_portfolio("BTC-USD")["current_balance"]
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=110)
    after = engine.get_portfolio("BTC-USD")["current_balance"]
    # initial - 500 (entry) + 500 (refund) + 50 (pnl) = initial + 50
    assert after == pytest.approx(initial + 50)


def test_close_trade_updates_win_count(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=110)
    p = engine.get_portfolio("BTC-USD")
    assert p["winning_trades"] == 1
    assert p["losing_trades"] == 0


def test_close_trade_updates_loss_count(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    tid = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(tid, exit_price=95, reason="stop_loss")
    p = engine.get_portfolio("BTC-USD")
    assert p["winning_trades"] == 0
    assert p["losing_trades"] == 1
    assert p["consecutive_losses"] == 1


def test_close_trade_resets_consecutive_losses_on_win(tmp_db_path):
    engine = PaperTradingEngine(db_path=tmp_db_path)
    # Lose, lose, win
    t1 = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(t1, exit_price=95, reason="stop_loss")
    t2 = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(t2, exit_price=95, reason="stop_loss")
    assert engine.get_portfolio("BTC-USD")["consecutive_losses"] == 2

    t3 = engine.execute_trade(_mk_signal(entry=100, stop=95, tp=110), _mk_pos(size=500, qty=5))
    engine.close_trade(t3, exit_price=110)
    assert engine.get_portfolio("BTC-USD")["consecutive_losses"] == 0


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
    assert engine.get_portfolio("BTC-USD")["daily_pnl"] < 0

    engine.reset_daily_pnl()
    assert engine.get_portfolio("BTC-USD")["daily_pnl"] == 0.0


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
