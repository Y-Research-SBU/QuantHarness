"""Tests for performance_tracker.py — metrics calculations."""

import math

import pytest

from performance_tracker import (
    PerformanceMetrics,
    calculate_max_drawdown,
    calculate_performance,
    calculate_sharpe_ratio,
    get_api_cost_summary,
    get_portfolio_snapshots,
    get_strategy_performance,
)


# ─────────────────────── calculate_performance ───────────────────────


def test_performance_empty_trades():
    m = calculate_performance([])
    assert m.total_trades == 0
    assert m.win_rate == 0.0


def test_performance_single_winning_trade():
    trades = [{"pnl": 100.0}]
    m = calculate_performance(trades)
    assert m.total_trades == 1
    assert m.winning_trades == 1
    assert m.losing_trades == 0
    assert m.total_pnl == 100
    assert m.win_rate == 1.0


def test_performance_single_losing_trade():
    m = calculate_performance([{"pnl": -50.0}])
    assert m.winning_trades == 0
    assert m.losing_trades == 1
    assert m.win_rate == 0.0


def test_performance_mixed():
    trades = [{"pnl": p} for p in [100, -50, 200, -30, 75]]
    m = calculate_performance(trades)
    assert m.total_trades == 5
    assert m.winning_trades == 3
    assert m.losing_trades == 2
    assert m.total_pnl == pytest.approx(295)
    assert m.win_rate == pytest.approx(0.6)


def test_performance_avg_win_and_loss():
    trades = [{"pnl": p} for p in [100, -50, 200]]
    m = calculate_performance(trades)
    assert m.avg_win == pytest.approx(150)
    assert m.avg_loss == pytest.approx(50)


def test_performance_largest_win_loss():
    trades = [{"pnl": p} for p in [100, -50, 500, -10]]
    m = calculate_performance(trades)
    assert m.largest_win == 500
    assert m.largest_loss == -50


def test_performance_profit_factor():
    # gross_profit = 300, gross_loss = 100 → PF = 3.0
    trades = [{"pnl": p} for p in [100, 200, -60, -40]]
    m = calculate_performance(trades)
    assert m.profit_factor == pytest.approx(3.0)


def test_performance_profit_factor_inf_on_no_losses():
    trades = [{"pnl": p} for p in [100, 200]]
    m = calculate_performance(trades)
    assert m.profit_factor == float("inf")


def test_performance_expectancy():
    # 60% win × 100 avg win - 40% × 50 avg loss = 60 - 20 = 40
    trades = [{"pnl": 100}] * 3 + [{"pnl": -50}] * 2
    m = calculate_performance(trades)
    assert m.expectancy == pytest.approx(40.0)


def test_performance_filters_none_pnls():
    trades = [{"pnl": 100}, {"pnl": None}, {"pnl": -50}]
    m = calculate_performance(trades)
    assert m.total_trades == 2


def test_performance_all_losses():
    trades = [{"pnl": -100}] * 5
    m = calculate_performance(trades)
    assert m.total_trades == 5
    assert m.winning_trades == 0
    assert m.total_pnl == -500


def test_performance_holding_period_hours():
    trades = [
        {"pnl": 100, "entry_time": "2024-01-01T00:00:00", "exit_time": "2024-01-01T04:00:00"},
    ]
    m = calculate_performance(trades)
    assert "hours" in m.avg_holding_period


def test_performance_holding_period_days():
    trades = [
        {"pnl": 100, "entry_time": "2024-01-01T00:00:00", "exit_time": "2024-01-03T00:00:00"},
    ]
    m = calculate_performance(trades)
    assert "days" in m.avg_holding_period


# ─────────────────────── Sharpe ratio ───────────────────────


def test_sharpe_single_trade_returns_zero():
    assert calculate_sharpe_ratio([100]) == 0.0


def test_sharpe_flat_pnl_returns_zero():
    # Zero variance → zero Sharpe
    assert calculate_sharpe_ratio([50, 50, 50, 50]) == 0.0


def test_sharpe_positive_returns_positive():
    # Uniform positive pnls mean std=0 → returns 0.
    # Use varying positive pnls to get meaningful Sharpe.
    pnls = [100, 80, 120, 110, 90]
    s = calculate_sharpe_ratio(pnls)
    assert s > 0


def test_sharpe_is_finite():
    pnls = [10, -5, 20, -3, 15]
    s = calculate_sharpe_ratio(pnls)
    assert math.isfinite(s)


def test_sharpe_empty_list():
    assert calculate_sharpe_ratio([]) == 0.0


# ─────────────────────── Max drawdown ───────────────────────


def test_max_drawdown_empty():
    dd, dd_pct = calculate_max_drawdown([])
    assert dd == 0.0
    assert dd_pct == 0.0


def test_max_drawdown_only_wins():
    dd, dd_pct = calculate_max_drawdown([100, 100, 100])
    # Always climbing → no drawdown
    assert dd == 0.0


def test_max_drawdown_single_loss():
    dd, dd_pct = calculate_max_drawdown([-500], initial_balance=10000)
    assert dd == pytest.approx(500)
    assert dd_pct == pytest.approx(0.05, abs=0.01)


def test_max_drawdown_peak_then_loss():
    # Peak at 11000, then drop to 10500 → dd = 500
    dd, dd_pct = calculate_max_drawdown([1000, -500], initial_balance=10000)
    assert dd == pytest.approx(500)


def test_max_drawdown_multiple_peaks():
    pnls = [500, -200, 300, -600, 100]
    dd, dd_pct = calculate_max_drawdown(pnls, initial_balance=10000)
    assert dd > 0
    assert dd_pct > 0


def test_max_drawdown_pct_bounded():
    pnls = [-100, -100, -100]
    dd, dd_pct = calculate_max_drawdown(pnls, initial_balance=10000)
    assert 0 <= dd_pct <= 1.0


# ─────────────────────── DB query wrappers ───────────────────────


def test_get_strategy_performance_empty(tmp_db_path):
    # Creating the db creates the tables empty.
    from db_schema import init_db
    init_db(tmp_db_path).close()
    result = get_strategy_performance(db_path=tmp_db_path)
    assert result == []


def test_get_portfolio_snapshots_empty(tmp_db_path):
    from db_schema import init_db
    init_db(tmp_db_path).close()
    result = get_portfolio_snapshots(db_path=tmp_db_path)
    assert result == []


def test_get_api_cost_summary_empty(tmp_db_path):
    from db_schema import init_db
    init_db(tmp_db_path).close()
    summary = get_api_cost_summary(db_path=tmp_db_path)
    assert "total_tokens" in summary
    # sqlite SUM on empty table returns None → treated as 0 by the None-coalescing in the view.
    assert summary["total_tokens"] in (0, None)
