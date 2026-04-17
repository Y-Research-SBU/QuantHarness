"""Tests for risk_manager.py — drawdown, daily-loss, correlation, loss streaks."""

import pytest

from risk_manager import RiskCheckResult, RiskManager


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager()


# ─────────────────────── Drawdown ───────────────────────


def test_drawdown_ok_at_start(rm):
    r = rm.check_drawdown(current_balance=10000, initial_balance=10000)
    assert r.allowed
    assert r.risk_level == "low"


def test_drawdown_small_loss_allowed(rm):
    r = rm.check_drawdown(current_balance=9500, initial_balance=10000)
    assert r.allowed


def test_drawdown_warning_near_limit(rm):
    # 8.5% drawdown: between 80% and 100% of 10% limit → warning.
    r = rm.check_drawdown(current_balance=9150, initial_balance=10000)
    assert r.allowed
    assert r.risk_level == "high"


def test_drawdown_circuit_breaker_triggers_at_10_pct(rm):
    r = rm.check_drawdown(current_balance=9000, initial_balance=10000)
    assert not r.allowed
    assert r.risk_level == "blocked"
    assert "CIRCUIT BREAKER" in r.reason


def test_drawdown_circuit_breaker_beyond_limit(rm):
    r = rm.check_drawdown(current_balance=8000, initial_balance=10000)
    assert not r.allowed


def test_drawdown_zero_initial_blocks(rm):
    r = rm.check_drawdown(current_balance=0, initial_balance=0)
    assert not r.allowed


# ─────────────────────── Daily loss ───────────────────────


def test_daily_loss_ok(rm):
    r = rm.check_daily_loss(daily_pnl=-100, initial_balance=10000)
    assert r.allowed


def test_daily_loss_limit_hit(rm):
    # 3% limit on 10k = 300
    r = rm.check_daily_loss(daily_pnl=-300, initial_balance=10000)
    assert not r.allowed
    assert "Daily loss limit" in r.reason


def test_daily_loss_positive_ok(rm):
    r = rm.check_daily_loss(daily_pnl=500, initial_balance=10000)
    assert r.allowed


def test_daily_loss_zero_initial_blocks(rm):
    r = rm.check_daily_loss(daily_pnl=-100, initial_balance=0)
    assert not r.allowed


# ─────────────────────── Correlation ───────────────────────


def test_correlation_no_positions_allowed(rm):
    r = rm.check_correlation(symbol="BTC-USD", direction="LONG", open_positions=[])
    assert r.allowed


def test_correlation_blocks_same_group_same_direction(rm):
    # BTC and ETH are both in the "crypto" correlation_group
    r = rm.check_correlation(
        symbol="BTC-USD",
        direction="LONG",
        open_positions=[{"symbol": "ETH-USD", "direction": "LONG"}],
    )
    assert not r.allowed
    assert "Correlation block" in r.reason


def test_correlation_allows_same_group_opposite_direction(rm):
    # Long ETH + Short BTC: hedged, not stacked.
    r = rm.check_correlation(
        symbol="BTC-USD",
        direction="SHORT",
        open_positions=[{"symbol": "ETH-USD", "direction": "LONG"}],
    )
    assert r.allowed


def test_correlation_allows_different_groups(rm):
    # BTC (crypto) + SPY (us_equity) → different groups.
    r = rm.check_correlation(
        symbol="BTC-USD",
        direction="LONG",
        open_positions=[{"symbol": "SPY", "direction": "LONG"}],
    )
    assert r.allowed


def test_correlation_allows_same_symbol(rm):
    # Same symbol twice isn't a correlation block (it's position mgmt elsewhere).
    r = rm.check_correlation(
        symbol="BTC-USD",
        direction="LONG",
        open_positions=[{"symbol": "BTC-USD", "direction": "LONG"}],
    )
    assert r.allowed


def test_correlation_unknown_symbol_allowed(rm):
    r = rm.check_correlation(
        symbol="UNKNOWN-USD",
        direction="LONG",
        open_positions=[{"symbol": "ETH-USD", "direction": "LONG"}],
    )
    assert r.allowed


# ─────────────────────── Position size multiplier ───────────────────────


def test_position_multiplier_normal(rm):
    assert rm.get_position_size_multiplier(0) == 1.0
    assert rm.get_position_size_multiplier(1) == 1.0
    assert rm.get_position_size_multiplier(2) == 1.0


def test_position_multiplier_reduced_after_threshold(rm):
    # After 3 consecutive losses, size is halved.
    assert rm.get_position_size_multiplier(3) == 0.5
    assert rm.get_position_size_multiplier(5) == 0.5
    assert rm.get_position_size_multiplier(100) == 0.5


# ─────────────────────── Drawdown helpers ───────────────────────


def test_calc_drawdown_pct(rm):
    assert rm.calculate_drawdown_pct(9000, 10000) == pytest.approx(0.1)


def test_calc_drawdown_pct_at_peak(rm):
    assert rm.calculate_drawdown_pct(10000, 10000) == 0.0


def test_calc_drawdown_pct_above_peak(rm):
    # Never negative
    assert rm.calculate_drawdown_pct(12000, 10000) == 0.0


def test_calc_drawdown_pct_zero_peak(rm):
    assert rm.calculate_drawdown_pct(100, 0) == 0.0


def test_circuit_breaker_active_below_threshold(rm):
    assert rm.is_circuit_breaker_active(9000, 10000) is True


def test_circuit_breaker_inactive_above_threshold(rm):
    assert rm.is_circuit_breaker_active(9500, 10000) is False


# ─────────────────────── check_trade_allowed (composite) ───────────────────────


def test_check_trade_happy_path(rm):
    r = rm.check_trade_allowed(
        symbol="BTC-USD",
        direction="LONG",
        portfolio_balance=10000,
        initial_balance=10000,
        daily_pnl=0,
        consecutive_losses=0,
        open_positions=[],
    )
    assert r.allowed


def test_check_trade_blocked_by_drawdown(rm):
    r = rm.check_trade_allowed(
        symbol="BTC-USD",
        direction="LONG",
        portfolio_balance=8000,  # 20% drawdown
        initial_balance=10000,
        daily_pnl=0,
        consecutive_losses=0,
        open_positions=[],
    )
    assert not r.allowed


def test_check_trade_blocked_by_daily_loss(rm):
    r = rm.check_trade_allowed(
        symbol="BTC-USD",
        direction="LONG",
        portfolio_balance=9700,
        initial_balance=10000,
        daily_pnl=-300,
        consecutive_losses=0,
        open_positions=[],
    )
    assert not r.allowed


def test_check_trade_blocked_by_correlation(rm):
    r = rm.check_trade_allowed(
        symbol="BTC-USD",
        direction="LONG",
        portfolio_balance=10000,
        initial_balance=10000,
        daily_pnl=0,
        consecutive_losses=0,
        open_positions=[{"symbol": "ETH-USD", "direction": "LONG"}],
    )
    assert not r.allowed


def test_check_trade_high_risk_on_loss_streak(rm):
    # Still allowed, but marked high-risk
    r = rm.check_trade_allowed(
        symbol="BTC-USD",
        direction="LONG",
        portfolio_balance=10000,
        initial_balance=10000,
        daily_pnl=0,
        consecutive_losses=5,
        open_positions=[],
    )
    assert r.allowed
    assert r.risk_level == "high"


# ─────────────────────── Constants ───────────────────────


def test_risk_constants_reasonable(rm):
    assert rm.MAX_RISK_PER_TRADE == 0.02
    assert rm.MAX_DRAWDOWN_PCT == 0.10
    assert rm.DAILY_LOSS_LIMIT == 0.03
    assert 0 < rm.POSITION_REDUCTION_FACTOR < 1
