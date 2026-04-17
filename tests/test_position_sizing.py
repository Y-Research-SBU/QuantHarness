"""Tests for position_sizing.py — Kelly criterion and risk-based sizing."""

import math

import pytest

from position_sizing import (
    PositionSizeResult,
    calculate_half_kelly,
    calculate_kelly_fraction,
    calculate_position_size,
    calculate_stop_loss,
)


# ─────────────────────── Kelly fraction ───────────────────────


def test_kelly_positive_edge():
    # 60% win rate with 2:1 RR → Kelly = 0.6 - 0.4/2 = 0.4
    assert calculate_kelly_fraction(0.6, 2.0) == pytest.approx(0.4)


def test_kelly_break_even():
    # 50% win rate with 1:1 RR → Kelly = 0
    assert calculate_kelly_fraction(0.5, 1.0) == pytest.approx(0.0)


def test_kelly_negative_edge():
    # 40% win rate with 1:1 RR → Kelly = 0.4 - 0.6 = -0.2
    assert calculate_kelly_fraction(0.4, 1.0) == pytest.approx(-0.2)


def test_kelly_zero_ratio_returns_zero():
    assert calculate_kelly_fraction(0.6, 0.0) == 0.0


def test_kelly_negative_ratio_returns_zero():
    assert calculate_kelly_fraction(0.6, -1.0) == 0.0


def test_kelly_perfect_win_rate():
    # 100% win rate → Kelly approaches 1.0
    assert calculate_kelly_fraction(1.0, 1.0) == pytest.approx(1.0)


def test_kelly_zero_win_rate():
    assert calculate_kelly_fraction(0.0, 1.0) == pytest.approx(-1.0)


# ─────────────────────── Half-Kelly ───────────────────────


def test_half_kelly_is_half_of_kelly():
    k = calculate_kelly_fraction(0.6, 2.0)
    hk = calculate_half_kelly(0.6, 2.0)
    assert hk == pytest.approx(k / 2.0)


def test_half_kelly_floored_at_zero():
    # Negative edge → half-Kelly clamps to 0
    assert calculate_half_kelly(0.4, 1.0) == 0.0


def test_half_kelly_zero_when_ratio_is_zero():
    assert calculate_half_kelly(0.6, 0.0) == 0.0


# ─────────────────────── Position sizing ───────────────────────


def test_position_size_basic_long():
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=95,
        direction="LONG",
        win_rate=0.5,
        avg_win_loss_ratio=1.5,
        max_risk_pct=0.02,
    )
    assert isinstance(r, PositionSizeResult)
    # Risk must be <= 2% of portfolio
    assert r.risk_per_trade_usd <= 200
    assert r.quantity > 0
    assert r.position_size_usd > 0


def test_position_size_basic_short():
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=105,
        direction="SHORT",
    )
    assert r.quantity > 0
    assert r.risk_per_trade_usd <= 200


def test_position_size_respects_max_risk_pct():
    # High Kelly, but max_risk_pct caps the risk.
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=90,
        direction="LONG",
        win_rate=0.9,
        avg_win_loss_ratio=5.0,
        max_risk_pct=0.01,  # cap at 1%
    )
    assert r.risk_per_trade_usd <= 100.01  # 1% of 10k, small tolerance


def test_position_size_respects_max_position_pct():
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=99.5,  # tiny stop → would want large quantity
        direction="LONG",
        win_rate=0.8,
        avg_win_loss_ratio=3.0,
        max_position_pct=0.25,
    )
    assert r.position_size_usd <= 2500.01  # 25% of 10k, small tolerance


def test_position_size_zero_balance_returns_zero():
    r = calculate_position_size(
        portfolio_balance=0,
        entry_price=100,
        stop_loss_price=95,
        direction="LONG",
    )
    assert r.position_size_usd == 0
    assert r.quantity == 0


def test_position_size_negative_balance_returns_zero():
    r = calculate_position_size(
        portfolio_balance=-100,
        entry_price=100,
        stop_loss_price=95,
        direction="LONG",
    )
    assert r.position_size_usd == 0


def test_position_size_zero_entry_price_returns_zero():
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=0,
        stop_loss_price=-5,
        direction="LONG",
    )
    assert r.quantity == 0


def test_position_size_wrong_direction_stop_returns_zero():
    # LONG but stop is above entry (wrong direction)
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=105,
        direction="LONG",
    )
    assert r.quantity == 0
    assert "wrong direction" in r.reason.lower()


def test_position_size_short_wrong_direction_stop():
    # SHORT but stop is below entry (wrong direction)
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=95,
        direction="SHORT",
    )
    assert r.quantity == 0


def test_position_size_take_profit_long():
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=95,
        direction="LONG",
        risk_reward_ratio=2.0,
    )
    # TP should be above entry by 2x the risk distance
    assert r.take_profit > 100
    assert r.take_profit == pytest.approx(110, rel=0.01)


def test_position_size_take_profit_short():
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=105,
        direction="SHORT",
        risk_reward_ratio=2.0,
    )
    # TP should be below entry by 2x the risk distance
    assert r.take_profit < 100
    assert r.take_profit == pytest.approx(90, rel=0.01)


def test_position_size_negative_kelly_uses_minimum():
    # Negative expected-value edge → Kelly is 0, but function still sizes a minimal trade.
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=95,
        direction="LONG",
        win_rate=0.3,
        avg_win_loss_ratio=1.0,
    )
    # Falls back to 0.5% of portfolio = $50 risk
    assert r.risk_per_trade_usd > 0
    assert r.risk_per_trade_usd <= 50.01


def test_position_size_risk_pct_calculation():
    r = calculate_position_size(
        portfolio_balance=10000,
        entry_price=100,
        stop_loss_price=95,
        direction="LONG",
        max_risk_pct=0.02,
    )
    # risk_pct should equal risk_per_trade_usd / portfolio_balance
    assert r.risk_pct == pytest.approx(r.risk_per_trade_usd / 10000, abs=1e-4)


# ─────────────────────── Stop loss ───────────────────────


def test_stop_loss_long_pct():
    stop = calculate_stop_loss(entry_price=100, direction="LONG", default_pct=0.03)
    assert stop == pytest.approx(97.0)


def test_stop_loss_short_pct():
    stop = calculate_stop_loss(entry_price=100, direction="SHORT", default_pct=0.03)
    assert stop == pytest.approx(103.0)


def test_stop_loss_uses_atr_when_provided():
    stop = calculate_stop_loss(entry_price=100, direction="LONG", atr=2.0, atr_multiplier=2.0)
    # 100 - 4 = 96
    assert stop == pytest.approx(96.0)


def test_stop_loss_atr_short():
    stop = calculate_stop_loss(entry_price=100, direction="SHORT", atr=2.0, atr_multiplier=2.0)
    assert stop == pytest.approx(104.0)


def test_stop_loss_falls_back_when_atr_zero():
    stop = calculate_stop_loss(entry_price=100, direction="LONG", atr=0, default_pct=0.05)
    assert stop == pytest.approx(95.0)


def test_stop_loss_returns_numeric():
    stop = calculate_stop_loss(entry_price=100, direction="LONG")
    assert isinstance(stop, float)
    assert not math.isnan(stop)
