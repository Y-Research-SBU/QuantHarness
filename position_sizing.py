"""
Position sizing using Half-Kelly criterion with risk management constraints.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    position_size_usd: float     # Dollar amount to risk
    quantity: float              # Number of units to buy/sell
    risk_per_trade_usd: float   # Actual $ risk
    risk_pct: float             # Risk as % of portfolio
    kelly_fraction: float       # Raw Kelly fraction
    half_kelly: float           # Half-Kelly fraction used
    stop_loss: float            # Stop-loss price
    take_profit: float          # Take-profit price
    reason: str                 # Explanation


def calculate_kelly_fraction(
    win_rate: float,
    avg_win_loss_ratio: float,
) -> float:
    """
    Calculate Kelly criterion fraction.
    
    Kelly = W - (1 - W) / R
    where W = win probability, R = avg win / avg loss ratio
    
    Args:
        win_rate: Historical win probability (0 to 1)
        avg_win_loss_ratio: Average win amount / average loss amount
    
    Returns:
        Kelly fraction (can be negative if edge is negative)
    """
    if avg_win_loss_ratio <= 0:
        return 0.0
    
    kelly = win_rate - (1.0 - win_rate) / avg_win_loss_ratio
    return kelly


def calculate_half_kelly(
    win_rate: float,
    avg_win_loss_ratio: float,
) -> float:
    """
    Calculate Half-Kelly fraction (more conservative).
    
    Returns:
        Half-Kelly fraction, floored at 0.
    """
    kelly = calculate_kelly_fraction(win_rate, avg_win_loss_ratio)
    return max(0.0, kelly / 2.0)


def calculate_position_size(
    portfolio_balance: float,
    entry_price: float,
    stop_loss_price: float,
    direction: str,
    win_rate: float = 0.5,
    avg_win_loss_ratio: float = 1.5,
    max_risk_pct: float = 0.02,
    risk_reward_ratio: float = 1.5,
    max_position_pct: float = 0.25,
    signal_strength: float = 1.0,
    min_position_size: float = 0.0,
    max_position_size: float = float("inf"),
) -> PositionSizeResult:
    """
    Calculate position size using Half-Kelly criterion with safety caps.
    
    Args:
        portfolio_balance: Current portfolio balance
        entry_price: Planned entry price
        stop_loss_price: Planned stop-loss price
        direction: "LONG" or "SHORT"
        win_rate: Historical win rate (0 to 1)
        avg_win_loss_ratio: Historical avg win / avg loss
        max_risk_pct: Maximum risk per trade as fraction of portfolio (default 2%)
        risk_reward_ratio: Target risk-reward ratio
        max_position_pct: Maximum position size as fraction of portfolio
    
    Returns:
        PositionSizeResult with all sizing details
    """
    # Validate inputs
    if portfolio_balance <= 0:
        return PositionSizeResult(
            position_size_usd=0, quantity=0, risk_per_trade_usd=0,
            risk_pct=0, kelly_fraction=0, half_kelly=0,
            stop_loss=stop_loss_price, take_profit=entry_price,
            reason="Portfolio balance is zero or negative"
        )
    
    if entry_price <= 0:
        return PositionSizeResult(
            position_size_usd=0, quantity=0, risk_per_trade_usd=0,
            risk_pct=0, kelly_fraction=0, half_kelly=0,
            stop_loss=stop_loss_price, take_profit=entry_price,
            reason="Invalid entry price"
        )
    
    # Calculate risk per unit
    if direction == "LONG":
        risk_per_unit = entry_price - stop_loss_price
    else:  # SHORT
        risk_per_unit = stop_loss_price - entry_price
    
    if risk_per_unit <= 0:
        return PositionSizeResult(
            position_size_usd=0, quantity=0, risk_per_trade_usd=0,
            risk_pct=0, kelly_fraction=0, half_kelly=0,
            stop_loss=stop_loss_price, take_profit=entry_price,
            reason="Stop loss is in wrong direction"
        )
    
    # Calculate Kelly fraction
    kelly = calculate_kelly_fraction(win_rate, avg_win_loss_ratio)
    half_k = calculate_half_kelly(win_rate, avg_win_loss_ratio)
    
    # Cap risk at max_risk_pct of portfolio
    max_risk_usd = portfolio_balance * max_risk_pct
    
    # Kelly-based risk: half_kelly * portfolio
    kelly_risk_usd = half_k * portfolio_balance
    
    # Use the smaller of Kelly and max risk
    risk_usd = min(kelly_risk_usd, max_risk_usd)
    
    # If Kelly is zero or negative, use a minimal position
    if risk_usd <= 0:
        risk_usd = portfolio_balance * 0.005  # 0.5% minimum if we decide to trade
    
    # Calculate quantity from risk
    quantity = risk_usd / risk_per_unit
    position_size_usd = quantity * entry_price
    
    # Scale position by signal strength (0.0 – 1.0 → 50% – 100%)
    strength_scale = 0.5 + 0.5 * max(0.0, min(1.0, signal_strength))
    position_size_usd *= strength_scale
    quantity *= strength_scale
    risk_usd *= strength_scale

    # Cap position size at max_position_pct of portfolio
    max_position_usd = portfolio_balance * max_position_pct
    if position_size_usd > max_position_usd:
        position_size_usd = max_position_usd
        quantity = position_size_usd / entry_price
        risk_usd = quantity * risk_per_unit

    # Apply explicit min/max position size bounds
    if max_position_size < float("inf") and position_size_usd > max_position_size:
        position_size_usd = max_position_size
        quantity = position_size_usd / entry_price
        risk_usd = quantity * risk_per_unit
    if position_size_usd < min_position_size:
        position_size_usd = min_position_size
        quantity = position_size_usd / entry_price
        risk_usd = quantity * risk_per_unit
    
    # Calculate take-profit
    if direction == "LONG":
        take_profit = entry_price + (risk_per_unit * risk_reward_ratio)
    else:
        take_profit = entry_price - (risk_per_unit * risk_reward_ratio)
    
    risk_pct = risk_usd / portfolio_balance if portfolio_balance > 0 else 0
    
    return PositionSizeResult(
        position_size_usd=round(position_size_usd, 2),
        quantity=round(quantity, 8),
        risk_per_trade_usd=round(risk_usd, 2),
        risk_pct=round(risk_pct, 6),
        kelly_fraction=round(kelly, 6),
        half_kelly=round(half_k, 6),
        # Round to 10 decimals to preserve precision for low-priced tokens
        # (e.g. BONK-USD ~$0.0000063, SHIB-USD ~$0.0000062). Rounding to 4 here
        # collapsed prices to 0.0 and produced 'stop_loss=0.0 invalid' rejections.
        stop_loss=round(stop_loss_price, 10),
        take_profit=round(take_profit, 10),
        reason="Position sized with Half-Kelly criterion"
    )


def calculate_stop_loss(
    entry_price: float,
    direction: str,
    atr: Optional[float] = None,
    atr_multiplier: float = 2.0,
    default_pct: float = 0.03,
) -> float:
    """
    Calculate stop-loss price.
    
    Uses ATR if available, otherwise falls back to percentage-based stop.
    
    Args:
        entry_price: Entry price
        direction: "LONG" or "SHORT"
        atr: Average True Range value (if available)
        atr_multiplier: Multiplier for ATR-based stops
        default_pct: Default stop-loss percentage if no ATR
    
    Returns:
        Stop-loss price
    """
    if atr and atr > 0:
        stop_distance = atr * atr_multiplier
    else:
        stop_distance = entry_price * default_pct
    
    if direction == "LONG":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance
