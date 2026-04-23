"""
Risk management system for paper trading.
Handles drawdown limits, correlation checks, daily loss limits, circuit breakers,
and consecutive loss tracking.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from market_config import MARKETS, get_correlation_groups

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    allowed: bool
    reason: str
    risk_level: str  # "low", "medium", "high", "blocked"


class RiskManager:
    """
    Centralized risk management for all markets.
    
    Rules:
    1. Max 2% portfolio risk per trade
    2. Max 10% portfolio drawdown → circuit breaker
    3. Correlation check: don't go long correlated assets simultaneously (>0.7)
    4. Daily loss limit: 3% → stop trading for the day
    5. Automatic position reduction after 3 consecutive losses
    """
    
    MAX_RISK_PER_TRADE = 0.02       # 2%
    MAX_DRAWDOWN_PCT = 0.10          # 10%
    CORRELATION_THRESHOLD = 0.7      # Don't stack >0.7 correlated positions
    DAILY_LOSS_LIMIT = 0.03          # 3%
    CONSECUTIVE_LOSS_THRESHOLD = 3   # Reduce size after 3 losses
    POSITION_REDUCTION_FACTOR = 0.5  # Cut size in half after losing streak
    
    def __init__(self, db_conn: Optional[sqlite3.Connection] = None):
        self.db_conn = db_conn
        self._correlation_cache: Dict[Tuple[str, str], float] = {}
    
    def check_trade_allowed(
        self,
        symbol: str,
        direction: str,
        portfolio_balance: float,
        initial_balance: float,
        daily_pnl: float,
        consecutive_losses: int,
        open_positions: Optional[List[Dict]] = None,
        open_position_value: float = 0.0,
    ) -> RiskCheckResult:
        """
        Run all risk checks for a proposed trade.
        
        Args:
            symbol: Market symbol
            direction: "LONG" or "SHORT"
            portfolio_balance: Current portfolio balance (cash, excluding open positions)
            initial_balance: Starting portfolio balance
            daily_pnl: P&L for today
            consecutive_losses: Number of consecutive losing trades
            open_positions: List of currently open positions [{symbol, direction, ...}]
            open_position_value: Total USD value of open positions for this symbol
                                (position_size allocated but not yet realized)
        
        Returns:
            RiskCheckResult indicating if trade is allowed
        """
        # Check 1: Circuit breaker (max drawdown)
        # Use equity = cash balance + allocated position value, not just cash.
        # When a trade opens, position_size is subtracted from current_balance,
        # but the capital is allocated, not lost. Drawdown should reflect
        # realized losses only.
        equity = portfolio_balance + open_position_value
        result = self.check_drawdown(equity, initial_balance)
        if not result.allowed:
            return result
        
        # Check 2: Daily loss limit
        result = self.check_daily_loss(daily_pnl, initial_balance)
        if not result.allowed:
            return result
        
        # Check 3: Correlation
        if open_positions:
            result = self.check_correlation(symbol, direction, open_positions)
            if not result.allowed:
                return result
        
        # Check 4: Consecutive losses (warning, not blocking)
        if consecutive_losses >= self.CONSECUTIVE_LOSS_THRESHOLD:
            return RiskCheckResult(
                allowed=True,
                reason=f"Warning: {consecutive_losses} consecutive losses. Position size reduced by {int(self.POSITION_REDUCTION_FACTOR * 100)}%.",
                risk_level="high"
            )
        
        return RiskCheckResult(
            allowed=True,
            reason="All risk checks passed",
            risk_level="low"
        )
    
    def check_drawdown(
        self,
        current_balance: float,
        initial_balance: float,
    ) -> RiskCheckResult:
        """Check if portfolio drawdown exceeds circuit breaker threshold."""
        if initial_balance <= 0:
            return RiskCheckResult(
                allowed=False,
                reason="Invalid initial balance",
                risk_level="blocked"
            )
        
        drawdown = (initial_balance - current_balance) / initial_balance
        
        if drawdown >= self.MAX_DRAWDOWN_PCT:
            return RiskCheckResult(
                allowed=False,
                reason=f"CIRCUIT BREAKER: Portfolio drawdown {drawdown:.1%} exceeds {self.MAX_DRAWDOWN_PCT:.0%} limit. All trading halted.",
                risk_level="blocked"
            )
        
        if drawdown >= self.MAX_DRAWDOWN_PCT * 0.8:  # 80% of limit = warning
            return RiskCheckResult(
                allowed=True,
                reason=f"Warning: Drawdown at {drawdown:.1%}, approaching {self.MAX_DRAWDOWN_PCT:.0%} circuit breaker.",
                risk_level="high"
            )
        
        return RiskCheckResult(allowed=True, reason="Drawdown OK", risk_level="low")
    
    def check_daily_loss(
        self,
        daily_pnl: float,
        initial_balance: float,
    ) -> RiskCheckResult:
        """Check if daily loss limit has been hit."""
        if initial_balance <= 0:
            return RiskCheckResult(allowed=False, reason="Invalid balance", risk_level="blocked")
        
        daily_loss_pct = abs(min(daily_pnl, 0)) / initial_balance
        
        if daily_loss_pct >= self.DAILY_LOSS_LIMIT:
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily loss limit hit: -{daily_loss_pct:.1%} (limit: {self.DAILY_LOSS_LIMIT:.0%}). Trading stopped for today.",
                risk_level="blocked"
            )
        
        return RiskCheckResult(allowed=True, reason="Daily loss OK", risk_level="low")
    
    def check_correlation(
        self,
        symbol: str,
        direction: str,
        open_positions: List[Dict],
    ) -> RiskCheckResult:
        """
        Check if adding this position would create too much correlated exposure.
        
        Uses correlation groups from market_config to avoid stacking correlated
        positions in the same direction.
        """
        if not open_positions:
            return RiskCheckResult(allowed=True, reason="No existing positions", risk_level="low")
        
        # Get correlation groups
        groups = get_correlation_groups()
        
        # Find which group this symbol belongs to
        my_group = None
        config = MARKETS.get(symbol)
        if config and config.correlation_group:
            my_group = config.correlation_group
        
        if not my_group:
            return RiskCheckResult(allowed=True, reason="No correlation group", risk_level="low")
        
        # Check if any open position is in the same group and direction
        group_symbols = groups.get(my_group, [])
        
        for pos in open_positions:
            pos_symbol = pos.get("symbol", "")
            pos_direction = pos.get("direction", "")
            
            if pos_symbol in group_symbols and pos_symbol != symbol:
                if pos_direction == direction:
                    # Same direction in correlated group
                    return RiskCheckResult(
                        allowed=False,
                        reason=f"Correlation block: Already {pos_direction} {pos_symbol} (same group '{my_group}'). "
                               f"Cannot go {direction} {symbol} simultaneously.",
                        risk_level="blocked"
                    )
        
        return RiskCheckResult(allowed=True, reason="Correlation check passed", risk_level="low")
    
    def get_position_size_multiplier(self, consecutive_losses: int) -> float:
        """
        Get position size multiplier based on consecutive losses.
        
        Returns:
            Multiplier (1.0 = normal, 0.5 = reduced after losing streak)
        """
        if consecutive_losses >= self.CONSECUTIVE_LOSS_THRESHOLD:
            return self.POSITION_REDUCTION_FACTOR
        return 1.0
    
    def calculate_drawdown_pct(
        self,
        current_balance: float,
        peak_balance: float,
    ) -> float:
        """Calculate current drawdown from peak."""
        if peak_balance <= 0:
            return 0.0
        return max(0.0, (peak_balance - current_balance) / peak_balance)
    
    def is_circuit_breaker_active(
        self,
        current_balance: float,
        initial_balance: float,
    ) -> bool:
        """Check if circuit breaker should be active."""
        if initial_balance <= 0:
            return True
        drawdown = (initial_balance - current_balance) / initial_balance
        return drawdown >= self.MAX_DRAWDOWN_PCT
