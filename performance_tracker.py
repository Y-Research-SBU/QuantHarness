"""
Performance tracking and analytics for paper trading.
Calculates: win rate, Sharpe ratio, max drawdown, profit factor, etc.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from db_schema import get_connection

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: str = ""
    expectancy: float = 0.0  # Expected P&L per trade


def calculate_performance(
    trades: List[Dict[str, Any]],
    initial_balance: float = 10000.0,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from a list of closed trades.
    
    Args:
        trades: List of trade dicts with at least 'pnl', 'entry_time', 'exit_time'
        initial_balance: Starting balance for Sharpe/drawdown calculations
    
    Returns:
        PerformanceMetrics dataclass
    """
    metrics = PerformanceMetrics()
    
    if not trades:
        return metrics
    
    pnls = [t.get("pnl", 0.0) for t in trades if t.get("pnl") is not None]
    
    if not pnls:
        return metrics
    
    metrics.total_trades = len(pnls)
    
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    metrics.winning_trades = len(wins)
    metrics.losing_trades = len(losses)
    metrics.win_rate = len(wins) / len(pnls) if pnls else 0.0
    
    metrics.total_pnl = sum(pnls)
    metrics.avg_trade_pnl = np.mean(pnls) if pnls else 0.0
    
    metrics.avg_win = np.mean(wins) if wins else 0.0
    metrics.avg_loss = np.mean([abs(l) for l in losses]) if losses else 0.0
    metrics.avg_win_loss_ratio = (
        metrics.avg_win / metrics.avg_loss if metrics.avg_loss > 0 else float('inf')
    )
    
    metrics.largest_win = max(pnls) if pnls else 0.0
    metrics.largest_loss = min(pnls) if pnls else 0.0
    
    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = sum(abs(l) for l in losses) if losses else 0.0
    metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    # Sharpe ratio (annualized, assuming daily returns)
    metrics.sharpe_ratio = calculate_sharpe_ratio(pnls, initial_balance)
    
    # Max drawdown
    metrics.max_drawdown, metrics.max_drawdown_pct = calculate_max_drawdown(pnls, initial_balance)
    
    # Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
    metrics.expectancy = (
        metrics.win_rate * metrics.avg_win - 
        (1 - metrics.win_rate) * metrics.avg_loss
    )
    
    # Average holding period
    holding_periods = []
    for t in trades:
        entry = t.get("entry_time")
        exit_ = t.get("exit_time")
        if entry and exit_:
            try:
                e = datetime.fromisoformat(str(entry))
                x = datetime.fromisoformat(str(exit_))
                holding_periods.append((x - e).total_seconds())
            except (ValueError, TypeError):
                pass
    
    if holding_periods:
        avg_seconds = np.mean(holding_periods)
        hours = avg_seconds / 3600
        if hours < 1:
            metrics.avg_holding_period = f"{avg_seconds / 60:.0f} minutes"
        elif hours < 24:
            metrics.avg_holding_period = f"{hours:.1f} hours"
        else:
            metrics.avg_holding_period = f"{hours / 24:.1f} days"
    
    return metrics


def calculate_sharpe_ratio(
    pnls: List[float],
    initial_balance: float = 10000.0,
    risk_free_rate: float = 0.04,
    annualization_factor: float = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio from trade P&Ls.
    
    Args:
        pnls: List of trade P&Ls
        initial_balance: Starting balance
        risk_free_rate: Annual risk-free rate (default 4%)
        annualization_factor: Trading days per year (252 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(pnls) < 2:
        return 0.0
    
    # Convert P&Ls to returns
    returns = [p / initial_balance for p in pnls]
    
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / annualization_factor
    
    sharpe = (avg_return - daily_rf) / std_return * math.sqrt(annualization_factor)
    return float(sharpe)


def calculate_max_drawdown(
    pnls: List[float],
    initial_balance: float = 10000.0,
) -> Tuple[float, float]:
    """
    Calculate maximum drawdown in dollars and percentage.
    
    Returns:
        Tuple of (max_drawdown_usd, max_drawdown_pct)
    """
    if not pnls:
        return 0.0, 0.0
    
    # Build equity curve
    equity = [initial_balance]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)
    
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    
    max_dd = float(np.max(drawdown))
    max_dd_pct = float(np.max(drawdown / peak)) if np.max(peak) > 0 else 0.0
    
    return max_dd, max_dd_pct


def get_strategy_performance(
    db_path: Optional[str] = None,
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get strategy performance from the database."""
    with get_connection(db_path) as conn:
        query = "SELECT * FROM strategy_performance WHERE 1=1"
        params: list = []
        
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY total_pnl DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_portfolio_snapshots(
    db_path: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Get portfolio snapshots for equity curve plotting."""
    with get_connection(db_path) as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE symbol = ? ORDER BY snapshot_time LIMIT ?",
                (symbol, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY snapshot_time LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_api_cost_summary(
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Get API cost summary."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT 
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as total_cost,
                COUNT(*) as total_calls,
                AVG(total_tokens) as avg_tokens_per_call
               FROM api_costs"""
        ).fetchone()
        
        if row:
            return dict(row)
        return {"total_tokens": 0, "total_cost": 0, "total_calls": 0, "avg_tokens_per_call": 0}
