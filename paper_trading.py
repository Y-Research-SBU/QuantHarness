"""
Paper Trading Engine for QuantAgent.
Manages virtual portfolios, trade execution, P&L tracking, and portfolio snapshots.
"""

import json
import logging
import sqlite3
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from db_schema import get_connection, init_db
from market_config import MARKETS, MarketConfig
from position_sizing import PositionSizeResult, calculate_position_size, calculate_stop_loss
from risk_manager import RiskCheckResult, RiskManager
from strategies import Signal

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Paper trading engine that manages virtual portfolios and executes paper trades.
    
    Features:
    - One portfolio per market (starting balance: $10,000)
    - Trade logging to SQLite
    - P&L tracking per trade, per strategy, per market
    - Risk management integration
    - Configurable schedule per market
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.risk_manager = RiskManager()
        self._ensure_portfolios()
    
    def _ensure_portfolios(self):
        """Create portfolio records for all configured markets."""
        with get_connection(self.db_path) as conn:
            for symbol, config in MARKETS.items():
                conn.execute(
                    """INSERT OR IGNORE INTO portfolios (symbol, initial_balance, current_balance, peak_balance)
                       VALUES (?, ?, ?, ?)""",
                    (symbol, config.initial_balance, config.initial_balance, config.initial_balance)
                )
    
    def get_portfolio(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data for a symbol."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM portfolios WHERE symbol = ?", (symbol,)
            ).fetchone()
            return dict(row) if row else None
    
    def get_all_portfolios(self) -> List[Dict[str, Any]]:
        """Get all portfolio data."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM portfolios ORDER BY symbol").fetchall()
            return [dict(r) for r in rows]
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open positions, optionally filtered by symbol."""
        with get_connection(self.db_path) as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE status = 'OPEN' AND symbol = ? ORDER BY entry_time",
                    (symbol,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time"
                ).fetchall()
            return [dict(r) for r in rows]
    
    def get_all_open_positions_summary(self) -> List[Dict]:
        """Get summary of all open positions for risk checks."""
        positions = self.get_open_positions()
        return [{"symbol": p["symbol"], "direction": p["direction"]} for p in positions]
    
    def execute_trade(
        self,
        signal: Signal,
        position_size: PositionSizeResult,
    ) -> Optional[int]:
        """
        Execute a paper trade based on a signal and position sizing.
        
        Args:
            signal: Trading signal from strategy
            position_size: Position sizing result
        
        Returns:
            Trade ID if executed, None if rejected
        """
        portfolio = self.get_portfolio(signal.symbol)
        if not portfolio:
            logger.error(f"No portfolio for {signal.symbol}")
            return None
        
        # ── Position stacking guard ──────────────────────────────────
        # Only allow ONE open position per symbol at a time.
        symbol_open_positions = self.get_open_positions(signal.symbol)
        if symbol_open_positions:
            logger.warning(
                f"Position stacking blocked: {signal.symbol} already has "
                f"{len(symbol_open_positions)} open position(s). Skipping new {signal.direction} trade."
            )
            return None

        # Risk check
        open_positions = self.get_all_open_positions_summary()
        # Calculate open position value for this symbol so equity-based
        # drawdown check doesn't treat allocated capital as a loss.
        open_position_value = sum(
            float(p.get("position_size", 0)) for p in symbol_open_positions
        )
        risk_check = self.risk_manager.check_trade_allowed(
            symbol=signal.symbol,
            direction=signal.direction,
            portfolio_balance=portfolio["current_balance"],
            initial_balance=portfolio["initial_balance"],
            daily_pnl=portfolio["daily_pnl"],
            consecutive_losses=portfolio["consecutive_losses"],
            open_positions=open_positions,
            open_position_value=open_position_value,
        )
        
        if not risk_check.allowed:
            logger.warning(f"Trade rejected for {signal.symbol}: {risk_check.reason}")
            return None
        
        # Apply position reduction if on losing streak
        multiplier = self.risk_manager.get_position_size_multiplier(
            portfolio["consecutive_losses"]
        )
        adjusted_size = position_size.position_size_usd * multiplier
        adjusted_qty = position_size.quantity * multiplier
        
        # Check sufficient balance
        if adjusted_size > portfolio["current_balance"]:
            logger.warning(f"Insufficient balance for {signal.symbol}: need ${adjusted_size:.2f}, have ${portfolio['current_balance']:.2f}")
            return None
        
        now = datetime.utcnow().isoformat()
        
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO trades 
                   (symbol, timeframe, strategy, direction, entry_price, position_size,
                    quantity, stop_loss, take_profit, agent_reasoning, decision_json,
                    entry_time, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
                (
                    signal.symbol,
                    signal.timeframe,
                    signal.strategy.value,
                    signal.direction,
                    signal.entry_price,
                    adjusted_size,
                    adjusted_qty,
                    position_size.stop_loss,
                    position_size.take_profit,
                    signal.reasoning,
                    json.dumps(signal.metadata, default=str),
                    now,
                )
            )
            trade_id = cursor.lastrowid
            
            # Deduct position cost from balance
            conn.execute(
                "UPDATE portfolios SET current_balance = current_balance - ?, total_trades = total_trades + 1, updated_at = ? WHERE symbol = ?",
                (adjusted_size, now, signal.symbol)
            )
            
            logger.info(f"Opened trade #{trade_id}: {signal.direction} {signal.symbol} @ {signal.entry_price} (${adjusted_size:.2f})")
            return trade_id
    
    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[Dict[str, Any]]:
        """
        Close an open paper trade.
        
        Args:
            trade_id: ID of the trade to close
            exit_price: Price at which to close
            reason: Reason for closing
        
        Returns:
            Trade details dict if successful, None otherwise
        """
        with get_connection(self.db_path) as conn:
            trade = conn.execute(
                "SELECT * FROM trades WHERE id = ? AND status = 'OPEN'", (trade_id,)
            ).fetchone()
            
            if not trade:
                logger.warning(f"Trade #{trade_id} not found or already closed")
                return None
            
            trade = dict(trade)
            
            # Calculate P&L
            if trade["direction"] == "LONG":
                pnl = (exit_price - trade["entry_price"]) * trade["quantity"]
            else:  # SHORT
                pnl = (trade["entry_price"] - exit_price) * trade["quantity"]
            
            pnl_pct = pnl / trade["position_size"] if trade["position_size"] > 0 else 0
            
            now = datetime.utcnow().isoformat()
            status = "STOPPED" if reason == "stop_loss" else "CLOSED"
            
            # Update trade
            conn.execute(
                """UPDATE trades 
                   SET exit_price = ?, pnl = ?, pnl_pct = ?, status = ?, exit_time = ?, updated_at = ?
                   WHERE id = ?""",
                (exit_price, pnl, pnl_pct, status, now, now, trade_id)
            )
            
            # Update portfolio
            symbol = trade["symbol"]
            portfolio = dict(conn.execute(
                "SELECT * FROM portfolios WHERE symbol = ?", (symbol,)
            ).fetchone())
            
            # Return position cost + P&L
            new_balance = portfolio["current_balance"] + trade["position_size"] + pnl
            new_total_pnl = portfolio["total_pnl"] + pnl
            new_daily_pnl = portfolio["daily_pnl"] + pnl
            
            # Track wins/losses — $0 P&L is breakeven, not a loss (REL-313)
            if pnl > 0:
                new_winning = portfolio["winning_trades"] + 1
                new_losing = portfolio["losing_trades"]
                new_consecutive_losses = 0
            elif pnl < 0:
                new_winning = portfolio["winning_trades"]
                new_losing = portfolio["losing_trades"] + 1
                new_consecutive_losses = portfolio["consecutive_losses"] + 1
            else:
                # Breakeven — don't count as win or loss, don't affect streak
                new_winning = portfolio["winning_trades"]
                new_losing = portfolio["losing_trades"]
                new_consecutive_losses = portfolio["consecutive_losses"]
            
            # Update peak balance and drawdown
            new_peak = max(portfolio["peak_balance"], new_balance)
            new_drawdown = max(
                portfolio["max_drawdown"],
                (new_peak - new_balance) / new_peak if new_peak > 0 else 0
            )
            
            # Circuit breaker — include remaining open position values
            # so allocated capital isn't mistaken for realized losses.
            remaining_open = conn.execute(
                "SELECT COALESCE(SUM(position_size), 0) FROM trades WHERE symbol = ? AND status = 'OPEN' AND id != ?",
                (symbol, trade_id)
            ).fetchone()[0]
            equity_after_close = new_balance + float(remaining_open)
            is_cb = 1 if self.risk_manager.is_circuit_breaker_active(
                equity_after_close, portfolio["initial_balance"]
            ) else 0
            
            conn.execute(
                """UPDATE portfolios 
                   SET current_balance = ?, total_pnl = ?, daily_pnl = ?,
                       winning_trades = ?, losing_trades = ?, consecutive_losses = ?,
                       peak_balance = ?, max_drawdown = ?, is_circuit_breaker_active = ?,
                       updated_at = ?
                   WHERE symbol = ?""",
                (new_balance, new_total_pnl, new_daily_pnl,
                 new_winning, new_losing, new_consecutive_losses,
                 new_peak, new_drawdown, is_cb, now, symbol)
            )
            
            # Update strategy performance
            self._update_strategy_performance(conn, trade, pnl)
            
            logger.info(f"Closed trade #{trade_id}: {trade['direction']} {symbol} @ {exit_price} | P&L: ${pnl:.2f} ({pnl_pct:.1%})")
            
            trade["exit_price"] = exit_price
            trade["pnl"] = pnl
            trade["pnl_pct"] = pnl_pct
            trade["status"] = status
            return trade
    
    def check_stops(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check all open positions against current prices for stop-loss/take-profit.
        
        Args:
            current_prices: Dict of symbol -> current price
        
        Returns:
            List of closed trade details
        """
        closed = []
        open_trades = self.get_open_positions()
        
        for trade in open_trades:
            symbol = trade["symbol"]
            if symbol not in current_prices:
                continue
            
            price = current_prices[symbol]
            
            # Check stop loss
            if trade["direction"] == "LONG":
                if trade["stop_loss"] and price <= trade["stop_loss"]:
                    result = self.close_trade(trade["id"], trade["stop_loss"], reason="stop_loss")
                    if result:
                        closed.append(result)
                    continue
                if trade["take_profit"] and price >= trade["take_profit"]:
                    result = self.close_trade(trade["id"], trade["take_profit"], reason="take_profit")
                    if result:
                        closed.append(result)
                    continue
            else:  # SHORT
                if trade["stop_loss"] and price >= trade["stop_loss"]:
                    result = self.close_trade(trade["id"], trade["stop_loss"], reason="stop_loss")
                    if result:
                        closed.append(result)
                    continue
                if trade["take_profit"] and price <= trade["take_profit"]:
                    result = self.close_trade(trade["id"], trade["take_profit"], reason="take_profit")
                    if result:
                        closed.append(result)
                    continue
        
        return closed
    
    def take_snapshot(self, symbol: str):
        """Take a portfolio snapshot for equity curve tracking."""
        portfolio = self.get_portfolio(symbol)
        if not portfolio:
            return
        
        open_positions = self.get_open_positions(symbol)
        open_count = len(open_positions)
        # Use equity (cash + allocated position value) for accurate snapshots
        open_position_value = sum(
            float(p.get("position_size", 0)) for p in open_positions
        )
        equity = portfolio["current_balance"] + open_position_value
        drawdown_pct = self.risk_manager.calculate_drawdown_pct(
            equity, portfolio["peak_balance"]
        )
        
        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO portfolio_snapshots 
                   (symbol, balance, total_pnl, open_positions, drawdown_pct)
                   VALUES (?, ?, ?, ?, ?)""",
                (symbol, equity, portfolio["total_pnl"],
                 open_count, drawdown_pct)
            )
    
    def reset_daily_pnl(self):
        """Reset daily P&L for all portfolios (call at start of trading day)."""
        today = date.today().isoformat()
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE portfolios SET daily_pnl = 0, daily_pnl_reset_date = ?",
                (today,)
            )
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get trade history with optional filters."""
        with get_connection(self.db_path) as conn:
            query = "SELECT * FROM trades WHERE status != 'OPEN'"
            params: list = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            query += " ORDER BY exit_time DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    
    def _update_strategy_performance(
        self,
        conn: sqlite3.Connection,
        trade: Dict,
        pnl: float,
    ):
        """Update strategy performance tracking after a trade closes."""
        strategy = trade["strategy"]
        symbol = trade["symbol"]
        timeframe = trade["timeframe"]
        
        # Get or create performance record
        perf = conn.execute(
            """SELECT * FROM strategy_performance 
               WHERE strategy = ? AND symbol = ? AND timeframe = ?""",
            (strategy, symbol, timeframe)
        ).fetchone()
        
        if perf:
            perf = dict(perf)
            total_trades = perf["total_trades"] + 1
            winning = perf["winning_trades"] + (1 if pnl > 0 else 0)
            losing = perf["losing_trades"] + (1 if pnl < 0 else 0)  # $0 = breakeven, not loss
            total_pnl = perf["total_pnl"] + pnl
            
            # Update averages
            if pnl > 0:
                avg_win = (perf["avg_win"] * perf["winning_trades"] + pnl) / winning if winning > 0 else 0
                avg_loss = perf["avg_loss"]
            elif pnl < 0:
                avg_win = perf["avg_win"]
                avg_loss = (perf["avg_loss"] * perf["losing_trades"] + abs(pnl)) / losing if losing > 0 else 0
            else:
                # Breakeven — no change to averages
                avg_win = perf["avg_win"]
                avg_loss = perf["avg_loss"]
            
            win_rate = winning / total_trades if total_trades > 0 else 0
            profit_factor = (avg_win * winning) / (avg_loss * losing) if (avg_loss * losing) > 0 else 0
            
            conn.execute(
                """UPDATE strategy_performance 
                   SET total_trades = ?, winning_trades = ?, losing_trades = ?,
                       total_pnl = ?, avg_win = ?, avg_loss = ?, win_rate = ?,
                       profit_factor = ?, updated_at = datetime('now')
                   WHERE strategy = ? AND symbol = ? AND timeframe = ?""",
                (total_trades, winning, losing, total_pnl, avg_win, avg_loss,
                 win_rate, profit_factor, strategy, symbol, timeframe)
            )
        else:
            avg_win = pnl if pnl > 0 else 0
            avg_loss = abs(pnl) if pnl < 0 else 0
            winning = 1 if pnl > 0 else 0
            losing = 1 if pnl < 0 else 0
            
            conn.execute(
                """INSERT INTO strategy_performance 
                   (strategy, symbol, timeframe, total_trades, winning_trades, losing_trades,
                    total_pnl, avg_win, avg_loss, win_rate, profit_factor)
                   VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, 0)""",
                (strategy, symbol, timeframe, winning, losing, pnl, avg_win, avg_loss,
                 winning)  # win_rate = winning (0 or 1) for first trade
            )
    
    def log_api_cost(
        self,
        symbol: str,
        timeframe: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        operation: str,
        cost_per_1k_input: float = 0.0015,
        cost_per_1k_output: float = 0.002,
    ):
        """Log API cost for a model call."""
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = (prompt_tokens / 1000 * cost_per_1k_input +
                         completion_tokens / 1000 * cost_per_1k_output)
        
        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO api_costs 
                   (symbol, timeframe, model, prompt_tokens, completion_tokens,
                    total_tokens, estimated_cost_usd, operation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, timeframe, model, prompt_tokens, completion_tokens,
                 total_tokens, estimated_cost, operation)
            )
