"""
Paper Trading Engine for QuantAgent.

Unified-portfolio model: a single MASTER portfolio holds all capital; any
market can draw from the shared pool. Per-symbol portfolio rows exist for
tracking per-market P&L analytics (total_pnl, wins/losses, trade count) but
do not hold any cash of their own.
"""

import json
import logging
import sqlite3
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from db_schema import get_connection, init_db
from market_config import MARKETS, MarketConfig
from position_sizing import PositionSizeResult, calculate_position_size, calculate_stop_loss
from risk_manager import RiskCheckResult, RiskManager
from strategies import Signal

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Paper trading engine with a single unified portfolio.

    Features:
    - ONE master portfolio ($10,000 starting capital) shared across all markets
    - Per-symbol rows track realised P&L analytics only (no cash)
    - Max concurrent positions, max exposure, and min/max position size caps
    - Trade logging to SQLite
    - Risk management integration (circuit breaker on master drawdown,
      correlation limits, loss-streak position reduction)
    """

    MASTER_SYMBOL = "__MASTER__"
    MASTER_INITIAL_BALANCE = 10000.0

    MAX_POSITIONS = 20              # Max concurrent open positions across all markets
    MAX_POSITION_SIZE = 500.0       # Max $ per position
    MIN_POSITION_SIZE = 50.0        # Min $ per position
    MAX_EXPOSURE_PCT = 0.60         # Max 60% of master portfolio deployed at once

    # Post-close cooldown: after a trade on a symbol is closed or stopped out,
    # block re-entries on that symbol for this many minutes. Prevents the
    # oscillation where a stopped-out SHORT gets reopened on the next scan
    # because the strategy is still firing against the prior setup.
    COOLDOWN_MINUTES = 30

    # Reject trades whose entry_price / stop_loss / take_profit come in at or
    # below this value. Catches data-feed bugs that return 0.0 for very low
    # priced tokens (which would disable stop/tp entirely since price >= 0).
    MIN_PRICE = 1e-8

    # Per-symbol circuit breaker: after N consecutive stop-losses on a single
    # symbol, pause that symbol for SYMBOL_CB_COOLDOWN_MINUTES. Prevents the
    # bot from repeatedly shorting into a rally (INJ-USD problem).
    SYMBOL_CB_MAX_CONSECUTIVE_LOSSES = 3
    SYMBOL_CB_COOLDOWN_MINUTES = 120  # 2 hours

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.risk_manager = RiskManager()
        self._ensure_portfolios()

    def _ensure_portfolios(self):
        """Create the master portfolio plus per-symbol analytics rows."""
        with get_connection(self.db_path) as conn:
            # Master portfolio: holds all capital.
            conn.execute(
                """INSERT OR IGNORE INTO portfolios
                   (symbol, initial_balance, current_balance, peak_balance)
                   VALUES (?, ?, ?, ?)""",
                (self.MASTER_SYMBOL, self.MASTER_INITIAL_BALANCE,
                 self.MASTER_INITIAL_BALANCE, self.MASTER_INITIAL_BALANCE)
            )
            # Per-symbol analytics rows: no capital, just P&L tracking.
            for symbol in MARKETS:
                conn.execute(
                    """INSERT OR IGNORE INTO portfolios
                       (symbol, initial_balance, current_balance, peak_balance)
                       VALUES (?, 0.0, 0.0, 0.0)""",
                    (symbol,)
                )

    # ────────────────────────── Portfolio reads ──────────────────────────

    def get_master_portfolio(self) -> Dict[str, Any]:
        """Return the single unified portfolio row."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM portfolios WHERE symbol = ?", (self.MASTER_SYMBOL,)
            ).fetchone()
            if row is None:
                # Shouldn't happen after _ensure_portfolios, but be defensive.
                return {
                    "symbol": self.MASTER_SYMBOL,
                    "initial_balance": self.MASTER_INITIAL_BALANCE,
                    "current_balance": self.MASTER_INITIAL_BALANCE,
                    "total_pnl": 0.0,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "consecutive_losses": 0,
                    "peak_balance": self.MASTER_INITIAL_BALANCE,
                    "max_drawdown": 0.0,
                    "is_circuit_breaker_active": 0,
                    "daily_pnl": 0.0,
                }
            return dict(row)

    def get_available_capital(self) -> float:
        """Master cash balance (already excludes allocated open positions)."""
        return float(self.get_master_portfolio()["current_balance"])

    def get_total_exposure(self) -> float:
        """Sum of all open position sizes across every market."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(position_size), 0) AS total FROM trades WHERE status = 'OPEN'"
            ).fetchone()
            return float(row["total"] or 0.0)

    def get_portfolio(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data for a symbol (master or per-symbol analytics row)."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM portfolios WHERE symbol = ?", (symbol,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_portfolios(self) -> List[Dict[str, Any]]:
        """Get all portfolio rows except the master."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM portfolios WHERE symbol != ? ORDER BY symbol",
                (self.MASTER_SYMBOL,)
            ).fetchall()
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

    def _recent_or_open_trade(self, symbol: str) -> bool:
        """True if ``symbol`` has an OPEN trade OR a trade closed within the
        cooldown window OR the per-symbol circuit breaker is active.
        Logs the reason so callers don't need to.
        """
        cutoff = (datetime.utcnow() - timedelta(minutes=self.COOLDOWN_MINUTES)).isoformat()
        cb_cutoff = (datetime.utcnow() - timedelta(minutes=self.SYMBOL_CB_COOLDOWN_MINUTES)).isoformat()
        with get_connection(self.db_path) as conn:
            open_row = conn.execute(
                "SELECT id FROM trades WHERE symbol = ? AND status = 'OPEN' LIMIT 1",
                (symbol,)
            ).fetchone()
            if open_row:
                logger.warning(
                    f"Position stacking blocked: {symbol} already has an open trade."
                )
                return True

            # Per-symbol circuit breaker: count consecutive recent stop-losses.
            if self.SYMBOL_CB_MAX_CONSECUTIVE_LOSSES > 0:
                recent_trades = conn.execute(
                    """SELECT status FROM trades
                       WHERE symbol = ? AND status != 'OPEN'
                         AND exit_time IS NOT NULL AND exit_time >= ?
                       ORDER BY exit_time DESC LIMIT ?""",
                    (symbol, cb_cutoff, self.SYMBOL_CB_MAX_CONSECUTIVE_LOSSES)
                ).fetchall()
                if len(recent_trades) >= self.SYMBOL_CB_MAX_CONSECUTIVE_LOSSES:
                    all_stopped = all(r["status"] == "STOPPED" for r in recent_trades)
                    if all_stopped:
                        logger.warning(
                            f"Symbol circuit breaker: {symbol} has "
                            f"{len(recent_trades)} consecutive stop-losses in the "
                            f"last {self.SYMBOL_CB_COOLDOWN_MINUTES}min — paused."
                        )
                        return True

            if self.COOLDOWN_MINUTES <= 0:
                return False
            recent = conn.execute(
                """SELECT id, status, exit_time FROM trades
                   WHERE symbol = ? AND status != 'OPEN'
                     AND exit_time IS NOT NULL AND exit_time >= ?
                   ORDER BY exit_time DESC LIMIT 1""",
                (symbol, cutoff)
            ).fetchone()
        if recent:
            logger.warning(
                f"Cooldown active for {symbol}: last trade #{recent['id']} "
                f"({recent['status']}) closed at {recent['exit_time']}; "
                f"waiting {self.COOLDOWN_MINUTES}min before re-entry."
            )
            return True
        return False

    # ────────────────────────── Trade execution ──────────────────────────

    def execute_trade(
        self,
        signal: Signal,
        position_size: PositionSizeResult,
    ) -> Optional[int]:
        """
        Execute a paper trade against the master portfolio.

        Checks:
        - Symbol is known
        - No existing open position on the same symbol (stacking guard)
        - Total open positions < MAX_POSITIONS
        - Total exposure after this trade < MAX_EXPOSURE_PCT of master initial
        - Risk manager (drawdown, daily loss, correlation) on master equity
        - Adjusted position size within [MIN_POSITION_SIZE, MAX_POSITION_SIZE]
        - Master has enough available cash

        Returns trade ID if executed, None if rejected.
        """
        # Unknown symbol guard
        if signal.symbol not in MARKETS:
            logger.error(f"Unknown market {signal.symbol}")
            return None

        # ── Price sanity check ───────────────────────────────────────
        # Reject trades with zero/near-zero prices — the data feed
        # occasionally returns 0.0 for very low-priced tokens, which
        # breaks stop-loss / take-profit (since price >= 0 always).
        if signal.entry_price is None or float(signal.entry_price) <= self.MIN_PRICE:
            logger.warning(
                f"Rejected {signal.symbol}: entry_price={signal.entry_price} "
                f"is zero or below minimum ({self.MIN_PRICE})"
            )
            return None
        if position_size.stop_loss is None or float(position_size.stop_loss) <= self.MIN_PRICE:
            logger.warning(
                f"Rejected {signal.symbol}: stop_loss={position_size.stop_loss} invalid"
            )
            return None
        if position_size.take_profit is None or float(position_size.take_profit) <= self.MIN_PRICE:
            logger.warning(
                f"Rejected {signal.symbol}: take_profit={position_size.take_profit} invalid"
            )
            return None

        # ── Stacking guard + cooldown ────────────────────────────────
        # Block a new entry if the symbol has an OPEN position OR was recently
        # closed/stopped (within COOLDOWN_MINUTES). The cooldown half also
        # fixes the oscillation where a stopped-out symbol's strategy keeps
        # firing and immediately re-enters.
        if self._recent_or_open_trade(signal.symbol):
            return None

        master = self.get_master_portfolio()
        all_open = self.get_open_positions()

        # Max concurrent positions
        if len(all_open) >= self.MAX_POSITIONS:
            logger.warning(
                f"Max positions ({self.MAX_POSITIONS}) already open — skipping {signal.symbol}"
            )
            return None

        total_exposure = sum(float(p.get("position_size", 0)) for p in all_open)

        # Risk check (uses master equity — cash + allocated capital)
        open_positions = [{"symbol": p["symbol"], "direction": p["direction"]} for p in all_open]
        risk_check = self.risk_manager.check_trade_allowed(
            symbol=signal.symbol,
            direction=signal.direction,
            portfolio_balance=master["current_balance"],
            initial_balance=master["initial_balance"],
            daily_pnl=master["daily_pnl"],
            consecutive_losses=master["consecutive_losses"],
            open_positions=open_positions,
            open_position_value=total_exposure,
        )
        if not risk_check.allowed:
            logger.warning(f"Trade rejected for {signal.symbol}: {risk_check.reason}")
            return None

        # Apply loss-streak multiplier (based on master consecutive losses)
        multiplier = self.risk_manager.get_position_size_multiplier(
            master["consecutive_losses"]
        )
        adjusted_size = position_size.position_size_usd * multiplier

        # Clamp to [MIN, MAX]
        if adjusted_size > self.MAX_POSITION_SIZE:
            adjusted_size = self.MAX_POSITION_SIZE
        if adjusted_size < self.MIN_POSITION_SIZE:
            logger.warning(
                f"Position too small for {signal.symbol}: ${adjusted_size:.2f} < "
                f"${self.MIN_POSITION_SIZE:.2f} minimum"
            )
            return None

        # Max exposure check (against master initial balance, so the cap is stable)
        max_exposure_usd = master["initial_balance"] * self.MAX_EXPOSURE_PCT
        if total_exposure + adjusted_size > max_exposure_usd:
            logger.warning(
                f"Max exposure hit: ${total_exposure:.2f} + ${adjusted_size:.2f} > "
                f"${max_exposure_usd:.2f} ({self.MAX_EXPOSURE_PCT:.0%} of master)"
            )
            return None

        # Sufficient cash check
        if adjusted_size > master["current_balance"]:
            logger.warning(
                f"Insufficient master balance: need ${adjusted_size:.2f}, "
                f"have ${master['current_balance']:.2f}"
            )
            return None

        # Recalc qty proportionally if size was clamped
        if position_size.position_size_usd > 0:
            size_ratio = adjusted_size / position_size.position_size_usd
        else:
            size_ratio = 1.0
        adjusted_qty = position_size.quantity * size_ratio

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

            # Deduct from master balance + bump master trade count
            conn.execute(
                """UPDATE portfolios
                   SET current_balance = current_balance - ?,
                       total_trades = total_trades + 1,
                       updated_at = ?
                   WHERE symbol = ?""",
                (adjusted_size, now, self.MASTER_SYMBOL)
            )

            # Bump per-symbol trade count (analytics)
            conn.execute(
                """UPDATE portfolios
                   SET total_trades = total_trades + 1, updated_at = ?
                   WHERE symbol = ?""",
                (now, signal.symbol)
            )

            # Explicit commit: we want the trade durable before returning, so
            # the log line below is a true reflection of DB state (and any
            # later code — here or in callers — cannot roll it back).
            conn.commit()

            logger.info(
                f"Opened trade #{trade_id}: {signal.direction} {signal.symbol} "
                f"@ {signal.entry_price} (${adjusted_size:.2f})"
            )
            return trade_id

    # ────────────────────────── Close trade ──────────────────────────

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[Dict[str, Any]]:
        """Close an open paper trade, returning capital + P&L to the master."""
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

            # Update the trade row
            conn.execute(
                """UPDATE trades
                   SET exit_price = ?, pnl = ?, pnl_pct = ?, status = ?, exit_time = ?, updated_at = ?
                   WHERE id = ?""",
                (exit_price, pnl, pnl_pct, status, now, now, trade_id)
            )

            symbol = trade["symbol"]

            # ── Update MASTER portfolio ──────────────────────────────
            master = dict(conn.execute(
                "SELECT * FROM portfolios WHERE symbol = ?", (self.MASTER_SYMBOL,)
            ).fetchone())

            master_new_balance = master["current_balance"] + trade["position_size"] + pnl
            master_new_total_pnl = master["total_pnl"] + pnl
            master_new_daily_pnl = master["daily_pnl"] + pnl

            # Wins/losses on master (global streak)
            if pnl > 0:
                master_winning = master["winning_trades"] + 1
                master_losing = master["losing_trades"]
                master_streak = 0
            elif pnl < 0:
                master_winning = master["winning_trades"]
                master_losing = master["losing_trades"] + 1
                master_streak = master["consecutive_losses"] + 1
            else:
                # Breakeven — no streak change
                master_winning = master["winning_trades"]
                master_losing = master["losing_trades"]
                master_streak = master["consecutive_losses"]

            # Peak/drawdown on master equity (cash + still-open positions)
            remaining_open = conn.execute(
                "SELECT COALESCE(SUM(position_size), 0) FROM trades "
                "WHERE status = 'OPEN' AND id != ?",
                (trade_id,)
            ).fetchone()[0]
            equity_after_close = master_new_balance + float(remaining_open)
            master_new_peak = max(master["peak_balance"], equity_after_close)
            master_new_drawdown = max(
                master["max_drawdown"],
                (master_new_peak - equity_after_close) / master_new_peak if master_new_peak > 0 else 0
            )
            master_is_cb = 1 if self.risk_manager.is_circuit_breaker_active(
                equity_after_close, master["initial_balance"]
            ) else 0

            conn.execute(
                """UPDATE portfolios
                   SET current_balance = ?, total_pnl = ?, daily_pnl = ?,
                       winning_trades = ?, losing_trades = ?, consecutive_losses = ?,
                       peak_balance = ?, max_drawdown = ?, is_circuit_breaker_active = ?,
                       updated_at = ?
                   WHERE symbol = ?""",
                (master_new_balance, master_new_total_pnl, master_new_daily_pnl,
                 master_winning, master_losing, master_streak,
                 master_new_peak, master_new_drawdown, master_is_cb, now, self.MASTER_SYMBOL)
            )

            # ── Update per-symbol analytics row ──────────────────────
            sym_row = conn.execute(
                "SELECT * FROM portfolios WHERE symbol = ?", (symbol,)
            ).fetchone()
            if sym_row is not None:
                sym = dict(sym_row)
                sym_new_total_pnl = sym["total_pnl"] + pnl
                sym_new_daily_pnl = sym["daily_pnl"] + pnl
                if pnl > 0:
                    sym_wins = sym["winning_trades"] + 1
                    sym_losses = sym["losing_trades"]
                    sym_streak = 0
                elif pnl < 0:
                    sym_wins = sym["winning_trades"]
                    sym_losses = sym["losing_trades"] + 1
                    sym_streak = sym["consecutive_losses"] + 1
                else:
                    sym_wins = sym["winning_trades"]
                    sym_losses = sym["losing_trades"]
                    sym_streak = sym["consecutive_losses"]

                conn.execute(
                    """UPDATE portfolios
                       SET total_pnl = ?, daily_pnl = ?,
                           winning_trades = ?, losing_trades = ?, consecutive_losses = ?,
                           updated_at = ?
                       WHERE symbol = ?""",
                    (sym_new_total_pnl, sym_new_daily_pnl,
                     sym_wins, sym_losses, sym_streak, now, symbol)
                )

            # Update strategy performance
            self._update_strategy_performance(conn, trade, pnl)

            logger.info(
                f"Closed trade #{trade_id}: {trade['direction']} {symbol} @ {exit_price} "
                f"| P&L: ${pnl:.2f} ({pnl_pct:.1%})"
            )

            trade["exit_price"] = exit_price
            trade["pnl"] = pnl
            trade["pnl_pct"] = pnl_pct
            trade["status"] = status

            # L5: Kronos outcome evaluation on the same connection so we
            # stay inside one transaction (avoids a nested write lock).
            self._record_kronos_outcome_safely(conn, trade, exit_price)
            return trade

    # ────────────────────────── Stops & snapshots ──────────────────────────

    def mark_to_market(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Update unrealised P&L on every OPEN trade using current prices.

        Writes ``pnl`` and ``pnl_pct`` on the trades table in-place; the
        status stays ``OPEN``. Missing / non-positive prices are skipped so a
        stale feed on one symbol doesn't zero out P&L on the rest.

        Returns a summary dict with:
          - ``total_unrealized_pnl``: sum of unrealised P&L across marked positions
          - ``positions_marked``: how many OPEN trades were updated
          - ``positions_skipped``: open trades we couldn't price (missing/invalid)
        """
        open_trades = self.get_open_positions()
        total_unrealized_pnl = 0.0
        marked = 0
        skipped = 0
        now = datetime.utcnow().isoformat()

        with get_connection(self.db_path) as conn:
            for trade in open_trades:
                symbol = trade["symbol"]
                price = current_prices.get(symbol)
                if price is None or float(price) <= 0:
                    skipped += 1
                    continue
                price = float(price)

                qty = float(trade["quantity"] or 0.0)
                if trade["direction"] == "LONG":
                    pnl = (price - float(trade["entry_price"])) * qty
                else:  # SHORT
                    pnl = (float(trade["entry_price"]) - price) * qty

                size = float(trade["position_size"] or 0.0)
                pnl_pct = (pnl / size) if size > 0 else 0.0

                conn.execute(
                    """UPDATE trades
                       SET pnl = ?, pnl_pct = ?, updated_at = ?
                       WHERE id = ? AND status = 'OPEN'""",
                    (pnl, pnl_pct, now, trade["id"]),
                )
                total_unrealized_pnl += pnl
                marked += 1

        return {
            "total_unrealized_pnl": total_unrealized_pnl,
            "positions_marked": marked,
            "positions_skipped": skipped,
        }

    def check_stops(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check all open positions for stop-loss/take-profit hits."""
        closed = []
        open_trades = self.get_open_positions()

        for trade in open_trades:
            symbol = trade["symbol"]
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

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
        """
        Take a portfolio snapshot for equity curve tracking.

        For MASTER: snapshot master equity (cash + all open position values).
        For a per-symbol row: snapshot cumulative P&L + the symbol's open position value.
        Unknown symbols are silently ignored.
        """
        portfolio = self.get_portfolio(symbol)
        if not portfolio:
            return

        if symbol == self.MASTER_SYMBOL:
            total_open_value = self.get_total_exposure()
            equity = portfolio["current_balance"] + total_open_value
            drawdown_pct = self.risk_manager.calculate_drawdown_pct(
                equity, portfolio["peak_balance"]
            )
            open_count = len(self.get_open_positions())
        else:
            open_positions = self.get_open_positions(symbol)
            open_count = len(open_positions)
            open_position_value = sum(
                float(p.get("position_size", 0)) for p in open_positions
            )
            # Per-symbol "equity" = cumulative realised P&L + open allocation
            equity = portfolio["total_pnl"] + open_position_value
            drawdown_pct = 0.0

        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO portfolio_snapshots
                   (symbol, balance, total_pnl, open_positions, drawdown_pct)
                   VALUES (?, ?, ?, ?, ?)""",
                (symbol, equity, portfolio["total_pnl"], open_count, drawdown_pct)
            )

    def reset_daily_pnl(self):
        """Reset daily P&L for master + all per-symbol rows (call at start of day)."""
        today = date.today().isoformat()
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE portfolios SET daily_pnl = 0, daily_pnl_reset_date = ?",
                (today,)
            )

    # ────────────────────────── Trade history ──────────────────────────

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

            if pnl > 0:
                avg_win = (perf["avg_win"] * perf["winning_trades"] + pnl) / winning if winning > 0 else 0
                avg_loss = perf["avg_loss"]
            elif pnl < 0:
                avg_win = perf["avg_win"]
                avg_loss = (perf["avg_loss"] * perf["losing_trades"] + abs(pnl)) / losing if losing > 0 else 0
            else:
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
                 winning)
            )

    def _record_kronos_outcome_safely(
        self,
        conn: sqlite3.Connection,
        trade: Dict[str, Any],
        exit_price: float,
    ) -> None:
        """Evaluate a Kronos prediction's outcome when the trade closes.

        Reads ``kronos_prediction_id`` / ``kronos_entry_price`` from the
        trade's ``decision_json`` (written by the scanner) and updates the
        matching ``kronos_predictions`` row via the caller's open connection
        so we don't fight the close-trade transaction for a write lock.

        Wrapped in a wide try/except: any failure here must not mask the
        actual trade close.
        """
        try:
            raw = trade.get("decision_json")
            if not raw:
                return
            if isinstance(raw, dict):
                md = raw
            else:
                try:
                    md = json.loads(raw) if isinstance(raw, str) else {}
                except (json.JSONDecodeError, TypeError):
                    return
            pred_id = md.get("kronos_prediction_id")
            if pred_id is None:
                return
            entry_price = float(md.get("kronos_entry_price") or trade.get("entry_price") or 0.0)
            if entry_price <= 0:
                return
            actual_pct = (float(exit_price) - entry_price) / entry_price * 100.0
            if actual_pct > 0.1:
                actual_direction = "UP"
            elif actual_pct < -0.1:
                actual_direction = "DOWN"
            else:
                actual_direction = "NEUTRAL"
            # The kronos_predictions table is created by the self-improvement
            # migration; guard against its absence in legacy DBs.
            row = conn.execute(
                "SELECT predicted_direction FROM kronos_predictions WHERE id = ?",
                (int(pred_id),),
            ).fetchone()
            if not row:
                return
            correct = 1 if actual_direction == row["predicted_direction"] else 0
            conn.execute(
                """UPDATE kronos_predictions
                   SET actual_direction = ?, actual_magnitude = ?, actual_price = ?,
                       evaluation_time = datetime('now'), correct = ?
                   WHERE id = ?""",
                (actual_direction, float(actual_pct), float(exit_price), correct, int(pred_id)),
            )
        except sqlite3.OperationalError as exc:
            # e.g. kronos_predictions table doesn't exist on an old DB.
            logger.debug("record_kronos_outcome skipped (schema): %s", exc)
        except Exception as exc:
            logger.debug("record_kronos_outcome skipped: %s", exc)

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
