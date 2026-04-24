"""
Hyperliquid live-trading executor.

Drop-in replacement for :class:`paper_trading.PaperTradingEngine` that routes
every ``execute_trade`` / ``close_trade`` call to the Hyperliquid perp DEX,
while still persisting trades to the same SQLite schema so the existing
dashboard, continuous runner and risk manager keep working.

Design notes
------------
* ``__init__`` defaults to **testnet**. Mainnet requires ``testnet=False``
  *and* an explicit caller (see ``run_live.py``). There is no way to hit
  mainnet by accident — including by forgetting a flag.
* Stop-loss / take-profit are sent to Hyperliquid as ``reduce_only`` trigger
  orders so the exchange manages them server-side. ``check_stops`` doesn't
  re-trigger anything; it reconciles exchange fills back into our SQLite
  records.
* Safety limits (``max_position_usd``, ``max_daily_loss_usd``) are enforced
  at the executor boundary on mainnet. Testnet still *records* them so
  tests/prod behave the same way except for the final refusal.
* All SDK calls are routed through ``self._exchange`` / ``self._info``,
  which makes the class trivially mockable in tests.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

from db_schema import get_connection
from hyperliquid_config import (
    DEFAULT_LEVERAGE,
    LIMIT_TIF,
    MAX_DAILY_LOSS_USD_MAINNET,
    MAX_POSITION_USD_MAINNET,
    ORDER_TYPE,
    PRIVATE_KEY_ENV_VAR,
    SLIPPAGE_BPS,
    USE_CROSS_MARGIN,
    leverage_for,
    to_hl_symbol,
)
from market_config import MARKETS
from position_sizing import PositionSizeResult
from strategies import Signal

logger = logging.getLogger(__name__)


class HyperliquidExecutorError(Exception):
    """Raised when the executor refuses an action or the API returns an error."""


@dataclass
class _Clients:
    """Bundle of Hyperliquid SDK clients + the wallet address they authenticate."""
    exchange: Any
    info: Any
    address: str


def _build_clients(private_key: str, testnet: bool) -> _Clients:
    """Construct ``Exchange`` and ``Info`` clients for the chosen network.

    Imported lazily so tests that mock the clients can avoid pulling in
    ``eth_account`` / network code at module load time.
    """
    from eth_account import Account
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    wallet = Account.from_key(private_key)
    info = Info(base_url=base_url, skip_ws=True)
    exchange = Exchange(wallet=wallet, base_url=base_url)
    return _Clients(exchange=exchange, info=info, address=wallet.address)


class HyperliquidExecutor:
    """
    Live-trading engine that mirrors :class:`PaperTradingEngine`'s public API.

    Parameters
    ----------
    private_key:
        Hex-encoded EOA private key. If omitted, read from the
        ``HYPERLIQUID_PRIVATE_KEY`` env var. Never logged.
    testnet:
        Default True. Must be set to False *explicitly* for mainnet.
    db_path:
        SQLite path for trade logging. Defaults to the package default.
    max_position_usd:
        Hard cap on notional per trade; mainnet default is
        ``MAX_POSITION_USD_MAINNET``. On testnet we still record the limit
        but don't refuse orders, so testnet mirrors mainnet behaviour.
    max_daily_loss_usd:
        Kill-switch threshold on realized daily P&L (absolute dollars).
    default_leverage:
        Cross-margin leverage to set on first order per symbol.
    dry_run:
        If True, build clients and run all pre-flight checks but skip the
        actual ``exchange.order`` / ``market_open`` calls. Useful for
        dress-rehearsal against a real account.
    clients_factory:
        Test hook — inject mock ``_Clients`` instead of talking to the SDK.
    """

    _LEVERAGE_SET: Dict[str, int]

    def __init__(
        self,
        private_key: Optional[str] = None,
        testnet: bool = True,
        db_path: Optional[str] = None,
        max_position_usd: Optional[float] = None,
        max_daily_loss_usd: Optional[float] = None,
        default_leverage: int = DEFAULT_LEVERAGE,
        dry_run: bool = False,
        clients_factory: Optional[Callable[[str, bool], _Clients]] = None,
    ) -> None:
        self.testnet = bool(testnet)
        self.db_path = db_path
        self.dry_run = bool(dry_run)
        self.default_leverage = int(default_leverage)
        # Safety caps: apply mainnet defaults unless caller overrides.
        self.max_position_usd = (
            float(max_position_usd)
            if max_position_usd is not None
            else MAX_POSITION_USD_MAINNET
        )
        self.max_daily_loss_usd = (
            float(max_daily_loss_usd)
            if max_daily_loss_usd is not None
            else MAX_DAILY_LOSS_USD_MAINNET
        )

        key = private_key if private_key is not None else os.environ.get(PRIVATE_KEY_ENV_VAR)
        if not key:
            raise HyperliquidExecutorError(
                f"Missing private key: pass private_key=... or set {PRIVATE_KEY_ENV_VAR}"
            )

        factory = clients_factory or _build_clients
        clients = factory(key, self.testnet)
        self._exchange = clients.exchange
        self._info = clients.info
        self._address = clients.address
        # Track which symbols we've already called update_leverage for this session
        self._LEVERAGE_SET = {}
        # Portfolio rows back the dashboard / performance_summary code paths
        self._ensure_portfolios()

        network = "TESTNET" if self.testnet else "MAINNET"
        logger.info(
            "HyperliquidExecutor ready — network=%s address=%s dry_run=%s max_pos=$%.2f max_daily_loss=$%.2f",
            network, self._address, self.dry_run, self.max_position_usd, self.max_daily_loss_usd,
        )

    # ------------------------------------------------------------------
    # Portfolio / dashboard glue (keeps dashboard & runner happy)
    # ------------------------------------------------------------------

    def _ensure_portfolios(self) -> None:
        """Create portfolio rows for every configured market — same as the paper engine."""
        with get_connection(self.db_path) as conn:
            for symbol, config in MARKETS.items():
                conn.execute(
                    """INSERT OR IGNORE INTO portfolios (symbol, initial_balance, current_balance, peak_balance)
                       VALUES (?, ?, ?, ?)""",
                    (symbol, config.initial_balance, config.initial_balance, config.initial_balance),
                )

    def get_portfolio(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the SQLite-backed portfolio row for a symbol.

        The on-exchange balance is common to every perp so we surface it
        via ``get_account_summary``; the per-symbol portfolio row exists
        to satisfy the existing risk checks and dashboard queries.
        """
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM portfolios WHERE symbol = ?", (symbol,)).fetchone()
            return dict(row) if row else None

    def get_all_portfolios(self) -> List[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM portfolios ORDER BY symbol").fetchall()
            return [dict(r) for r in rows]

    def reset_daily_pnl(self) -> None:
        """Zero out daily_pnl for every portfolio — called at start of each cycle."""
        today = date.today().isoformat()
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE portfolios SET daily_pnl = 0, daily_pnl_reset_date = ?",
                (today,),
            )

    def take_snapshot(self, symbol: str) -> None:
        """Record an equity snapshot for the dashboard equity curve."""
        portfolio = self.get_portfolio(symbol)
        if not portfolio:
            return
        open_positions = self.get_open_positions(symbol)
        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO portfolio_snapshots
                   (symbol, balance, total_pnl, open_positions, drawdown_pct)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    symbol,
                    float(portfolio["current_balance"]),
                    float(portfolio["total_pnl"]),
                    len(open_positions),
                    0.0,
                ),
            )

    def log_api_cost(self, *args, **kwargs) -> None:
        """Compatibility shim — the live executor doesn't incur LLM costs here."""
        # Delegate to the same table PaperTradingEngine uses so scan logs stay unified.
        try:
            # Mirror PaperTradingEngine.log_api_cost signature
            symbol = kwargs.get("symbol") or (args[0] if len(args) > 0 else "")
            timeframe = kwargs.get("timeframe") or (args[1] if len(args) > 1 else "")
            model = kwargs.get("model") or (args[2] if len(args) > 2 else "")
            prompt_tokens = int(kwargs.get("prompt_tokens", args[3] if len(args) > 3 else 0))
            completion_tokens = int(kwargs.get("completion_tokens", args[4] if len(args) > 4 else 0))
            operation = kwargs.get("operation") or (args[5] if len(args) > 5 else "")
        except Exception:
            return
        total = prompt_tokens + completion_tokens
        with get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO api_costs
                   (symbol, timeframe, model, prompt_tokens, completion_tokens,
                    total_tokens, estimated_cost_usd, operation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, timeframe, model, prompt_tokens, completion_tokens, total, 0.0, operation),
            )

    # ------------------------------------------------------------------
    # Exchange queries
    # ------------------------------------------------------------------

    def get_account_summary(self) -> Dict[str, Any]:
        """Live balance / margin / equity from the exchange user state."""
        state = self._info.user_state(self._address)
        margin_summary = state.get("marginSummary", {}) if isinstance(state, dict) else {}
        cross_summary = state.get("crossMarginSummary", {}) if isinstance(state, dict) else {}
        return {
            "address": self._address,
            "testnet": self.testnet,
            "account_value": float(margin_summary.get("accountValue", 0.0) or 0.0),
            "total_margin_used": float(margin_summary.get("totalMarginUsed", 0.0) or 0.0),
            "total_ntl_pos": float(margin_summary.get("totalNtlPos", 0.0) or 0.0),
            "total_raw_usd": float(margin_summary.get("totalRawUsd", 0.0) or 0.0),
            "cross_account_value": float(cross_summary.get("accountValue", 0.0) or 0.0),
            "withdrawable": float(state.get("withdrawable", 0.0) or 0.0) if isinstance(state, dict) else 0.0,
        }

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Open positions from the exchange (authoritative) for the given symbol.

        Returns a list of dicts using the internal (yfinance-style) symbol so
        callers don't need to know about the Hyperliquid naming.
        """
        state = self._info.user_state(self._address)
        asset_positions = state.get("assetPositions", []) if isinstance(state, dict) else []
        out: List[Dict[str, Any]] = []
        for ap in asset_positions:
            pos = ap.get("position", {}) if isinstance(ap, dict) else {}
            coin = pos.get("coin")
            if not coin:
                continue
            # Translate Hyperliquid coin back to our internal symbol.
            from hyperliquid_config import to_internal_symbol
            internal = to_internal_symbol(coin) or coin
            if symbol is not None and internal != symbol:
                continue
            try:
                szi = float(pos.get("szi", 0.0) or 0.0)
            except (TypeError, ValueError):
                szi = 0.0
            if szi == 0:
                continue
            direction = "LONG" if szi > 0 else "SHORT"
            out.append({
                "symbol": internal,
                "hl_coin": coin,
                "direction": direction,
                "quantity": abs(szi),
                "entry_price": float(pos.get("entryPx", 0.0) or 0.0),
                "position_size": float(pos.get("positionValue", 0.0) or 0.0),
                "unrealized_pnl": float(pos.get("unrealizedPnl", 0.0) or 0.0),
                "leverage": (pos.get("leverage") or {}).get("value") if isinstance(pos.get("leverage"), dict) else None,
                "margin_used": float(pos.get("marginUsed", 0.0) or 0.0),
            })
        return out

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Single-symbol wrapper around :meth:`get_open_positions`."""
        positions = self.get_open_positions(symbol)
        return positions[0] if positions else None

    def get_all_open_positions_summary(self) -> List[Dict[str, Any]]:
        """Minimal form used by the risk manager — matches PaperTradingEngine."""
        return [
            {"symbol": p["symbol"], "direction": p["direction"]}
            for p in self.get_open_positions()
        ]

    def get_fills(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Recent fills from the exchange (most recent first)."""
        fills = self._info.user_fills(self._address) or []
        # Hyperliquid returns a list of fill dicts; truncate safely.
        try:
            return list(fills)[:limit]
        except TypeError:
            return []

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Resting orders (including unfilled SL/TP triggers)."""
        try:
            orders = self._info.open_orders(self._address) or []
            return list(orders)
        except TypeError:
            return []

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(
        self,
        signal: Signal,
        position_size: PositionSizeResult,
    ) -> Optional[str]:
        """
        Place an order on Hyperliquid that matches ``signal`` + ``position_size``.

        Returns the Hyperliquid order id (as a string) when the entry was
        accepted — resting or filled — and None if the order was rejected
        (either by our safety limits or by the exchange).

        Also attaches ``reduce_only`` stop-loss and take-profit triggers so
        the exchange will close the position for us if price breaches them.
        """
        hl_coin = to_hl_symbol(signal.symbol)
        if not hl_coin:
            logger.error("Hyperliquid doesn't list %s (no SYMBOL_MAP entry)", signal.symbol)
            return None

        is_buy = signal.direction == "LONG"
        notional = float(position_size.position_size_usd)
        qty = float(position_size.quantity)

        # ── Safety gates ──────────────────────────────────────────────
        if notional <= 0 or qty <= 0:
            logger.warning("Refusing zero/negative size for %s", signal.symbol)
            return None

        if not self.testnet and notional > self.max_position_usd:
            logger.error(
                "BLOCKED (mainnet): position size $%.2f exceeds cap $%.2f for %s",
                notional, self.max_position_usd, signal.symbol,
            )
            return None

        if not self.testnet and self._daily_loss_exceeded():
            logger.error(
                "BLOCKED (mainnet): daily-loss kill-switch engaged ($%.2f cap)",
                self.max_daily_loss_usd,
            )
            return None

        # Refuse to stack on top of an existing position on the same symbol.
        if self.get_position(signal.symbol) is not None:
            logger.warning(
                "Refusing to stack: %s already has an open position on Hyperliquid.",
                signal.symbol,
            )
            return None

        # ── Leverage ──────────────────────────────────────────────────
        target_leverage = leverage_for(signal.symbol) or self.default_leverage
        if self._LEVERAGE_SET.get(signal.symbol) != target_leverage:
            try:
                if not self.dry_run:
                    self._exchange.update_leverage(
                        leverage=target_leverage,
                        name=hl_coin,
                        is_cross=USE_CROSS_MARGIN,
                    )
                self._LEVERAGE_SET[signal.symbol] = target_leverage
            except Exception as exc:
                logger.warning("update_leverage failed for %s: %s", signal.symbol, exc)

        # ── Place the entry order ────────────────────────────────────
        if self.dry_run:
            logger.info(
                "[DRY-RUN] would %s %s qty=%s @ %s (notional=$%.2f) — skipping send.",
                "BUY" if is_buy else "SELL", hl_coin, qty, signal.entry_price, notional,
            )
            oid = f"dryrun-{int(time.time() * 1000)}"
            self._record_trade(signal, position_size, oid, sl_oid=None, tp_oid=None)
            return oid

        entry_response = self._place_entry(
            hl_coin=hl_coin, is_buy=is_buy, qty=qty, limit_px=float(signal.entry_price),
        )
        entry_oid = self._parse_order_id(entry_response)
        if entry_oid is None:
            logger.error("Entry order rejected for %s: %s", signal.symbol, entry_response)
            return None

        # ── SL / TP trigger orders (reduce-only, opposing side) ──────
        sl_oid = self._place_trigger(
            hl_coin=hl_coin,
            close_is_buy=(not is_buy),
            qty=qty,
            trigger_px=float(position_size.stop_loss),
            tpsl="sl",
        ) if position_size.stop_loss else None
        tp_oid = self._place_trigger(
            hl_coin=hl_coin,
            close_is_buy=(not is_buy),
            qty=qty,
            trigger_px=float(position_size.take_profit),
            tpsl="tp",
        ) if position_size.take_profit else None

        self._record_trade(signal, position_size, entry_oid, sl_oid=sl_oid, tp_oid=tp_oid)
        logger.info(
            "Placed %s %s qty=%s @ %s — entry_oid=%s sl_oid=%s tp_oid=%s",
            "BUY" if is_buy else "SELL", hl_coin, qty, signal.entry_price,
            entry_oid, sl_oid, tp_oid,
        )
        return str(entry_oid)

    def close_trade(
        self,
        trade_id: int,
        exit_price: Optional[float] = None,
        reason: str = "manual",
        symbol: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Close the open position via Hyperliquid ``market_close`` and update
        the SQLite row for ``trade_id``.

        The ``exit_price`` / ``symbol`` parameters are optional for symmetry
        with ``PaperTradingEngine.close_trade``. If absent they're read from
        the stored trade row. ``exit_price`` from the caller is ignored if
        the market_close fill comes back with its own price.
        """
        with get_connection(self.db_path) as conn:
            trade = conn.execute(
                "SELECT * FROM trades WHERE id = ? AND status = 'OPEN'", (trade_id,)
            ).fetchone()
            if not trade:
                logger.warning("close_trade: trade #%s not open", trade_id)
                return None
            trade = dict(trade)

        sym = symbol or trade["symbol"]
        hl_coin = to_hl_symbol(sym)
        if not hl_coin:
            logger.error("close_trade: unsupported symbol %s", sym)
            return None

        if self.dry_run:
            fill_price = float(exit_price) if exit_price is not None else float(trade["entry_price"])
        else:
            try:
                resp = self._exchange.market_close(coin=hl_coin, slippage=SLIPPAGE_BPS / 10000.0)
            except Exception as exc:
                logger.exception("market_close failed for %s: %s", sym, exc)
                return None
            fill_price = self._parse_avg_px(resp)
            if fill_price is None:
                fill_price = float(exit_price) if exit_price is not None else float(trade["entry_price"])
            # Cancel any remaining trigger orders for this symbol (SL/TP leftovers).
            try:
                self.cancel_all_orders(sym)
            except Exception:
                pass

        return self._finalize_closed_trade(trade_id, trade, fill_price, reason)

    def check_stops(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Reconcile exchange fills with local open trades.

        Hyperliquid enforces our SL/TP trigger orders server-side, so we
        don't re-check prices here. Instead we look at on-exchange
        positions — any OPEN row whose symbol no longer has a live position
        is treated as closed, and we update SQLite accordingly.
        """
        closed: List[Dict[str, Any]] = []
        try:
            live_symbols = {p["symbol"] for p in self.get_open_positions()}
        except Exception as exc:
            logger.warning("check_stops: could not fetch live positions: %s", exc)
            return closed

        open_rows = self._fetch_open_rows()
        for trade in open_rows:
            sym = trade["symbol"]
            if sym in live_symbols:
                continue
            fill_price = current_prices.get(sym) or self._recent_fill_price(sym)
            if fill_price is None:
                fill_price = float(trade["entry_price"])
            reason = "stop_loss" if self._looks_like_stop(trade, fill_price) else "take_profit"
            result = self._finalize_closed_trade(trade["id"], trade, float(fill_price), reason)
            if result is not None:
                closed.append(result)
        return closed

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel resting orders. Returns the number of cancels attempted.

        When ``symbol`` is given we only cancel that coin's orders; with
        ``None`` we cancel everything.
        """
        orders = self.get_open_orders()
        target_coin = to_hl_symbol(symbol) if symbol else None
        attempts = 0
        for o in orders:
            coin = o.get("coin")
            oid = o.get("oid")
            if oid is None:
                continue
            if target_coin is not None and coin != target_coin:
                continue
            try:
                self._exchange.cancel(coin, int(oid))
                attempts += 1
            except Exception as exc:
                logger.warning("cancel failed for coin=%s oid=%s: %s", coin, oid, exc)
        return attempts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _place_entry(self, hl_coin: str, is_buy: bool, qty: float, limit_px: float) -> Any:
        """Route the entry through ``market_open`` or ``order`` based on config."""
        if ORDER_TYPE == "market":
            return self._exchange.market_open(
                name=hl_coin,
                is_buy=is_buy,
                sz=qty,
                slippage=SLIPPAGE_BPS / 10000.0,
            )
        order_type = {"limit": {"tif": LIMIT_TIF}}
        return self._exchange.order(
            name=hl_coin,
            is_buy=is_buy,
            sz=qty,
            limit_px=limit_px,
            order_type=order_type,
            reduce_only=False,
        )

    def _place_trigger(
        self,
        hl_coin: str,
        close_is_buy: bool,
        qty: float,
        trigger_px: float,
        tpsl: str,   # "tp" or "sl"
    ) -> Optional[int]:
        """Submit a reduce-only trigger order that closes the position server-side."""
        order_type = {
            "trigger": {
                "isMarket": True,
                "triggerPx": trigger_px,
                "tpsl": tpsl,
            }
        }
        try:
            resp = self._exchange.order(
                name=hl_coin,
                is_buy=close_is_buy,
                sz=qty,
                limit_px=trigger_px,
                order_type=order_type,
                reduce_only=True,
            )
            return self._parse_order_id(resp)
        except Exception as exc:
            logger.warning("Failed to place %s trigger for %s: %s", tpsl, hl_coin, exc)
            return None

    @staticmethod
    def _parse_order_id(response: Any) -> Optional[int]:
        """Best-effort pluck of ``oid`` from an Exchange response.

        Hyperliquid responses look like::

            {"status":"ok","response":{"type":"order",
              "data":{"statuses":[{"resting":{"oid": 1234}}]}}}

        or with ``"filled": {...}``. Returns None if we can't find one.
        """
        if not isinstance(response, dict):
            return None
        if response.get("status") and response.get("status") != "ok":
            return None
        data = (response.get("response") or {}).get("data") or {}
        statuses = data.get("statuses") or []
        for st in statuses:
            if not isinstance(st, dict):
                continue
            for key in ("resting", "filled"):
                block = st.get(key)
                if isinstance(block, dict) and "oid" in block:
                    try:
                        return int(block["oid"])
                    except (TypeError, ValueError):
                        return None
        return None

    @staticmethod
    def _parse_avg_px(response: Any) -> Optional[float]:
        """Extract the filled avg price from a market_close / market_open response."""
        if not isinstance(response, dict):
            return None
        data = (response.get("response") or {}).get("data") or {}
        for st in data.get("statuses") or []:
            if isinstance(st, dict):
                filled = st.get("filled")
                if isinstance(filled, dict) and "avgPx" in filled:
                    try:
                        return float(filled["avgPx"])
                    except (TypeError, ValueError):
                        return None
        return None

    def _daily_loss_exceeded(self) -> bool:
        """True when cumulative realized daily P&L across all portfolios is past the cap."""
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT COALESCE(SUM(daily_pnl), 0) FROM portfolios").fetchone()
        total = float(row[0]) if row else 0.0
        return total <= -abs(self.max_daily_loss_usd)

    def _fetch_open_rows(self) -> List[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time"
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _looks_like_stop(trade: Dict[str, Any], fill_price: float) -> bool:
        """Heuristic: was the fill on the stop side of the entry?"""
        sl = trade.get("stop_loss")
        if not sl:
            return False
        direction = trade.get("direction")
        if direction == "LONG":
            return float(fill_price) <= float(sl) * 1.001
        return float(fill_price) >= float(sl) * 0.999

    def _recent_fill_price(self, symbol: str) -> Optional[float]:
        """Last fill price for ``symbol`` from the exchange, if available."""
        hl_coin = to_hl_symbol(symbol)
        if not hl_coin:
            return None
        try:
            for fill in self.get_fills(limit=50):
                if fill.get("coin") == hl_coin and "px" in fill:
                    return float(fill["px"])
        except Exception:
            return None
        return None

    def _record_trade(
        self,
        signal: Signal,
        position_size: PositionSizeResult,
        entry_oid: Optional[int],
        sl_oid: Optional[int],
        tp_oid: Optional[int],
    ) -> None:
        """Insert the trade into SQLite so dashboard/runner see it."""
        now = datetime.utcnow().isoformat()
        metadata = dict(signal.metadata or {})
        metadata.update({
            "exchange": "hyperliquid",
            "network": "testnet" if self.testnet else "mainnet",
            "hl_address": self._address,
            "hl_entry_oid": entry_oid,
            "hl_sl_oid": sl_oid,
            "hl_tp_oid": tp_oid,
            "hl_coin": to_hl_symbol(signal.symbol),
        })
        with get_connection(self.db_path) as conn:
            conn.execute(
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
                    float(signal.entry_price),
                    float(position_size.position_size_usd),
                    float(position_size.quantity),
                    float(position_size.stop_loss) if position_size.stop_loss else None,
                    float(position_size.take_profit) if position_size.take_profit else None,
                    signal.reasoning,
                    json.dumps(metadata, default=str),
                    now,
                ),
            )
            conn.execute(
                "UPDATE portfolios SET total_trades = total_trades + 1, updated_at = ? WHERE symbol = ?",
                (now, signal.symbol),
            )

    def _finalize_closed_trade(
        self,
        trade_id: int,
        trade: Dict[str, Any],
        exit_price: float,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Update trades + portfolios after a close, return the closed trade dict."""
        if trade["direction"] == "LONG":
            pnl = (exit_price - trade["entry_price"]) * trade["quantity"]
        else:
            pnl = (trade["entry_price"] - exit_price) * trade["quantity"]
        pnl_pct = pnl / trade["position_size"] if trade["position_size"] else 0.0
        status = "STOPPED" if reason in ("stop_loss", "take_profit") else "CLOSED"
        now = datetime.utcnow().isoformat()

        with get_connection(self.db_path) as conn:
            conn.execute(
                """UPDATE trades
                   SET exit_price = ?, pnl = ?, pnl_pct = ?, status = ?,
                       exit_time = ?, updated_at = ?
                   WHERE id = ?""",
                (exit_price, pnl, pnl_pct, status, now, now, trade_id),
            )
            symbol = trade["symbol"]
            portfolio = conn.execute(
                "SELECT * FROM portfolios WHERE symbol = ?", (symbol,)
            ).fetchone()
            if portfolio:
                portfolio = dict(portfolio)
                new_total_pnl = portfolio["total_pnl"] + pnl
                new_daily = portfolio["daily_pnl"] + pnl
                if pnl > 0:
                    wins = portfolio["winning_trades"] + 1
                    losses = portfolio["losing_trades"]
                    streak = 0
                elif pnl < 0:
                    wins = portfolio["winning_trades"]
                    losses = portfolio["losing_trades"] + 1
                    streak = portfolio["consecutive_losses"] + 1
                else:
                    wins = portfolio["winning_trades"]
                    losses = portfolio["losing_trades"]
                    streak = portfolio["consecutive_losses"]
                conn.execute(
                    """UPDATE portfolios
                       SET total_pnl = ?, daily_pnl = ?,
                           winning_trades = ?, losing_trades = ?, consecutive_losses = ?,
                           updated_at = ?
                       WHERE symbol = ?""",
                    (new_total_pnl, new_daily, wins, losses, streak, now, symbol),
                )

        logger.info(
            "Closed trade #%s %s %s @ %s (reason=%s) — P&L=$%.2f (%.2f%%)",
            trade_id, trade["direction"], trade["symbol"], exit_price, reason, pnl, pnl_pct * 100,
        )
        trade = dict(trade)
        trade.update({"exit_price": exit_price, "pnl": pnl, "pnl_pct": pnl_pct, "status": status})
        return trade
