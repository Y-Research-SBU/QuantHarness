"""
Real-time price feed for the QuantAgent dashboard.

Provides a :class:`PriceFeed` that keeps a thread-safe cache of the latest
price per symbol and invokes an ``on_update`` callback when prices change:

  * Crypto symbols are subscribed to Binance's public ticker stream
    (``wss://stream.binance.com:9443/stream``) using a single combined
    websocket connection.
  * Non-crypto symbols (stocks / ETFs / forex / commodities) are polled via
    yfinance every 5 s during US equity hours, 60 s otherwise.

The feed is intentionally decoupled from Flask-SocketIO: the dashboard owns
the emit step and wires ``on_update`` to a ``socketio.emit('price_update', …)``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, time as dtime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ────────────────────────── symbol mapping ──────────────────────────

# Any symbol whose Binance stream name differs from a trivial
# ``<coin>usdt`` lowercasing lives here. Driven by TRADINGVIEW_SYMBOLS
# in dashboard.py but we keep a small explicit map so this module stands
# alone (and tests don't need to import dashboard).
_BINANCE_OVERRIDES: Dict[str, str] = {
    "RNDR-USD": "renderusdt",
    # FTM-USD removed — yfinance reports delisted; Sonic (S/SONIC) is unavailable
    # under any tested ticker as of 2026-04-26 (REL-339).
}


def quant_to_binance(symbol: str) -> Optional[str]:
    """Map a QuantAgent symbol (e.g. ``BTC-USD``) to a Binance stream ticker
    (``btcusdt``). Returns ``None`` for non-crypto symbols.
    """
    if not symbol or "-USD" not in symbol:
        return None
    if symbol in _BINANCE_OVERRIDES:
        return _BINANCE_OVERRIDES[symbol]
    base = symbol.split("-USD", 1)[0].lower()
    if not base:
        return None
    return f"{base}usdt"


def build_binance_mapping(symbols: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (quant→binance, binance→quant) for the crypto subset of ``symbols``."""
    q2b: Dict[str, str] = {}
    b2q: Dict[str, str] = {}
    for sym in symbols:
        b = quant_to_binance(sym)
        if b:
            q2b[sym] = b
            b2q[b] = sym
    return q2b, b2q


def split_symbols(symbols: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Partition symbols into (crypto-on-binance, everything-else)."""
    crypto: List[str] = []
    other: List[str] = []
    for s in symbols:
        if quant_to_binance(s):
            crypto.append(s)
        else:
            other.append(s)
    return crypto, other


# ────────────────────────── P&L helpers ──────────────────────────


def compute_unrealized_pnl(
    entry_price: float,
    current_price: float,
    quantity: float,
    direction: str,
) -> float:
    """Return absolute unrealised P&L for one position.

    Uses ``(current - entry) * qty`` for LONG and ``(entry - current) * qty``
    for SHORT. Invalid inputs yield ``0.0`` rather than raising — the feed
    path must never crash the dashboard.
    """
    try:
        entry = float(entry_price)
        current = float(current_price)
        qty = float(quantity)
    except (TypeError, ValueError):
        return 0.0
    if direction == "LONG":
        return (current - entry) * qty
    if direction == "SHORT":
        return (entry - current) * qty
    return 0.0


# ────────────────────────── market hours ──────────────────────────


def _is_us_market_open(now_utc: Optional[datetime] = None) -> bool:
    """Rough approximation of regular US equity hours (Mon–Fri, 13:30–20:00 UTC).

    Ignores holidays deliberately — this only decides between a 5 s and 60 s
    yfinance poll cadence, so occasional over-polling on holidays is fine.
    """
    now = now_utc or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(13, 30) <= t <= dtime(20, 0)


# ────────────────────────── feed core ──────────────────────────


PriceCallback = Callable[[str, float], None]


class PriceFeed:
    """Manages crypto WS + stock polling and exposes the latest prices.

    The feed owns two background threads:
      * Binance combined ticker stream
      * yfinance poller for non-crypto symbols

    ``on_update(symbol, price)`` is invoked from those threads. The callable
    must be thread-safe — in the dashboard we pass ``socketio.emit`` which is.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        on_update: Optional[PriceCallback] = None,
        *,
        stock_poll_sleep: Optional[Callable[[float], None]] = None,
        binance_ws_url: str = "wss://stream.binance.com:9443/stream",
    ) -> None:
        self._symbols: List[str] = list(symbols)
        self._on_update = on_update or (lambda s, p: None)
        self._binance_ws_url = binance_ws_url
        self._sleep = stock_poll_sleep or (lambda s: self._stop.wait(s))

        self.crypto_symbols, self.other_symbols = split_symbols(self._symbols)
        self.q2b, self.b2q = build_binance_mapping(self.crypto_symbols)

        self._prices: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._ws_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._ws_connected = threading.Event()

    # ----- public API -----

    def start(self) -> None:
        if self._ws_thread or self._poll_thread:
            return
        self._stop.clear()
        if self.crypto_symbols:
            self._ws_thread = threading.Thread(
                target=self._run_binance, name="price-feed-binance", daemon=True,
            )
            self._ws_thread.start()
        if self.other_symbols:
            self._poll_thread = threading.Thread(
                target=self._run_stock_poll, name="price-feed-stocks", daemon=True,
            )
            self._poll_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._ws_connected.clear()

    def is_ws_connected(self) -> bool:
        return self._ws_connected.is_set()

    def get_prices(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._prices)

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol)

    def set_price(self, symbol: str, price: float) -> None:
        """Inject a price (primarily for tests)."""
        self._record(symbol, price)

    # ----- internals -----

    def _record(self, symbol: str, price: float) -> None:
        try:
            p = float(price)
        except (TypeError, ValueError):
            return
        if p <= 0:
            return
        with self._lock:
            prev = self._prices.get(symbol)
            self._prices[symbol] = p
        if prev != p:
            try:
                self._on_update(symbol, p)
            except Exception:
                logger.exception("price_feed on_update callback failed for %s", symbol)

    def _binance_stream_url(self) -> str:
        streams = "/".join(f"{b}@ticker" for b in self.q2b.values())
        return f"{self._binance_ws_url}?streams={streams}"

    def _run_binance(self) -> None:
        try:
            import websocket  # type: ignore
        except ImportError:
            logger.warning(
                "websocket-client not installed; crypto WS price feed disabled"
            )
            return

        url = self._binance_stream_url()

        def on_message(_ws, message: str) -> None:
            try:
                payload = json.loads(message)
            except ValueError:
                return
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                return
            sym_upper = data.get("s")
            last = data.get("c")
            if not sym_upper or last is None:
                return
            quant_sym = self.b2q.get(sym_upper.lower())
            if quant_sym:
                self._record(quant_sym, last)

        def on_open(_ws) -> None:
            self._ws_connected.set()
            logger.info("Binance WS connected (%d streams)", len(self.q2b))

        def on_close(_ws, *_args) -> None:
            self._ws_connected.clear()
            logger.info("Binance WS closed")

        def on_error(_ws, err) -> None:
            logger.warning("Binance WS error: %s", err)

        # Auto-reconnect loop with simple backoff.
        backoff = 1.0
        while not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=on_message,
                    on_open=on_open,
                    on_close=on_close,
                    on_error=on_error,
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception:
                logger.exception("Binance WS crashed; reconnecting")
            self._ws_connected.clear()
            if self._stop.is_set():
                break
            self._stop.wait(min(backoff, 30.0))
            backoff = min(backoff * 2, 30.0)

    def _run_stock_poll(self) -> None:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            logger.warning("yfinance not installed; stock price polling disabled")
            return

        while not self._stop.is_set():
            try:
                self._poll_stocks_once(yf)
            except Exception:
                logger.exception("stock price poll failed")
            interval = 5.0 if _is_us_market_open() else 60.0
            self._sleep(interval)

    def _poll_stocks_once(self, yf) -> None:
        if not self.other_symbols:
            return
        try:
            data = yf.download(
                tickers=self.other_symbols,
                period="1d",
                interval="1m",
                progress=False,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
            )
        except Exception as e:
            logger.warning("yfinance download failed: %s", e)
            return
        if data is None or getattr(data, "empty", True):
            return

        # For multi-symbol responses yfinance returns a column MultiIndex
        # ('BTC-USD', 'Close'). For single-symbol it's just ('Close',).
        if len(self.other_symbols) == 1:
            sym = self.other_symbols[0]
            price = self._last_close(data)
            if price is not None:
                self._record(sym, price)
            return

        for sym in self.other_symbols:
            try:
                sub = data[sym]
            except Exception:
                continue
            price = self._last_close(sub)
            if price is not None:
                self._record(sym, price)

    @staticmethod
    def _last_close(df) -> Optional[float]:
        try:
            closes = df["Close"].dropna()
        except Exception:
            return None
        if len(closes) == 0:
            return None
        try:
            return float(closes.iloc[-1])
        except Exception:
            return None
