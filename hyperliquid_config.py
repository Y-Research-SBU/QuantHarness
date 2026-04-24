"""
Hyperliquid exchange configuration.

Maps QuantAgent internal symbols (yfinance-style, e.g. "BTC-USD") to
Hyperliquid perpetual names, and holds exchange-specific defaults and
safety limits used by :mod:`hyperliquid_executor` and :mod:`run_live`.

Extending to a new Hyperliquid perpetual: add the symbol to ``SYMBOL_MAP``
and (optionally) give it a non-default entry in ``LEVERAGE_BY_SYMBOL``.
"""

from __future__ import annotations

from typing import Dict, Optional


# ── Symbol mapping ───────────────────────────────────────────────────
# QuantAgent uses yfinance-style tickers internally. Hyperliquid lists
# perpetuals under plain coin names (``BTC``, ``ETH``…). Keep this the
# single source of truth — the executor looks up both directions.
SYMBOL_MAP: Dict[str, str] = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
}

REVERSE_SYMBOL_MAP: Dict[str, str] = {v: k for k, v in SYMBOL_MAP.items()}


# ── Order routing defaults ────────────────────────────────────────────
# ``limit`` uses a post-only-equivalent Gtc order at the signal's entry
# price; ``market`` uses ``Exchange.market_open`` with ``SLIPPAGE_BPS``.
ORDER_TYPE: str = "limit"
SLIPPAGE_BPS: int = 10          # 10 bps = 0.10% max slippage for market orders
LIMIT_TIF: str = "Gtc"          # "Gtc", "Alo", or "Ioc"


# ── Leverage ─────────────────────────────────────────────────────────
DEFAULT_LEVERAGE: int = 3
LEVERAGE_BY_SYMBOL: Dict[str, int] = {
    # Per-symbol overrides go here if a market needs different leverage.
}
USE_CROSS_MARGIN: bool = True   # Cross margin (True) vs isolated (False)


# ── Safety limits (mainnet) ──────────────────────────────────────────
# The executor enforces these on mainnet; on testnet they're advisory
# (still logged, not enforced) so paper-trading validation isn't blocked
# by fake-money position sizing.
MAX_POSITION_USD_MAINNET: float = 100.0         # Hard cap per position
MAX_DAILY_LOSS_USD_MAINNET: float = 50.0        # Kill switch (realized P&L)
MAINNET_CONFIRMATION_PHRASE: str = "I UNDERSTAND"


# ── Env var names ────────────────────────────────────────────────────
PRIVATE_KEY_ENV_VAR: str = "HYPERLIQUID_PRIVATE_KEY"
ACCOUNT_ADDRESS_ENV_VAR: str = "HYPERLIQUID_ACCOUNT_ADDRESS"  # Optional — for agent wallets


# ── Helpers ──────────────────────────────────────────────────────────


def to_hl_symbol(internal_symbol: str) -> Optional[str]:
    """Convert ``"BTC-USD"`` → ``"BTC"``. Returns None if unsupported."""
    return SYMBOL_MAP.get(internal_symbol)


def to_internal_symbol(hl_symbol: str) -> Optional[str]:
    """Convert ``"BTC"`` → ``"BTC-USD"``. Returns None if unsupported."""
    return REVERSE_SYMBOL_MAP.get(hl_symbol)


def leverage_for(internal_symbol: str) -> int:
    """Leverage to use for a given QuantAgent symbol."""
    return LEVERAGE_BY_SYMBOL.get(internal_symbol, DEFAULT_LEVERAGE)


def supported_symbols() -> Dict[str, str]:
    """All (internal → Hyperliquid) symbol pairs we know about."""
    return dict(SYMBOL_MAP)
