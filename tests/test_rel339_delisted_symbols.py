"""REL-339 — Verify delisted symbols stay out of the live config.

yfinance reported the following symbols as delisted on 2026-04-24/26 and they
generated error logs every scan cycle:

    MATIC-USD, UNI-USD, SUI-USD, GRT-USD, IMX-USD,
    STX-USD, FTM-USD, PEPE-USD, POPCAT-USD

Manual yfinance verification on 2026-04-26 confirmed:
  * All nine return empty history for ``period="5d"``.
  * Candidate replacements POL-USD (MATIC rename) and S-USD (FTM rename to
    Sonic) are also empty on yfinance — no working replacement ticker.

These tests are a regression guard so the symbols cannot quietly creep back
into ``MARKETS`` (market_config.py), the strategy whitelist, or the Binance
price-feed overrides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo root importable when running this test file in isolation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_config import MARKETS, get_all_symbols  # noqa: E402
from price_feed import _BINANCE_OVERRIDES, quant_to_binance  # noqa: E402
from strategy_whitelist import WHITELIST  # noqa: E402


# Canonical list — do not edit without a follow-up ticket.
DELISTED_SYMBOLS = (
    "MATIC-USD",
    "UNI-USD",
    "SUI-USD",
    "GRT-USD",
    "IMX-USD",
    "STX-USD",
    "FTM-USD",
    "PEPE-USD",
    "POPCAT-USD",
)


@pytest.mark.parametrize("symbol", DELISTED_SYMBOLS)
def test_delisted_symbol_not_in_markets(symbol: str) -> None:
    """Each delisted symbol must be absent from MARKETS."""
    assert symbol not in MARKETS, (
        f"{symbol} is delisted on yfinance (REL-339) — must not be in MARKETS"
    )


@pytest.mark.parametrize("symbol", DELISTED_SYMBOLS)
def test_delisted_symbol_not_in_get_all_symbols(symbol: str) -> None:
    """Belt-and-braces: also assert against the public accessor."""
    assert symbol not in get_all_symbols(), (
        f"{symbol} surfaced via get_all_symbols() — must be removed"
    )


@pytest.mark.parametrize("symbol", DELISTED_SYMBOLS)
def test_delisted_symbol_not_in_whitelist(symbol: str) -> None:
    """Each delisted symbol must be absent from the strategy whitelist."""
    assert symbol not in WHITELIST, (
        f"{symbol} is delisted (REL-339) — must not be in the strategy whitelist"
    )


@pytest.mark.parametrize("symbol", DELISTED_SYMBOLS)
def test_delisted_symbol_not_in_binance_overrides(symbol: str) -> None:
    """Delisted symbols must not have a Binance stream override.

    The override map only matters when the symbol is also live in MARKETS,
    but keeping it in sync prevents revivals from picking up a stale stream
    name silently.
    """
    assert symbol not in _BINANCE_OVERRIDES, (
        f"{symbol} is delisted (REL-339) — drop its Binance override"
    )


def test_delisted_set_disjoint_from_markets() -> None:
    """Bulk assertion mirroring the parametrised tests."""
    overlap = set(DELISTED_SYMBOLS) & set(MARKETS.keys())
    assert overlap == set(), f"Delisted symbols leaked into MARKETS: {overlap}"


def test_delisted_set_disjoint_from_whitelist() -> None:
    overlap = set(DELISTED_SYMBOLS) & set(WHITELIST.keys())
    assert overlap == set(), f"Delisted symbols leaked into WHITELIST: {overlap}"


def test_quant_to_binance_still_resolves_delisted_form() -> None:
    """``quant_to_binance`` is a pure string mapper and will still produce a
    Binance stream name for any ``*-USD`` input (that's by design — the call
    site filters by MARKETS membership). We assert the *generic* form rather
    than the override, to document that no override exists.
    """
    # MATIC-USD -> 'maticusdt' via the generic path, NOT via an override.
    assert quant_to_binance("MATIC-USD") == "maticusdt"
    assert "MATIC-USD" not in _BINANCE_OVERRIDES
