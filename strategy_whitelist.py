"""
Strategy whitelist derived from 5-year backtest (2026-04-25).

Source: backtest_results/backtest_optimized_v2_20260425_155648.json
Filter: Sharpe ratio >= 0.30 AND total_trades >= 10
Result: 99 cells across 52 symbols (out of 457 candidate cells)

Selected portfolio metrics (equal-weight on these cells):
  Mean Sharpe:    +0.56
  Mean return:    +18.95%
  Win rate:       51.0%
  Max drawdown:   9.89%
  Profit factor:  1.88

Caveats:
  - Daily candles only (5-year backtest)
  - Kronos strategies tested only on 8-symbol subset (separately)
  - No slippage modeling beyond 0.05% commission
  - Survivorship-biased universe (post-COVID bull included)

The Kronos family (kronos_momentum_confirm, kronos_divergence,
multi_timeframe_kronos) are NOT in this whitelist because they were
tested on a smaller sample. We allow them globally on crypto via
LIVE_KRONOS_ALLOWED_CATEGORIES below until a full Kronos backtest runs.
"""

from typing import Dict, FrozenSet, Iterable, List, Set

# ════════════════════════════════════════════════════════════════
# Whitelist: (symbol -> set of allowed strategy names)
# ════════════════════════════════════════════════════════════════

WHITELIST: Dict[str, FrozenSet[str]] = {
    "AAPL": frozenset({"breakout"}),
    "ADA-USD": frozenset({"momentum"}),
    "AMD": frozenset({"bb_squeeze", "multi_factor"}),
    "AMZN": frozenset({"vwap_reversion", "ema_crossover"}),
    "AR-USD": frozenset({"ema_crossover", "momentum"}),
    "ARM": frozenset({"mean_reversion"}),
    "ATOM-USD": frozenset({"bb_squeeze"}),
    "AVAX-USD": frozenset({"vwap_reversion", "ema_crossover", "bb_squeeze", "multi_factor"}),
    "AVGO": frozenset({"bb_squeeze"}),
    "BTC-USD": frozenset({"ema_crossover"}),
    "CL=F": frozenset({"ema_crossover"}),
    "COIN": frozenset({"multi_factor", "mean_reversion"}),
    "CRV-USD": frozenset({"ema_crossover", "vwap_reversion"}),
    "CRWD": frozenset({"bb_squeeze", "mean_reversion", "breakout"}),
    "DIA": frozenset({"ema_crossover", "bb_squeeze"}),
    "DOGE-USD": frozenset({"vwap_reversion", "ema_crossover"}),
    "DYDX-USD": frozenset({"momentum", "multi_factor", "bb_squeeze"}),
    "EIGEN-USD": frozenset({"ema_crossover"}),
    "ENA-USD": frozenset({"multi_factor", "momentum"}),
    "ETH-USD": frozenset({"vwap_reversion", "bb_squeeze", "multi_factor"}),
    "EURUSD=X": frozenset({"mean_reversion", "ema_crossover"}),
    "FET-USD": frozenset({"bb_squeeze", "vwap_reversion"}),
    "FLOKI-USD": frozenset({"bb_squeeze", "ema_crossover"}),
    "GBPUSD=X": frozenset({"bb_squeeze", "momentum", "multi_factor"}),
    "GC=F": frozenset({"breakout", "multi_factor", "bb_squeeze"}),
    "GOOGL": frozenset({"momentum"}),
    "GS": frozenset({"mean_reversion", "momentum"}),
    "HBAR-USD": frozenset({"momentum"}),
    "JPM": frozenset({"bb_squeeze", "momentum"}),
    "LINK-USD": frozenset({"mean_reversion", "vwap_reversion", "ema_crossover"}),
    "LLY": frozenset({"breakout", "multi_factor"}),
    "MKR-USD": frozenset({"ema_crossover", "momentum"}),
    "MRVL": frozenset({"ema_crossover"}),
    "MSFT": frozenset({"breakout"}),
    "MSTR": frozenset({"multi_factor", "vwap_reversion"}),
    "NET": frozenset({"vwap_reversion", "bb_squeeze"}),
    "NVDA": frozenset({"multi_factor", "breakout"}),
    "ONDO-USD": frozenset({"momentum", "multi_factor"}),
    "PENDLE-USD": frozenset({"ema_crossover", "momentum", "bb_squeeze"}),
    "PLTR": frozenset({"ema_crossover"}),
    "RUNE-USD": frozenset({"vwap_reversion", "multi_factor", "ema_crossover"}),
    "SEI-USD": frozenset({"multi_factor", "momentum"}),
    "SHIB-USD": frozenset({"vwap_reversion", "multi_factor"}),
    "SNOW": frozenset({"bb_squeeze"}),
    "SOL-USD": frozenset({"ema_crossover", "multi_factor", "momentum"}),
    "TIA-USD": frozenset({"mean_reversion"}),
    "TSLA": frozenset({"bb_squeeze", "multi_factor"}),
    "TURBO-USD": frozenset({"ema_crossover", "bb_squeeze"}),
    "WIF-USD": frozenset({"ema_crossover"}),
    "WLD-USD": frozenset({"ema_crossover", "bb_squeeze", "multi_factor", "momentum"}),
    "XLE": frozenset({"vwap_reversion"}),
    "XLF": frozenset({"bb_squeeze"}),
    "XLV": frozenset({"vwap_reversion", "momentum", "mean_reversion"}),
}

# ════════════════════════════════════════════════════════════════
# Blacklist: (symbol, strategy) pairs that lost badly in backtest.
# These are double-disabled even if some other rule would allow them.
# Sharpe < -0.50 AND >=5 trades in the optimized backtest.
# ════════════════════════════════════════════════════════════════

BLACKLIST: Set[tuple] = {
    ("SEI-USD", "mean_reversion"),
    ("ENA-USD", "vwap_reversion"),
    ("WIF-USD", "mean_reversion"),
    ("TSM", "vwap_reversion"),
    ("SOL-USD", "mean_reversion"),
    ("WIF-USD", "momentum"),
    ("ENA-USD", "mean_reversion"),
    ("BTC-USD", "breakout"),
    ("DOGE-USD", "mean_reversion"),
    ("AR-USD", "mean_reversion"),
    ("BTC-USD", "momentum"),
    ("TIA-USD", "multi_factor"),
    ("EIGEN-USD", "multi_factor"),
    ("BTC-USD", "bb_squeeze"),
    ("TIA-USD", "ema_crossover"),
    # Per-asset-class observations (Sharpe heatmap):
    ("AMZN", "multi_factor"),
    ("IWM", "multi_factor"),
    ("XLK", "bb_squeeze"),
    ("GC=F", "mean_reversion"),
    ("XOM", "momentum"),
    ("NVDA", "vwap_reversion"),
    ("XLK", "vwap_reversion"),
    ("GOOGL", "vwap_reversion"),
}

# ════════════════════════════════════════════════════════════════
# Kronos family: not in WHITELIST (untested on full universe yet).
# Allow on crypto only — that's where Kronos was originally validated.
# ════════════════════════════════════════════════════════════════

KRONOS_STRATEGIES: FrozenSet[str] = frozenset({
    "kronos_momentum_confirm",
    "kronos_divergence",
    "multi_timeframe_kronos",
})

KRONOS_ALLOWED_CATEGORIES: FrozenSet[str] = frozenset({"crypto"})


def is_allowed(symbol: str, strategy: str, category: str = "") -> bool:
    """
    Return True if (symbol, strategy) is allowed to trade live.

    Logic (in order):
      1. Blacklist always wins → False.
      2. Kronos strategies → allowed only on configured asset categories
         (default: crypto only).
      3. Symbol present in whitelist → strategy must be in its allowed set.
      4. Symbol NOT in whitelist → False (closed-list policy).
    """
    if (symbol, strategy) in BLACKLIST:
        return False

    if strategy in KRONOS_STRATEGIES:
        return category.lower() in KRONOS_ALLOWED_CATEGORIES

    allowed = WHITELIST.get(symbol)
    if allowed is None:
        return False
    return strategy in allowed


def filter_strategies(
    symbol: str,
    candidate_strategies: Iterable[str],
    category: str = "",
) -> List[str]:
    """Return only the strategies allowed to trade for `symbol`."""
    return [s for s in candidate_strategies if is_allowed(symbol, s, category)]


def whitelist_summary() -> dict:
    """Quick stats for /api/whitelist."""
    cell_count = sum(len(v) for v in WHITELIST.values())
    return {
        "symbols": len(WHITELIST),
        "cells": cell_count,
        "blacklist_pairs": len(BLACKLIST),
        "kronos_strategies": list(KRONOS_STRATEGIES),
        "kronos_allowed_categories": list(KRONOS_ALLOWED_CATEGORIES),
        "source_backtest": "backtest_optimized_v2_20260425_155648.json",
        "selection_filter": "sharpe >= 0.30 AND trades >= 10",
        "expected_sharpe": 0.56,
        "expected_return_pct": 18.95,
        "expected_win_rate": 0.51,
        "expected_mdd_pct": 9.89,
        "expected_profit_factor": 1.88,
    }


__all__ = [
    "WHITELIST",
    "BLACKLIST",
    "KRONOS_STRATEGIES",
    "KRONOS_ALLOWED_CATEGORIES",
    "is_allowed",
    "filter_strategies",
    "whitelist_summary",
]
