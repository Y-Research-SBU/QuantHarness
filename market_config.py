"""
Market configuration for multi-market trading.
Defines all tradeable markets, their symbols, timeframes, and strategy assignments.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MarketCategory(str, Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"
    COMMODITIES = "commodities"
    FOREX = "forex"


class StrategyType(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MULTI_FACTOR = "multi_factor"
    # Kronos-powered strategies (see strategies.py)
    KRONOS_MOMENTUM_CONFIRM = "kronos_momentum_confirm"
    KRONOS_DIVERGENCE = "kronos_divergence"
    MULTI_TIMEFRAME_KRONOS = "multi_timeframe_kronos"


@dataclass
class MarketConfig:
    """Configuration for a single market/asset."""
    symbol: str                          # yfinance symbol (e.g., "BTC-USD")
    display_name: str                    # Human-readable name
    category: MarketCategory             # Market category
    timeframes: List[str]                # Timeframes to analyze (e.g., ["4h", "1h"])
    enabled_strategies: List[StrategyType] = field(default_factory=lambda: list(StrategyType))
    scan_interval_hours: float = 4.0     # How often to scan (hours)
    initial_balance: float = 10000.0     # Starting paper balance
    max_position_pct: float = 0.25       # Max % of portfolio per position
    correlation_group: Optional[str] = None  # Group for correlation checks


# All supported markets
_CRYPTO_DEFAULT_STRATEGIES = [
    StrategyType.MOMENTUM,
    StrategyType.MEAN_REVERSION,
    StrategyType.BREAKOUT,
    StrategyType.MULTI_FACTOR,
    StrategyType.KRONOS_MOMENTUM_CONFIRM,
    StrategyType.KRONOS_DIVERGENCE,
    StrategyType.MULTI_TIMEFRAME_KRONOS,
]


MARKETS: Dict[str, MarketConfig] = {
    # Crypto — Kronos strategies enabled by default
    "BTC-USD": MarketConfig(
        symbol="BTC-USD",
        display_name="Bitcoin",
        category=MarketCategory.CRYPTO,
        timeframes=["4h", "1h"],
        enabled_strategies=list(_CRYPTO_DEFAULT_STRATEGIES),
        scan_interval_hours=4.0,
        correlation_group="crypto",
    ),
    "ETH-USD": MarketConfig(
        symbol="ETH-USD",
        display_name="Ethereum",
        category=MarketCategory.CRYPTO,
        timeframes=["4h", "1h"],
        enabled_strategies=list(_CRYPTO_DEFAULT_STRATEGIES),
        scan_interval_hours=4.0,
        correlation_group="crypto",
    ),
    "SOL-USD": MarketConfig(
        symbol="SOL-USD",
        display_name="Solana",
        category=MarketCategory.CRYPTO,
        timeframes=["4h", "1h"],
        enabled_strategies=list(_CRYPTO_DEFAULT_STRATEGIES),
        scan_interval_hours=4.0,
        correlation_group="crypto",
    ),
    # US Stocks
    "SPY": MarketConfig(
        symbol="SPY",
        display_name="S&P 500 ETF",
        category=MarketCategory.STOCKS,
        timeframes=["1d", "4h"],
        scan_interval_hours=24.0,
        correlation_group="us_equity",
    ),
    "QQQ": MarketConfig(
        symbol="QQQ",
        display_name="Nasdaq 100 ETF",
        category=MarketCategory.STOCKS,
        timeframes=["1d", "4h"],
        scan_interval_hours=24.0,
        correlation_group="us_equity",
    ),
    "AAPL": MarketConfig(
        symbol="AAPL",
        display_name="Apple Inc.",
        category=MarketCategory.STOCKS,
        timeframes=["1d", "4h"],
        scan_interval_hours=24.0,
        correlation_group="us_tech",
    ),
    "TSLA": MarketConfig(
        symbol="TSLA",
        display_name="Tesla Inc.",
        category=MarketCategory.STOCKS,
        timeframes=["1d", "4h"],
        scan_interval_hours=24.0,
        correlation_group="us_tech",
    ),
    "NVDA": MarketConfig(
        symbol="NVDA",
        display_name="NVIDIA Corp.",
        category=MarketCategory.STOCKS,
        timeframes=["1d", "4h"],
        scan_interval_hours=24.0,
        correlation_group="us_tech",
    ),
    # Commodities
    "GC=F": MarketConfig(
        symbol="GC=F",
        display_name="Gold Futures",
        category=MarketCategory.COMMODITIES,
        timeframes=["1d"],
        scan_interval_hours=24.0,
        correlation_group="commodities",
    ),
    "CL=F": MarketConfig(
        symbol="CL=F",
        display_name="Crude Oil Futures",
        category=MarketCategory.COMMODITIES,
        timeframes=["1d"],
        scan_interval_hours=24.0,
        correlation_group="commodities",
    ),
    # Forex
    "EURUSD=X": MarketConfig(
        symbol="EURUSD=X",
        display_name="EUR/USD",
        category=MarketCategory.FOREX,
        timeframes=["4h"],
        scan_interval_hours=4.0,
        correlation_group="forex",
    ),
    "GBPUSD=X": MarketConfig(
        symbol="GBPUSD=X",
        display_name="GBP/USD",
        category=MarketCategory.FOREX,
        timeframes=["4h"],
        scan_interval_hours=4.0,
        correlation_group="forex",
    ),
}


def get_markets_by_category(category: MarketCategory) -> Dict[str, MarketConfig]:
    """Get all markets in a specific category."""
    return {k: v for k, v in MARKETS.items() if v.category == category}


def get_all_symbols() -> List[str]:
    """Get all market symbols."""
    return list(MARKETS.keys())


def get_correlation_groups() -> Dict[str, List[str]]:
    """Get symbols grouped by correlation group."""
    groups: Dict[str, List[str]] = {}
    for symbol, config in MARKETS.items():
        if config.correlation_group:
            groups.setdefault(config.correlation_group, []).append(symbol)
    return groups
