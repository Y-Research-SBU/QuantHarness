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
    # StrategyType.MOMENTUM,  # disabled: 17% win rate over 12 trades, sharpe -3.19 (2026-04-24)
    StrategyType.MEAN_REVERSION,
    StrategyType.BREAKOUT,
    StrategyType.MULTI_FACTOR,
    StrategyType.KRONOS_MOMENTUM_CONFIRM,
    # StrategyType.KRONOS_DIVERGENCE,  # disabled: 0% win rate over 8 trades, sharpe -4.64 (2026-04-24)
    StrategyType.MULTI_TIMEFRAME_KRONOS,
]


# Crypto markets fan out across short and long horizons. The intraday
# timeframes (5m/15m) give the self-improvement loop 10–50× more closed
# trades per day; the higher timeframes still anchor longer-trend signals.
_CRYPTO_TIMEFRAMES = ["1d", "4h", "1h", "15m", "5m"]


def _crypto(symbol: str, display_name: str, group: str = "crypto_major") -> MarketConfig:
    """Helper to create a crypto market config with standard settings."""
    return MarketConfig(
        symbol=symbol,
        display_name=display_name,
        category=MarketCategory.CRYPTO,
        timeframes=list(_CRYPTO_TIMEFRAMES),
        enabled_strategies=list(_CRYPTO_DEFAULT_STRATEGIES),
        scan_interval_hours=0.0,
        correlation_group=group,
    )


MARKETS: Dict[str, MarketConfig] = {
    # ══════════════════════════════════════════════════════════════
    # CRYPTO — 50 markets across majors, L1/L2, DeFi, AI, memes
    # ══════════════════════════════════════════════════════════════

    # --- Tier 0: Majors ---
    "BTC-USD": _crypto("BTC-USD", "Bitcoin", "crypto_major"),
    "ETH-USD": _crypto("ETH-USD", "Ethereum", "crypto_major"),
    "SOL-USD": _crypto("SOL-USD", "Solana", "crypto_major"),

    # --- Tier 1: Large-cap alts ---
    "DOGE-USD": _crypto("DOGE-USD", "Dogecoin", "crypto_meme"),
    "ADA-USD": _crypto("ADA-USD", "Cardano", "crypto_l1"),
    "AVAX-USD": _crypto("AVAX-USD", "Avalanche", "crypto_l1"),
    "LINK-USD": _crypto("LINK-USD", "Chainlink", "crypto_defi"),
    "DOT-USD": _crypto("DOT-USD", "Polkadot", "crypto_l1"),
    "MATIC-USD": _crypto("MATIC-USD", "Polygon", "crypto_l2"),
    "ATOM-USD": _crypto("ATOM-USD", "Cosmos", "crypto_l1"),
    # "UNI-USD": removed — yfinance reports delisted (2026-04-24)
    # "ARB-USD": removed — yfinance reports delisted (2026-04-24)
    "AAVE-USD": _crypto("AAVE-USD", "Aave", "crypto_defi"),

    # --- Tier 1b: Ecosystem tokens ---

    "OP-USD": _crypto("OP-USD", "Optimism", "crypto_l2"),
    # "SUI-USD": removed — yfinance reports delisted (2026-04-24)
    # "APT-USD": removed — yfinance reports delisted (2026-04-24)
    "SEI-USD": _crypto("SEI-USD", "Sei", "crypto_l1"),
    "TIA-USD": _crypto("TIA-USD", "Celestia", "crypto_l1"),
    # "INJ-USD": removed — 5 consecutive stop-losses, chronic counter-trend loser (2026-04-24)
    "NEAR-USD": _crypto("NEAR-USD", "NEAR Protocol", "crypto_l1"),

    # --- Tier 2: AI / DePIN tokens ---
    "FET-USD": _crypto("FET-USD", "Fetch.ai", "crypto_ai"),
    "RNDR-USD": _crypto("RNDR-USD", "Render", "crypto_ai"),
    "WLD-USD": _crypto("WLD-USD", "Worldcoin", "crypto_ai"),
    "TAO-USD": _crypto("TAO-USD", "Bittensor", "crypto_ai"),
    "AR-USD": _crypto("AR-USD", "Arweave", "crypto_ai"),

    # --- Tier 2b: DeFi / Liquid staking ---
    "PENDLE-USD": _crypto("PENDLE-USD", "Pendle", "crypto_defi"),
    "ENA-USD": _crypto("ENA-USD", "Ethena", "crypto_defi"),
    "ONDO-USD": _crypto("ONDO-USD", "Ondo Finance", "crypto_defi"),
    "DYDX-USD": _crypto("DYDX-USD", "dYdX", "crypto_defi"),
    "JUP-USD": _crypto("JUP-USD", "Jupiter", "crypto_defi"),
    "RUNE-USD": _crypto("RUNE-USD", "THORChain", "crypto_defi"),

    # --- Tier 2c: Infrastructure ---
    "STX-USD": _crypto("STX-USD", "Stacks", "crypto_l2"),
    "FTM-USD": _crypto("FTM-USD", "Fantom/Sonic", "crypto_l1"),
    "STRK-USD": _crypto("STRK-USD", "Starknet", "crypto_l2"),
    "EIGEN-USD": _crypto("EIGEN-USD", "EigenLayer", "crypto_l2"),

    # --- Tier 3: Meme / Momentum ---
    "PEPE-USD": _crypto("PEPE-USD", "Pepe", "crypto_meme"),
    "WIF-USD": _crypto("WIF-USD", "dogwifhat", "crypto_meme"),
    "BONK-USD": _crypto("BONK-USD", "Bonk", "crypto_meme"),
    "FLOKI-USD": _crypto("FLOKI-USD", "Floki", "crypto_meme"),
    "SHIB-USD": _crypto("SHIB-USD", "Shiba Inu", "crypto_meme"),
    "ORDI-USD": _crypto("ORDI-USD", "ORDI", "crypto_meme"),
    "POPCAT-USD": _crypto("POPCAT-USD", "Popcat", "crypto_meme"),
    "TURBO-USD": _crypto("TURBO-USD", "Turbo", "crypto_meme"),

    # --- Tier 3b: Mid-cap with high vol ---
    "LDO-USD": _crypto("LDO-USD", "Lido DAO", "crypto_defi"),
    "MKR-USD": _crypto("MKR-USD", "Maker", "crypto_defi"),
    "CRV-USD": _crypto("CRV-USD", "Curve", "crypto_defi"),
    # GRT-USD and IMX-USD removed — delisted per yfinance (2026-04-24)
    "HBAR-USD": _crypto("HBAR-USD", "Hedera", "crypto_l1"),

    # ══════════════════════════════════════════════════════════════
    # STOCKS — Indices, Mega-cap tech, Semiconductors, Financials, Energy, etc.
    # ══════════════════════════════════════════════════════════════

    # --- Index ETFs ---
    "SPY": MarketConfig(symbol="SPY", display_name="S&P 500 ETF", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_index"),
    "QQQ": MarketConfig(symbol="QQQ", display_name="Nasdaq 100 ETF", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_index"),
    "IWM": MarketConfig(symbol="IWM", display_name="Russell 2000 ETF", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_index"),
    "DIA": MarketConfig(symbol="DIA", display_name="Dow Jones ETF", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_index"),

    # --- Sector ETFs ---
    "XLF": MarketConfig(symbol="XLF", display_name="Financials ETF", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_sector"),
    "XLE": MarketConfig(symbol="XLE", display_name="Energy ETF", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_sector"),
    "XLK": MarketConfig(symbol="XLK", display_name="Tech ETF", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_sector"),
    "XLV": MarketConfig(symbol="XLV", display_name="Healthcare ETF", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_sector"),

    # --- Mega-cap Tech ---
    "AAPL": MarketConfig(symbol="AAPL", display_name="Apple", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "MSFT": MarketConfig(symbol="MSFT", display_name="Microsoft", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "GOOGL": MarketConfig(symbol="GOOGL", display_name="Google", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "AMZN": MarketConfig(symbol="AMZN", display_name="Amazon", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "META": MarketConfig(symbol="META", display_name="Meta", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "TSLA": MarketConfig(symbol="TSLA", display_name="Tesla", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "NVDA": MarketConfig(symbol="NVDA", display_name="NVIDIA", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_semi"),

    # --- Semiconductors ---
    "AMD": MarketConfig(symbol="AMD", display_name="AMD", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_semi"),
    "AVGO": MarketConfig(symbol="AVGO", display_name="Broadcom", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_semi"),
    "TSM": MarketConfig(symbol="TSM", display_name="TSMC", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_semi"),
    "ARM": MarketConfig(symbol="ARM", display_name="ARM Holdings", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_semi"),
    "MRVL": MarketConfig(symbol="MRVL", display_name="Marvell", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_semi"),

    # --- High-vol individual stocks ---
    "COIN": MarketConfig(symbol="COIN", display_name="Coinbase", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_crypto_equity"),
    "MSTR": MarketConfig(symbol="MSTR", display_name="MicroStrategy", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_crypto_equity"),
    "PLTR": MarketConfig(symbol="PLTR", display_name="Palantir", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "SOFI": MarketConfig(symbol="SOFI", display_name="SoFi", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_fintech"),
    "SNOW": MarketConfig(symbol="SNOW", display_name="Snowflake", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "NET": MarketConfig(symbol="NET", display_name="Cloudflare", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "CRWD": MarketConfig(symbol="CRWD", display_name="CrowdStrike", category=MarketCategory.STOCKS, timeframes=["1d", "4h"], scan_interval_hours=24.0, correlation_group="us_tech"),
    "JPM": MarketConfig(symbol="JPM", display_name="JP Morgan", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_bank"),
    "GS": MarketConfig(symbol="GS", display_name="Goldman Sachs", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_bank"),
    "XOM": MarketConfig(symbol="XOM", display_name="Exxon Mobil", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_energy"),
    "LLY": MarketConfig(symbol="LLY", display_name="Eli Lilly", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_pharma"),
    "BA": MarketConfig(symbol="BA", display_name="Boeing", category=MarketCategory.STOCKS, timeframes=["1d"], scan_interval_hours=24.0, correlation_group="us_industrial"),
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
