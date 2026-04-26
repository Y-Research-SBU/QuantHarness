"""
Multi-market data fetcher using yfinance.
Fetches OHLCV data for all configured markets and timeframes.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from market_config import MARKETS, MarketConfig

try:
    from data_cache import get_cache
except Exception:  # pragma: no cover — cache is optional
    get_cache = None  # type: ignore

logger = logging.getLogger(__name__)

# yfinance interval mapping
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1wk",
}

# Max lookback periods per interval (yfinance limits)
MAX_LOOKBACK_DAYS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "4h": 730,
    "1d": 730,
    "1w": 730,
}

# Default number of bars to fetch
DEFAULT_BARS = {
    "1m": 200,
    "5m": 200,
    "15m": 200,
    "30m": 200,
    "1h": 200,
    "4h": 100,
    "1d": 100,
    "1w": 52,
}


def fetch_market_data(
    symbol: str,
    interval: str = "4h",
    bars: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single market.

    Reads from the shared :class:`OHLCVCache` when available (Redis-backed)
    and falls back to a direct yfinance call on miss / when Redis is down.

    Args:
        symbol: yfinance symbol (e.g., "BTC-USD")
        interval: Time interval (e.g., "4h", "1d")
        bars: Number of bars to fetch (overrides date range)
        start_date: Start of date range
        end_date: End of date range (defaults to now)
        use_cache: If False, bypass the cache entirely (read AND write).

    Returns:
        DataFrame with columns: Datetime, Open, High, Low, Close, Volume
    """
    yf_interval = INTERVAL_MAP.get(interval, interval)

    # Cache lookup. We only consult the cache when the caller didn't
    # specify a custom date window — those queries are not what the
    # daemon populates.
    cache = None
    cache_eligible = (
        use_cache and start_date is None and end_date is None and get_cache is not None
    )
    if cache_eligible:
        try:
            cache = get_cache()
            if cache.is_connected():
                cached = cache.get(symbol, interval, bars)
                if cached is not None and not cached.empty:
                    return cached
        except Exception as exc:
            logger.debug("cache read failed for %s/%s: %s", symbol, interval, exc)
    
    if end_date is None:
        end_date = datetime.now()
    
    if start_date is None:
        # Calculate start date based on desired bars
        num_bars = bars or DEFAULT_BARS.get(interval, 100)
        # Rough estimate: add buffer for weekends/holidays
        if interval in ("1d", "1w"):
            lookback_days = int(num_bars * 1.5)
        elif interval == "4h":
            lookback_days = int(num_bars * 4 / 24 * 1.5)
        elif interval == "1h":
            lookback_days = int(num_bars / 24 * 1.5)
        else:
            lookback_days = int(num_bars / (24 * 60 / _interval_minutes(interval)) * 1.5)
        
        max_days = MAX_LOOKBACK_DAYS.get(interval, 730)
        lookback_days = min(lookback_days, max_days)
        start_date = end_date - timedelta(days=lookback_days)
    
    try:
        df = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            interval=yf_interval,
            auto_adjust=True,
            prepost=False,
            progress=False,
        )
        
        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} ({interval})")
            return pd.DataFrame()
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        
        # Rename columns
        rename_map = {"Date": "Datetime", "index": "Datetime"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Ensure Datetime column exists
        if "Datetime" not in df.columns:
            # Try to find a datetime-like column
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df = df.rename(columns={col: "Datetime"})
                    break
        
        required = ["Datetime", "Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            logger.warning(f"Missing columns for {symbol}. Available: {list(df.columns)}")
            return pd.DataFrame()
        
        # Select and clean
        cols = required + (["Volume"] if "Volume" in df.columns else [])
        df = df[cols].copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Drop incomplete / NaN bars. yfinance occasionally returns a partial
        # "today" daily bar with NaN OHLCV (especially for crypto on weekends),
        # which silently corrupts every downstream consumer:
        #   - mark_to_market skips the symbol -> open positions show pnl=NULL
        #   - check_stops compares price <= stop with NaN, which is always
        #     False, so SL/TP never fire and positions get stuck OPEN.
        # Always drop rows where the Close is missing/non-finite so callers
        # only ever see complete bars.
        before = len(df)
        df = df[df["Close"].notna()].reset_index(drop=True)
        if len(df) < before:
            logger.debug(
                "%s/%s: dropped %d incomplete bar(s) with NaN Close",
                symbol, interval, before - len(df),
            )

        # Limit to requested bars
        if bars and len(df) > bars:
            df = df.tail(bars).reset_index(drop=True)

        # Populate the cache (best-effort; errors are swallowed).
        if cache_eligible and cache is not None and cache.is_connected():
            try:
                cache.set(symbol, interval, df, bars)
            except Exception as exc:
                logger.debug("cache write failed for %s/%s: %s", symbol, interval, exc)

        return df

    except Exception as e:
        logger.error(f"Error fetching {symbol} ({interval}): {e}")
        return pd.DataFrame()


def fetch_all_markets(
    symbols: Optional[List[str]] = None,
    bars: Optional[int] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch data for all configured markets and timeframes.
    
    Returns:
        Dict mapping symbol -> timeframe -> DataFrame
    """
    if symbols is None:
        symbols = list(MARKETS.keys())
    
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    for symbol in symbols:
        config = MARKETS.get(symbol)
        if not config:
            logger.warning(f"Unknown symbol: {symbol}")
            continue
        
        results[symbol] = {}
        for tf in config.timeframes:
            df = fetch_market_data(symbol, tf, bars=bars)
            if not df.empty:
                results[symbol][tf] = df
                logger.info(f"Fetched {len(df)} bars for {symbol} ({tf})")
            else:
                logger.warning(f"No data for {symbol} ({tf})")
    
    return results


def compute_correlation(
    symbol_a: str,
    symbol_b: str,
    period: int = 30,
    interval: str = "1d",
) -> float:
    """
    Compute rolling correlation between two assets.
    
    Returns:
        Pearson correlation coefficient (-1 to 1), or 0.0 on error.
    """
    try:
        df_a = fetch_market_data(symbol_a, interval, bars=period + 10)
        df_b = fetch_market_data(symbol_b, interval, bars=period + 10)
        
        if df_a.empty or df_b.empty:
            return 0.0
        
        # Calculate returns
        returns_a = df_a["Close"].pct_change().dropna().tail(period)
        returns_b = df_b["Close"].pct_change().dropna().tail(period)
        
        # Align lengths
        min_len = min(len(returns_a), len(returns_b))
        if min_len < 5:
            return 0.0
        
        returns_a = returns_a.tail(min_len).values
        returns_b = returns_b.tail(min_len).values
        
        corr = np.corrcoef(returns_a, returns_b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    except Exception as e:
        logger.error(f"Correlation error ({symbol_a}, {symbol_b}): {e}")
        return 0.0


def prepare_kline_dict(df: pd.DataFrame) -> dict:
    """
    Convert DataFrame to dict format expected by the existing agent pipeline.
    """
    result = {}
    for col in ["Datetime", "Open", "High", "Low", "Close"]:
        if col == "Datetime":
            result[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            result[col] = df[col].tolist()
    if "Volume" in df.columns:
        result["Volume"] = df["Volume"].tolist()
    return result


def _interval_minutes(interval: str) -> int:
    """Convert interval string to minutes."""
    mapping = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
    return mapping.get(interval, 60)
