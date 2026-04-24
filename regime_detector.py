"""
Market regime classification (Level 4 of the self-improvement system).

Classifies an OHLCV window into one of:
    - trending_up   (strong positive trend)
    - trending_down (strong negative trend)
    - ranging       (low ADX, narrow Bollinger bands)
    - volatile     (wide ATR / wide Bollinger bands without clear trend)
    - unknown       (insufficient data)

Features used:
    - ADX (14)              — trend strength
    - ATR / close %         — volatility as a fraction of price
    - Bollinger width       — (upper - lower) / middle
    - SMA slope             — short-term direction
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


REGIMES: List[str] = ["trending_up", "trending_down", "ranging", "volatile"]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] supporting both 'Close' and 'close' conventions."""
    if name in df.columns:
        return df[name]
    alt = name.lower() if name[0].isupper() else name.capitalize()
    if alt in df.columns:
        return df[alt]
    raise KeyError(f"Column {name!r} (or {alt!r}) not found in DataFrame")


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ADX value (0-100).

    Implements the classic Wilder ADX. Returns ``np.nan`` if not enough data.
    """
    if len(df) < period * 2 + 1:
        return float("nan")

    high = _col(df, "High").astype(float).values
    low = _col(df, "Low").astype(float).values
    close = _col(df, "Close").astype(float).values

    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    tr = np.zeros(len(df))

    for i in range(1, len(df)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder smoothing
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean().values / np.where(atr == 0, 1e-9, atr)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean().values / np.where(atr == 0, 1e-9, atr)
    denom = np.where((plus_di + minus_di) == 0, 1e-9, (plus_di + minus_di))
    dx = 100 * np.abs(plus_di - minus_di) / denom
    adx = pd.Series(dx).ewm(alpha=1 / period, adjust=False).mean().values

    val = float(adx[-1])
    if np.isnan(val):
        return float("nan")
    return val


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """Return ATR as a fraction of the latest close (e.g. 0.02 = 2%)."""
    if len(df) < period + 1:
        return float("nan")
    high = _col(df, "High").astype(float).values
    low = _col(df, "Low").astype(float).values
    close = _col(df, "Close").astype(float).values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values[-1]
    last_close = close[-1]
    if last_close == 0:
        return float("nan")
    return float(atr / last_close)


def compute_bollinger_width(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> float:
    """Return Bollinger band width as fraction of middle band."""
    if len(df) < period:
        return float("nan")
    close = _col(df, "Close").astype(float)
    ma = close.rolling(period).mean().iloc[-1]
    sd = close.rolling(period).std().iloc[-1]
    if ma == 0 or np.isnan(ma) or np.isnan(sd):
        return float("nan")
    upper = ma + mult * sd
    lower = ma - mult * sd
    return float((upper - lower) / ma)


def compute_sma_slope(df: pd.DataFrame, period: int = 20) -> float:
    """Return normalized slope of the SMA over the last ``period`` bars.

    Positive = trending up, negative = trending down. Normalized by the SMA
    value so it's comparable across price levels.
    """
    if len(df) < period + 1:
        return float("nan")
    close = _col(df, "Close").astype(float)
    sma = close.rolling(period).mean().dropna()
    if len(sma) < 2:
        return float("nan")
    # Slope over last ``period`` SMA values (simple linear regression)
    recent = sma.iloc[-period:].values if len(sma) >= period else sma.values
    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent, 1)
    base = recent[-1] if recent[-1] != 0 else 1e-9
    return float(slope / base)


class RegimeDetector:
    """Classifies market state from OHLCV data."""

    REGIMES: List[str] = list(REGIMES)

    # Thresholds — tuned to common values in trading literature.
    TRENDING_ADX = 25.0
    RANGING_ADX = 20.0
    VOLATILE_ATR_PCT = 0.04         # >4% ATR/price = volatile
    VOLATILE_BB_WIDTH = 0.12        # >12% band width relative to mid
    UPTREND_SLOPE = 0.001           # +0.1% per bar on SMA (normalized)

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        sma_period: int = 20,
    ) -> None:
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.sma_period = sma_period

    def get_regime_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Return every regime feature as a float (may be NaN if insufficient data)."""
        return {
            "adx": compute_adx(df, self.adx_period),
            "atr_pct": compute_atr_pct(df, self.atr_period),
            "bb_width": compute_bollinger_width(df, self.bb_period),
            "sma_slope": compute_sma_slope(df, self.sma_period),
        }

    def classify(self, df: pd.DataFrame) -> str:
        """Classify a window into one of REGIMES or 'unknown'."""
        if df is None or len(df) < max(self.adx_period * 2 + 1, self.bb_period, self.sma_period + 1):
            return "unknown"

        feats = self.get_regime_features(df)
        adx = feats["adx"]
        atr_pct = feats["atr_pct"]
        bb_width = feats["bb_width"]
        slope = feats["sma_slope"]

        if any(np.isnan(v) for v in (adx, atr_pct, bb_width, slope)):
            return "unknown"

        # Volatile takes priority: big ranges without clear direction.
        volatile = atr_pct >= self.VOLATILE_ATR_PCT or bb_width >= self.VOLATILE_BB_WIDTH
        if volatile and adx < self.TRENDING_ADX:
            return "volatile"

        if adx >= self.TRENDING_ADX:
            if slope >= self.UPTREND_SLOPE:
                return "trending_up"
            if slope <= -self.UPTREND_SLOPE:
                return "trending_down"
            # High ADX but neutral slope — treat as volatile.
            return "volatile"

        if adx < self.RANGING_ADX:
            return "ranging"

        # Middle zone: fall back to volatility or slope signal.
        if volatile:
            return "volatile"
        if slope >= self.UPTREND_SLOPE:
            return "trending_up"
        if slope <= -self.UPTREND_SLOPE:
            return "trending_down"
        return "ranging"

    def classify_with_features(self, df: pd.DataFrame) -> Dict[str, object]:
        """Return {'regime': str, 'features': dict} — handy for logging."""
        feats = self.get_regime_features(df) if df is not None and len(df) > 0 else {}
        return {"regime": self.classify(df), "features": feats}
