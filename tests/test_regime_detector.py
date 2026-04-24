"""Tests for regime_detector (Level 4 of the self-improvement system)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detector import (
    REGIMES,
    RegimeDetector,
    compute_adx,
    compute_atr_pct,
    compute_bollinger_width,
    compute_sma_slope,
)


# ---------------------------------------------------------------------------
# Synthetic OHLCV builders
# ---------------------------------------------------------------------------


def _make_df(closes: np.ndarray, spread: float = 0.2) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "Open": closes - spread * 0.1,
            "High": closes + spread,
            "Low": closes - spread,
            "Close": closes,
            "Volume": np.ones(n) * 1000,
        }
    )


def _trending_up_df(n: int = 200) -> pd.DataFrame:
    prices = np.linspace(100.0, 200.0, n)
    return _make_df(prices, spread=0.3)


def _trending_down_df(n: int = 200) -> pd.DataFrame:
    prices = np.linspace(200.0, 100.0, n)
    return _make_df(prices, spread=0.3)


def _ranging_df(n: int = 200, base: float = 100.0, band: float = 1.0) -> pd.DataFrame:
    # Tight sinusoidal oscillation → low ADX, narrow BB.
    rng = np.random.default_rng(0)
    x = np.arange(n)
    noise = rng.normal(0, 0.05, size=n)
    prices = base + band * np.sin(x * 0.3) * 0.2 + noise
    return _make_df(prices, spread=0.05)


def _volatile_df(n: int = 200, base: float = 100.0) -> pd.DataFrame:
    # Large high/low excursions → wide ATR, wide BB, no clear direction.
    rng = np.random.default_rng(1)
    closes = base + rng.normal(0, 5.0, size=n).cumsum() * 0.1
    df = _make_df(closes, spread=0.5)
    df["High"] = closes + np.abs(rng.normal(0, 5.0, size=n))
    df["Low"] = closes - np.abs(rng.normal(0, 5.0, size=n))
    return df


# ---------------------------------------------------------------------------
# Individual feature tests
# ---------------------------------------------------------------------------


def test_compute_adx_on_trending_data_is_high():
    df = _trending_up_df()
    adx = compute_adx(df)
    assert adx > 25, f"trending ADX should be high, got {adx}"


def test_compute_adx_on_ranging_data_is_low():
    df = _ranging_df()
    adx = compute_adx(df)
    assert adx < 25, f"ranging ADX should be low, got {adx}"


def test_compute_adx_insufficient_data_returns_nan():
    df = _make_df(np.linspace(100, 110, 5))
    adx = compute_adx(df)
    assert np.isnan(adx)


def test_compute_atr_pct_positive():
    df = _trending_up_df()
    assert compute_atr_pct(df) > 0


def test_compute_bollinger_width_ranging_is_narrow():
    df = _ranging_df()
    width = compute_bollinger_width(df)
    assert width < 0.1, f"ranging BB width expected narrow, got {width}"


def test_compute_sma_slope_sign_matches_trend():
    up = compute_sma_slope(_trending_up_df())
    down = compute_sma_slope(_trending_down_df())
    assert up > 0
    assert down < 0


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


def test_detect_trending_up():
    detector = RegimeDetector()
    regime = detector.classify(_trending_up_df())
    assert regime == "trending_up"


def test_detect_trending_down():
    detector = RegimeDetector()
    regime = detector.classify(_trending_down_df())
    assert regime == "trending_down"


def test_detect_ranging():
    detector = RegimeDetector()
    regime = detector.classify(_ranging_df())
    assert regime == "ranging"


def test_detect_volatile():
    detector = RegimeDetector()
    regime = detector.classify(_volatile_df())
    assert regime in {"volatile", "ranging"}, f"unexpected regime {regime}"
    # Specifically for volatile synthetic data, we expect at least wide ATR%
    feats = detector.get_regime_features(_volatile_df())
    assert feats["atr_pct"] > 0.01 or feats["bb_width"] > 0.05


def test_classify_with_insufficient_data_returns_unknown():
    detector = RegimeDetector()
    # Only a handful of bars — less than any indicator period.
    tiny = _make_df(np.linspace(100, 102, 5))
    assert detector.classify(tiny) == "unknown"


def test_classify_with_none_returns_unknown():
    detector = RegimeDetector()
    assert detector.classify(None) == "unknown"


def test_regime_features_complete():
    detector = RegimeDetector()
    feats = detector.get_regime_features(_trending_up_df())
    assert set(feats.keys()) == {"adx", "atr_pct", "bb_width", "sma_slope"}
    for v in feats.values():
        assert isinstance(v, float)


def test_classify_with_features_returns_both():
    detector = RegimeDetector()
    out = detector.classify_with_features(_trending_up_df())
    assert "regime" in out and "features" in out
    assert out["regime"] in REGIMES


def test_classify_with_features_handles_empty():
    detector = RegimeDetector()
    out = detector.classify_with_features(_make_df(np.array([100.0, 101.0])))
    assert out["regime"] == "unknown"


def test_regimes_constant_has_four_members():
    assert set(REGIMES) == {"trending_up", "trending_down", "ranging", "volatile"}


def test_lowercase_columns_supported():
    """Detector should work even if columns are 'close' instead of 'Close'."""
    df = _trending_up_df()
    df = df.rename(columns={c: c.lower() for c in df.columns if c != "Datetime"})
    detector = RegimeDetector()
    regime = detector.classify(df)
    assert regime in REGIMES


def _volatile_trending_up_df(n: int = 200, base: float = 100.0) -> pd.DataFrame:
    """High volatility (wide ATR/BB) but clear upward slope — should be
    classified as 'trending_up' not 'volatile' after the strong-slope fix."""
    rng = np.random.default_rng(42)
    # Strong upward drift + large noise
    drift = np.linspace(0, 60, n)
    noise = rng.normal(0, 3.0, size=n)
    closes = base + drift + noise
    # Wide spread to ensure high ATR
    return _make_df(closes, spread=4.0)


def _volatile_trending_down_df(n: int = 200, base: float = 200.0) -> pd.DataFrame:
    """High volatility but clear downward slope."""
    rng = np.random.default_rng(42)
    drift = np.linspace(0, -80, n)  # steeper drop for clear slope signal
    noise = rng.normal(0, 3.0, size=n)
    closes = base + drift + noise
    return _make_df(closes, spread=4.0)


def test_volatile_but_strongly_trending_up_is_trending():
    """When an asset is volatile but has a strong upward SMA slope (>= 2x
    UPTREND_SLOPE), classify as trending_up to prevent counter-trend shorts."""
    detector = RegimeDetector()
    df = _volatile_trending_up_df()
    regime = detector.classify(df)
    assert regime == "trending_up", (
        f"Expected trending_up for volatile asset with strong upward slope, got {regime}"
    )


def test_volatile_but_strongly_trending_down_is_trending():
    """Same as above but downward."""
    detector = RegimeDetector()
    df = _volatile_trending_down_df()
    regime = detector.classify(df)
    assert regime == "trending_down", (
        f"Expected trending_down for volatile asset with strong downward slope, got {regime}"
    )
