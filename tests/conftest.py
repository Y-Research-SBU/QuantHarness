"""Shared pytest fixtures for QuantAgent tests."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a stub API key so modules that validate at import time don't crash.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")


@pytest.fixture
def tmp_db_path() -> Generator[str, None, None]:
    """Provide a fresh temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield path
    finally:
        for suffix in ("", "-wal", "-shm"):
            p = path + suffix
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with 100 bars."""
    np.random.seed(42)
    n = 100
    base = 100.0
    prices = base + np.cumsum(np.random.randn(n) * 0.5)
    highs = prices + np.abs(np.random.randn(n) * 0.3)
    lows = prices - np.abs(np.random.randn(n) * 0.3)
    opens = prices + np.random.randn(n) * 0.1
    volumes = np.random.randint(1000, 10000, size=n)
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h")

    df = pd.DataFrame({
        "Datetime": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": volumes,
    })
    df.attrs["symbol"] = "TEST-USD"
    df.attrs["timeframe"] = "1h"
    return df


@pytest.fixture
def uptrend_ohlcv() -> pd.DataFrame:
    """Synthetic data with a strong uptrend for momentum tests."""
    n = 100
    prices = np.linspace(100, 130, n)
    df = pd.DataFrame({
        "Datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h"),
        "Open": prices - 0.1,
        "High": prices + 0.2,
        "Low": prices - 0.2,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, size=n),
    })
    df.attrs["symbol"] = "UP-USD"
    df.attrs["timeframe"] = "1h"
    return df


@pytest.fixture
def downtrend_ohlcv() -> pd.DataFrame:
    """Synthetic data with a strong downtrend."""
    n = 100
    prices = np.linspace(130, 100, n)
    df = pd.DataFrame({
        "Datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h"),
        "Open": prices + 0.1,
        "High": prices + 0.2,
        "Low": prices - 0.2,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, size=n),
    })
    df.attrs["symbol"] = "DN-USD"
    df.attrs["timeframe"] = "1h"
    return df


@pytest.fixture
def overbought_ohlcv() -> pd.DataFrame:
    """Synthetic data that pushes RSI well above 70 for mean-reversion tests."""
    n = 60
    # Sharp ramp up with barely any pullbacks keeps avg_loss tiny → RSI very high.
    prices = np.linspace(50, 200, n) + np.random.uniform(-0.05, 0.2, n)
    df = pd.DataFrame({
        "Datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h"),
        "Open": prices - 0.1,
        "High": prices + 0.5,
        "Low": prices - 0.1,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, size=n),
    })
    df.attrs["symbol"] = "OB-USD"
    df.attrs["timeframe"] = "1h"
    return df


@pytest.fixture
def oversold_ohlcv() -> pd.DataFrame:
    """Synthetic data that pushes RSI well below 30 for mean-reversion tests."""
    n = 60
    prices = np.linspace(200, 50, n) - np.random.uniform(-0.05, 0.2, n)
    df = pd.DataFrame({
        "Datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h"),
        "Open": prices + 0.1,
        "High": prices + 0.1,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, size=n),
    })
    df.attrs["symbol"] = "OS-USD"
    df.attrs["timeframe"] = "1h"
    return df
