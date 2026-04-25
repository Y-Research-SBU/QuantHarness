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
# The dashboard module instantiates a module-level ``app = create_app()`` for
# gunicorn. In tests we never want that to spin up the Binance WS / yfinance
# background threads — they add noise and hammer external endpoints.
os.environ.setdefault("DASHBOARD_DISABLE_PRICE_FEED", "1")


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


# ─────────────────────────────────────────────────────────────
# Trading-engine fixtures used by integration/regression/E2E tests
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def engine(tmp_db_path):
    """Fresh PaperTradingEngine on an isolated temp SQLite database.

    Post-close cooldown is disabled here so tests that rapidly open-and-close
    the same symbol (streaks, invariant checks, equity sequences) still run.
    Tests that specifically exercise cooldown behavior use the
    ``engine_with_cooldown`` fixture below.
    """
    from paper_trading import PaperTradingEngine
    engine = PaperTradingEngine(db_path=tmp_db_path)
    engine.COOLDOWN_MINUTES = 0
    engine.MIN_HOLD_MINUTES = 0
    return engine


@pytest.fixture
def engine_with_cooldown(tmp_db_path):
    """PaperTradingEngine with the production-default 30-min cooldown active."""
    from paper_trading import PaperTradingEngine
    return PaperTradingEngine(db_path=tmp_db_path)


@pytest.fixture
def make_signal():
    """Factory for Signal objects with sensible defaults."""
    from market_config import StrategyType
    from strategies import Signal

    def _make(
        symbol: str = "BTC-USD",
        direction: str = "LONG",
        entry: float = 100.0,
        stop: float = 95.0,
        tp: float = 115.0,
        timeframe: str = "1h",
        strategy: StrategyType = StrategyType.MOMENTUM,
        strength: float = 0.8,
    ) -> Signal:
        return Signal(
            direction=direction,
            strength=strength,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            risk_reward_ratio=2.0,
            reasoning="test signal",
            metadata={},
        )

    return _make


@pytest.fixture
def make_position():
    """Factory for PositionSizeResult with sensible defaults."""
    from position_sizing import PositionSizeResult

    def _make(
        size: float = 1000.0,
        qty: float = 10.0,
        stop: float = 95.0,
        tp: float = 115.0,
    ) -> PositionSizeResult:
        return PositionSizeResult(
            position_size_usd=size,
            quantity=qty,
            risk_per_trade_usd=size * 0.02,
            risk_pct=0.02,
            kelly_fraction=0.2,
            half_kelly=0.1,
            stop_loss=stop,
            take_profit=tp,
            reason="test position",
        )

    return _make


@pytest.fixture
def open_trade(engine, make_signal, make_position):
    """Open a single BTC-USD LONG trade and return (trade_id, engine)."""
    signal = make_signal()
    position = make_position()
    trade_id = engine.execute_trade(signal, position)
    assert trade_id is not None
    return trade_id, engine


@pytest.fixture
def dashboard_app(tmp_db_path, tmp_path):
    """Flask test client for the dashboard pointed at a temp DB + backtest dir.

    The DB schema is initialized so the dashboard can query empty tables
    without raising OperationalError.
    """
    import dashboard
    from db_schema import init_db
    init_db(tmp_db_path)

    backtest_dir = tmp_path / "backtest_results"
    backtest_dir.mkdir()
    app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(backtest_dir))
    app.config["TESTING"] = True
    return app


@pytest.fixture
def dashboard_client(dashboard_app):
    return dashboard_app.test_client()
