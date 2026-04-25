"""Tests for the OHLCV cache daemon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd
import pytest

import data_daemon
from data_cache import OHLCVCache


@dataclass
class _FakeMarket:
    timeframes: List[str] = field(default_factory=lambda: ["1h", "1d"])


class _FakeClient:
    def __init__(self):
        self.store = {}

    def setex(self, key, ttl, value):
        self.store[key] = (ttl, value)

    def get(self, key):
        v = self.store.get(key)
        return v[1] if v else None

    def ping(self):
        return True


@pytest.fixture
def cache():
    return OHLCVCache(client=_FakeClient())


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Datetime": pd.to_datetime(["2026-04-25 12:00:00"]),
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [500],
    })


@pytest.fixture
def fake_markets():
    return {
        "BTC-USD": _FakeMarket(timeframes=["1h", "4h"]),
        "AAPL": _FakeMarket(timeframes=["1d"]),
    }


def test_cells_iterates_every_symbol_timeframe(cache, fake_markets):
    d = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    cells = d.cells()
    assert len(cells) == 3
    assert ("BTC-USD", "1h") in cells
    assert ("BTC-USD", "4h") in cells
    assert ("AAPL", "1d") in cells


def test_due_returns_true_first_time(cache, fake_markets):
    d = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    assert d._due(("BTC-USD", "1h"), "1h", 0.0) is True


def test_due_respects_cadence(cache, fake_markets):
    d = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    d._last_refresh[("BTC-USD", "1h")] = 0.0
    assert d._due(("BTC-USD", "1h"), "1h", 30.0) is False
    # 1h cadence is 5*60, so well after
    assert d._due(("BTC-USD", "1h"), "1h", 10000.0) is True


def test_refresh_one_writes_to_cache(monkeypatch, cache, fake_markets, sample_df):
    monkeypatch.setattr(data_daemon, "fetch_market_data", lambda *a, **kw: sample_df)
    d = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    assert d.refresh_one("BTC-USD", "1h") is True
    out = cache.get("BTC-USD", "1h", 200)
    assert out is not None
    assert len(out) == 1


def test_refresh_one_returns_false_on_empty(monkeypatch, cache, fake_markets):
    monkeypatch.setattr(data_daemon, "fetch_market_data", lambda *a, **kw: pd.DataFrame())
    d = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    assert d.refresh_one("BTC-USD", "1h") is False


def test_refresh_one_handles_fetch_exceptions(monkeypatch, cache, fake_markets):
    def boom(*a, **kw):
        raise RuntimeError("yfinance down")
    monkeypatch.setattr(data_daemon, "fetch_market_data", boom)
    d = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    assert d.refresh_one("BTC-USD", "1h") is False


def test_refresh_one_no_op_when_cache_disconnected(monkeypatch, fake_markets, sample_df):
    monkeypatch.setattr(data_daemon, "fetch_market_data", lambda *a, **kw: sample_df)
    null = OHLCVCache(client=None)
    d = data_daemon.DataDaemon(cache=null, markets=fake_markets)
    assert d.refresh_one("BTC-USD", "1h") is False


def test_run_once_walks_all_cells(monkeypatch, cache, fake_markets, sample_df):
    monkeypatch.setattr(data_daemon, "fetch_market_data", lambda *a, **kw: sample_df)
    d = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    stats = d.run_once()
    assert stats["success"] == 3
    assert stats["miss"] == 0


def test_run_once_skips_recently_refreshed(monkeypatch, cache, fake_markets, sample_df):
    monkeypatch.setattr(data_daemon, "fetch_market_data", lambda *a, **kw: sample_df)
    d = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    d.run_once()
    # Immediate second pass should be all skipped (cadences are minutes long).
    stats2 = d.run_once()
    assert stats2["success"] == 0
    assert stats2["skipped"] == 3
