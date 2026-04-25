"""Tests for data_cache.OHLCVCache (Redis-backed OHLCV cache)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

import data_cache
from data_cache import OHLCVCache, TTL_BY_INTERVAL, DEFAULT_TTL


# ──────────────────────────────────────────────────────────────────────
# Fake redis client used as the test double
# ──────────────────────────────────────────────────────────────────────


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttls = {}
        self._fail_get = False
        self._fail_set = False
        self._fail_ping = False

    def setex(self, key, ttl, value):
        if self._fail_set:
            raise RuntimeError("boom")
        self.store[key] = value
        self.ttls[key] = ttl
        return True

    def get(self, key):
        if self._fail_get:
            raise RuntimeError("boom")
        return self.store.get(key)

    def ping(self):
        if self._fail_ping:
            raise RuntimeError("boom")
        return True


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Datetime": pd.to_datetime(["2026-04-25 12:00:00", "2026-04-25 13:00:00"]),
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.5, 100.5],
        "Close": [101.0, 102.5],
        "Volume": [1000, 1100],
    })


# ──────────────────────────────────────────────────────────────────────
# Connection state
# ──────────────────────────────────────────────────────────────────────


def test_ohlcv_cache_disconnected_when_no_client():
    c = OHLCVCache(client=None)
    assert c.is_connected() is False
    assert c.ping() is False


def test_ohlcv_cache_connected_when_client_set():
    c = OHLCVCache(client=FakeRedis())
    assert c.is_connected() is True
    assert c.ping() is True


def test_ping_handles_client_failure():
    fake = FakeRedis()
    fake._fail_ping = True
    c = OHLCVCache(client=fake)
    assert c.ping() is False


# ──────────────────────────────────────────────────────────────────────
# TTL table
# ──────────────────────────────────────────────────────────────────────


def test_ttl_for_known_intervals():
    assert OHLCVCache.ttl_for("5m") == 30
    assert OHLCVCache.ttl_for("15m") == 120
    assert OHLCVCache.ttl_for("1h") == 600
    assert OHLCVCache.ttl_for("4h") == 1800
    assert OHLCVCache.ttl_for("1d") == 4 * 3600


def test_ttl_for_unknown_interval_uses_default():
    assert OHLCVCache.ttl_for("bogus") == DEFAULT_TTL


# ──────────────────────────────────────────────────────────────────────
# get / set roundtrip
# ──────────────────────────────────────────────────────────────────────


def test_set_then_get_roundtrips_dataframe(sample_df):
    fake = FakeRedis()
    c = OHLCVCache(client=fake)
    assert c.set("BTC-USD", "1h", sample_df, bars=2) is True
    out = c.get("BTC-USD", "1h", bars=2)
    assert out is not None
    assert list(out.columns) == list(sample_df.columns)
    assert len(out) == 2
    assert float(out["Close"].iloc[0]) == 101.0


def test_set_uses_correct_ttl(sample_df):
    fake = FakeRedis()
    c = OHLCVCache(client=fake)
    c.set("BTC-USD", "5m", sample_df)
    key = list(fake.store.keys())[0]
    assert fake.ttls[key] == 30
    fake.store.clear()
    fake.ttls.clear()
    c.set("BTC-USD", "1d", sample_df)
    key = list(fake.store.keys())[0]
    assert fake.ttls[key] == 4 * 3600


def test_get_returns_none_on_miss():
    c = OHLCVCache(client=FakeRedis())
    assert c.get("NOPE", "1h") is None


def test_set_skipped_when_disconnected(sample_df):
    c = OHLCVCache(client=None)
    assert c.set("BTC-USD", "1h", sample_df) is False


def test_set_skipped_for_empty_df():
    fake = FakeRedis()
    c = OHLCVCache(client=fake)
    assert c.set("BTC-USD", "1h", pd.DataFrame()) is False
    assert fake.store == {}


def test_set_swallows_client_errors(sample_df):
    fake = FakeRedis()
    fake._fail_set = True
    c = OHLCVCache(client=fake)
    assert c.set("BTC-USD", "1h", sample_df) is False


def test_get_swallows_client_errors():
    fake = FakeRedis()
    fake._fail_get = True
    c = OHLCVCache(client=fake)
    assert c.get("BTC-USD", "1h") is None


def test_get_with_and_without_bars_uses_distinct_keys(sample_df):
    fake = FakeRedis()
    c = OHLCVCache(client=fake)
    c.set("BTC-USD", "1h", sample_df, bars=200)
    c.set("BTC-USD", "1h", sample_df)
    assert len(fake.store) == 2


# ──────────────────────────────────────────────────────────────────────
# from_env() respects QUANTAGENT_CACHE_DISABLED
# ──────────────────────────────────────────────────────────────────────


def test_from_env_returns_disconnected_when_disabled(monkeypatch):
    monkeypatch.setenv("QUANTAGENT_CACHE_DISABLED", "1")
    c = OHLCVCache.from_env()
    assert c.is_connected() is False


def test_get_cache_singleton(monkeypatch):
    monkeypatch.setenv("QUANTAGENT_CACHE_DISABLED", "1")
    data_cache.reset_cache()
    a = data_cache.get_cache()
    b = data_cache.get_cache()
    assert a is b


def test_reset_cache_drops_singleton(monkeypatch):
    monkeypatch.setenv("QUANTAGENT_CACHE_DISABLED", "1")
    data_cache.reset_cache()
    a = data_cache.get_cache()
    data_cache.reset_cache()
    b = data_cache.get_cache()
    assert a is not b
