"""
Redis-backed OHLCV cache.

Wraps a Redis client (or a no-op fallback when Redis isn't available) and
serialises pandas frames to compact JSON. The data daemon populates the
cache; readers (data_fetcher.py) consult it before falling back to direct
yfinance calls.

Cache TTLs are interval-aware (5m=30s, 15m=2m, 1h=10m, 4h=30m, 1d=4h)
so we never serve a bar more than ~half a candle stale.

Configuration via env:
  REDIS_URL — defaults to redis://localhost:6379/0
  QUANTAGENT_CACHE_DISABLED=1 — force pass-through (useful in tests)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Per-interval TTL in seconds. Intervals not listed default to ``DEFAULT_TTL``.
TTL_BY_INTERVAL: Dict[str, int] = {
    "5m": 30,
    "15m": 120,
    "1h": 600,
    "4h": 1800,
    "1d": 4 * 3600,
}
DEFAULT_TTL = 300


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def cache_disabled() -> bool:
    return os.environ.get("QUANTAGENT_CACHE_DISABLED") == "1"


def _make_key(symbol: str, interval: str, bars: Optional[int]) -> str:
    if bars:
        return f"qa:ohlcv:{symbol}:{interval}:{bars}"
    return f"qa:ohlcv:{symbol}:{interval}"


def _df_to_json(df: pd.DataFrame) -> str:
    """Serialise a DataFrame to compact, timezone-stable JSON."""
    out = df.copy()
    if "Datetime" in out.columns:
        out["Datetime"] = pd.to_datetime(out["Datetime"]).astype(str)
    return json.dumps({
        "columns": list(out.columns),
        "rows": out.values.tolist(),
    })


def _json_to_df(payload: str) -> pd.DataFrame:
    data = json.loads(payload)
    df = pd.DataFrame(data["rows"], columns=data["columns"])
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df


class _NullCache:
    """Stub used when Redis isn't reachable. All ops are no-ops."""

    def get(self, *_args, **_kwargs) -> Optional[pd.DataFrame]:
        return None

    def set(self, *_args, **_kwargs) -> bool:
        return False

    def ping(self) -> bool:
        return False

    def ttl_for(self, interval: str) -> int:
        return TTL_BY_INTERVAL.get(interval, DEFAULT_TTL)


class OHLCVCache:
    """Get/set OHLCV frames keyed by (symbol, interval[, bars])."""

    def __init__(self, client=None, namespace_prefix: str = "qa:ohlcv:"):
        self._client = client
        self._prefix = namespace_prefix
        self._connected = client is not None

    @classmethod
    def from_env(cls) -> "OHLCVCache":
        if cache_disabled():
            logger.info("OHLCVCache disabled via QUANTAGENT_CACHE_DISABLED=1")
            return cls(client=None)
        try:
            import redis  # type: ignore
        except ImportError:
            logger.warning("redis library not installed — cache disabled")
            return cls(client=None)
        try:
            client = redis.Redis.from_url(_redis_url(), decode_responses=True,
                                          socket_connect_timeout=1.0)
            client.ping()
            return cls(client=client)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — cache disabled", exc)
            return cls(client=None)

    def is_connected(self) -> bool:
        return bool(self._connected)

    def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    @staticmethod
    def ttl_for(interval: str) -> int:
        return TTL_BY_INTERVAL.get(interval, DEFAULT_TTL)

    # ------------------------------------------------------------------
    # Get / Set
    # ------------------------------------------------------------------

    def get(self, symbol: str, interval: str, bars: Optional[int] = None) -> Optional[pd.DataFrame]:
        if self._client is None:
            return None
        try:
            payload = self._client.get(_make_key(symbol, interval, bars))
            if not payload:
                return None
            return _json_to_df(payload)
        except Exception as exc:
            logger.debug("cache get failed for %s/%s: %s", symbol, interval, exc)
            return None

    def set(self, symbol: str, interval: str, df: pd.DataFrame, bars: Optional[int] = None) -> bool:
        if self._client is None:
            return False
        if df is None or df.empty:
            return False
        try:
            payload = _df_to_json(df)
            ttl = self.ttl_for(interval)
            self._client.setex(_make_key(symbol, interval, bars), ttl, payload)
            return True
        except Exception as exc:
            logger.debug("cache set failed for %s/%s: %s", symbol, interval, exc)
            return False


# ── Module-level singleton (lazy) ──────────────────────────────────────
_SHARED_CACHE: Optional[OHLCVCache] = None


def get_cache() -> OHLCVCache:
    """Return the lazy shared OHLCVCache. Re-resolved if env changed."""
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = OHLCVCache.from_env()
    return _SHARED_CACHE


def reset_cache() -> None:
    """Drop the shared cache (testing helper)."""
    global _SHARED_CACHE
    _SHARED_CACHE = None


__all__ = [
    "OHLCVCache",
    "TTL_BY_INTERVAL",
    "DEFAULT_TTL",
    "get_cache",
    "reset_cache",
    "cache_disabled",
]
