"""
Long-running data daemon — populates the Redis OHLCV cache.

Runs in a single process: every ``poll_seconds`` it walks every (symbol,
interval) pair in MARKETS, fetches the latest bars from yfinance, and
writes them to the cache. The next round of scanner reads then hits
Redis instead of yfinance.

Usage:
    python3 data_daemon.py [--poll-seconds 30] [--once]

If Redis isn't reachable the daemon logs a warning and keeps polling
yfinance directly (so the cache transparently re-attaches when Redis
comes back up).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from data_cache import OHLCVCache, TTL_BY_INTERVAL, get_cache, reset_cache
from data_fetcher import fetch_market_data
from market_config import MARKETS

logger = logging.getLogger(__name__)


# Per-interval refresh cadences (s). Faster than half the TTL so the
# cache rarely returns stale bars.
DEFAULT_POLL_BY_INTERVAL: Dict[str, int] = {
    "5m": 20,
    "15m": 60,
    "1h": 5 * 60,
    "4h": 15 * 60,
    "1d": 2 * 3600,
}


class DataDaemon:
    """Polls every (symbol, interval) and pushes fresh bars into the cache."""

    def __init__(
        self,
        cache: Optional[OHLCVCache] = None,
        markets=None,
        poll_seconds: int = 30,
        bars_per_fetch: int = 200,
    ):
        self.cache = cache if cache is not None else get_cache()
        self.markets = markets if markets is not None else MARKETS
        self.poll_seconds = poll_seconds
        self.bars_per_fetch = bars_per_fetch
        self._stop = False
        # last_refresh[(symbol, interval)] = monotonic seconds
        self._last_refresh: Dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def request_stop(self, *_args) -> None:
        self._stop = True

    def install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)
        except (ValueError, OSError):
            pass

    # ------------------------------------------------------------------
    # Work
    # ------------------------------------------------------------------

    def cells(self) -> List[tuple]:
        out: List[tuple] = []
        for symbol, cfg in self.markets.items():
            for tf in cfg.timeframes:
                out.append((symbol, tf))
        return out

    def _due(self, key: tuple, interval: str, now: float) -> bool:
        cadence = DEFAULT_POLL_BY_INTERVAL.get(interval, max(60, self.poll_seconds))
        last = self._last_refresh.get(key)
        if last is None:
            return True
        return (now - last) >= cadence

    def refresh_one(self, symbol: str, interval: str) -> bool:
        """Fetch one bar set and push it into the cache. Returns True on success."""
        try:
            df = fetch_market_data(
                symbol, interval, bars=self.bars_per_fetch, use_cache=False,
            )
        except Exception as exc:
            logger.warning("daemon fetch failed for %s/%s: %s", symbol, interval, exc)
            return False
        if df is None or df.empty:
            return False
        if self.cache is None or not self.cache.is_connected():
            return False
        return bool(self.cache.set(symbol, interval, df, self.bars_per_fetch))

    def run_once(self) -> Dict[str, int]:
        """Walk every cell once. Returns {success, miss, skipped}."""
        result = {"success": 0, "miss": 0, "skipped": 0}
        now = time.monotonic()
        for symbol, interval in self.cells():
            key = (symbol, interval)
            if not self._due(key, interval, now):
                result["skipped"] += 1
                continue
            ok = self.refresh_one(symbol, interval)
            if ok:
                result["success"] += 1
                self._last_refresh[key] = now
            else:
                result["miss"] += 1
        return result

    def run(self) -> None:
        logger.info(
            "data daemon starting \u2014 cells=%d, poll=%ds, redis=%s",
            len(self.cells()), self.poll_seconds, self.cache.is_connected() if self.cache else False,
        )
        while not self._stop:
            try:
                stats = self.run_once()
                logger.info("daemon tick: %s", stats)
            except Exception as exc:
                logger.exception("daemon tick failed: %s", exc)
            for _ in range(self.poll_seconds):
                if self._stop:
                    break
                time.sleep(1)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="QuantAgent OHLCV cache daemon")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--bars", type=int, default=200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    reset_cache()
    daemon = DataDaemon(poll_seconds=args.poll_seconds, bars_per_fetch=args.bars)
    daemon.install_signal_handlers()
    if args.once:
        daemon.run_once()
    else:
        daemon.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
