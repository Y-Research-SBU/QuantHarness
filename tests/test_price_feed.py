"""Tests for the real-time price feed (price_feed.py).

We exercise the pure logic — symbol mapping, P&L helpers, the thread-safe
cache, and the stock-polling glue — without actually talking to Binance or
yfinance. The Binance reconnect thread is explicitly *not* started in these
tests; instead we drive the internal record path directly.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from price_feed import (
    PriceFeed,
    build_binance_mapping,
    compute_unrealized_pnl,
    quant_to_binance,
    split_symbols,
    _is_us_market_open,
)


# ──────────────────────── symbol mapping ────────────────────────


@pytest.mark.parametrize(
    "quant,binance",
    [
        ("BTC-USD", "btcusdt"),
        ("ETH-USD", "ethusdt"),
        ("SOL-USD", "solusdt"),
        ("SHIB-USD", "shibusdt"),
        ("RNDR-USD", "renderusdt"),  # override
        ("FTM-USD", "ftmusdt"),      # override
    ],
)
def test_quant_to_binance_maps_crypto(quant, binance):
    assert quant_to_binance(quant) == binance


@pytest.mark.parametrize("non_crypto", ["SPY", "QQQ", "AAPL", "GC=F", "EURUSD=X", ""])
def test_quant_to_binance_returns_none_for_non_crypto(non_crypto):
    assert quant_to_binance(non_crypto) is None


def test_quant_to_binance_handles_none():
    assert quant_to_binance(None) is None  # type: ignore[arg-type]


def test_build_binance_mapping_bidirectional():
    q2b, b2q = build_binance_mapping(["BTC-USD", "SPY", "ETH-USD"])
    assert q2b == {"BTC-USD": "btcusdt", "ETH-USD": "ethusdt"}
    assert b2q == {"btcusdt": "BTC-USD", "ethusdt": "ETH-USD"}


def test_split_symbols_partitions_by_class():
    crypto, other = split_symbols(
        ["BTC-USD", "SPY", "AAPL", "SOL-USD", "EURUSD=X", "GC=F"]
    )
    assert crypto == ["BTC-USD", "SOL-USD"]
    assert other == ["SPY", "AAPL", "EURUSD=X", "GC=F"]


# ──────────────────────── P&L helper ────────────────────────


def test_compute_unrealized_pnl_long():
    # +$500 on a 10-share LONG bought at 100, now at 150.
    assert compute_unrealized_pnl(100, 150, 10, "LONG") == pytest.approx(500.0)


def test_compute_unrealized_pnl_short():
    # +$500 on a 10-share SHORT bought at 150, now at 100.
    assert compute_unrealized_pnl(150, 100, 10, "SHORT") == pytest.approx(500.0)


def test_compute_unrealized_pnl_losing_long():
    assert compute_unrealized_pnl(200, 180, 5, "LONG") == pytest.approx(-100.0)


def test_compute_unrealized_pnl_unknown_direction_is_zero():
    assert compute_unrealized_pnl(100, 150, 10, "WEIRD") == 0.0


def test_compute_unrealized_pnl_bad_input_returns_zero():
    assert compute_unrealized_pnl(None, 150, 10, "LONG") == 0.0
    assert compute_unrealized_pnl("abc", 150, 10, "LONG") == 0.0


# ──────────────────────── market-hours heuristic ────────────────────────


def test_market_open_weekday_midday():
    dt = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)  # Wed 15:00 UTC
    assert _is_us_market_open(dt) is True


def test_market_closed_weekend():
    dt = datetime(2026, 4, 25, 15, 0, tzinfo=timezone.utc)  # Saturday
    assert _is_us_market_open(dt) is False


def test_market_closed_overnight():
    dt = datetime(2026, 4, 22, 3, 0, tzinfo=timezone.utc)  # Wed 3am UTC
    assert _is_us_market_open(dt) is False


# ──────────────────────── PriceFeed cache + callback ────────────────────────


def test_feed_records_price_and_invokes_callback():
    calls = []
    feed = PriceFeed(["BTC-USD", "ETH-USD"], on_update=lambda s, p: calls.append((s, p)))

    feed.set_price("BTC-USD", 68000.0)
    feed.set_price("ETH-USD", 3500.0)

    assert feed.get_price("BTC-USD") == 68000.0
    assert feed.get_prices() == {"BTC-USD": 68000.0, "ETH-USD": 3500.0}
    assert calls == [("BTC-USD", 68000.0), ("ETH-USD", 3500.0)]


def test_feed_suppresses_duplicate_callbacks():
    calls = []
    feed = PriceFeed(["BTC-USD"], on_update=lambda s, p: calls.append((s, p)))

    feed.set_price("BTC-USD", 68000.0)
    feed.set_price("BTC-USD", 68000.0)  # identical — no callback
    feed.set_price("BTC-USD", 68100.0)

    assert calls == [("BTC-USD", 68000.0), ("BTC-USD", 68100.0)]


def test_feed_rejects_non_positive_prices():
    calls = []
    feed = PriceFeed(["BTC-USD"], on_update=lambda s, p: calls.append((s, p)))
    feed.set_price("BTC-USD", 0)
    feed.set_price("BTC-USD", -5)
    feed.set_price("BTC-USD", "nope")  # type: ignore[arg-type]
    assert calls == []
    assert feed.get_price("BTC-USD") is None


def test_feed_get_prices_returns_snapshot_copy():
    feed = PriceFeed(["BTC-USD"])
    feed.set_price("BTC-USD", 1.0)
    snap = feed.get_prices()
    snap["BTC-USD"] = 999.0
    # Mutating the snapshot does NOT mutate the feed.
    assert feed.get_price("BTC-USD") == 1.0


def test_feed_thread_safety_under_concurrent_writes():
    feed = PriceFeed(["BTC-USD", "ETH-USD"])
    n = 500

    def writer(sym, base):
        for i in range(n):
            feed.set_price(sym, base + i)

    t1 = threading.Thread(target=writer, args=("BTC-USD", 1000.0))
    t2 = threading.Thread(target=writer, args=("ETH-USD", 2000.0))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert feed.get_price("BTC-USD") == 1000.0 + (n - 1)
    assert feed.get_price("ETH-USD") == 2000.0 + (n - 1)


def test_feed_is_ws_connected_starts_false():
    feed = PriceFeed(["BTC-USD"])
    assert feed.is_ws_connected() is False


def test_feed_stream_url_builds_combined_subscription():
    feed = PriceFeed(["BTC-USD", "ETH-USD", "SPY"])
    url = feed._binance_stream_url()
    assert url.startswith("wss://stream.binance.com:9443/stream?streams=")
    # Non-crypto symbols must NOT appear.
    assert "spy" not in url.lower()
    # Both crypto streams must appear.
    assert "btcusdt@ticker" in url
    assert "ethusdt@ticker" in url


# ──────────────────────── stock poll glue ────────────────────────


class _FakeYf:
    """Minimal yfinance stand-in used by _poll_stocks_once tests."""

    def __init__(self, df):
        self._df = df

    def download(self, **kwargs):
        return self._df


def _multi_symbol_frame(symbols, closes):
    # Build a yfinance-style multi-index DataFrame:
    # columns = MultiIndex([(symbol, field)]).
    cols = pd.MultiIndex.from_product([symbols, ["Close"]])
    data = {(s, "Close"): [c] for s, c in zip(symbols, closes)}
    return pd.DataFrame(data, columns=cols)


def test_poll_stocks_once_records_multi_symbol_prices():
    feed = PriceFeed(["SPY", "QQQ"])
    df = _multi_symbol_frame(["SPY", "QQQ"], [500.0, 420.0])
    feed._poll_stocks_once(_FakeYf(df))
    assert feed.get_price("SPY") == 500.0
    assert feed.get_price("QQQ") == 420.0


def test_poll_stocks_once_handles_single_symbol_frame():
    feed = PriceFeed(["SPY"])
    df = pd.DataFrame({"Close": [499.5]})
    feed._poll_stocks_once(_FakeYf(df))
    assert feed.get_price("SPY") == 499.5


def test_poll_stocks_once_skips_missing_symbols():
    feed = PriceFeed(["SPY", "QQQ"])
    df = _multi_symbol_frame(["SPY"], [500.0])  # QQQ column missing
    feed._poll_stocks_once(_FakeYf(df))
    assert feed.get_price("SPY") == 500.0
    assert feed.get_price("QQQ") is None


def test_poll_stocks_once_empty_frame_is_noop():
    feed = PriceFeed(["SPY"])
    feed._poll_stocks_once(_FakeYf(pd.DataFrame()))
    assert feed.get_prices() == {}


def test_start_is_idempotent():
    # Starting twice shouldn't spawn extra threads. We avoid actually running
    # Binance: give it only non-crypto symbols so _run_binance is not used,
    # and swap the poll sleep to a no-op that also aborts the loop.
    calls = []

    def immediate_sleep(_sec):
        feed._stop.set()

    feed = PriceFeed(
        ["SPY"],
        on_update=lambda s, p: calls.append((s, p)),
        stock_poll_sleep=immediate_sleep,
    )
    # Replace the internal poll to avoid needing yfinance in tests.
    feed._poll_stocks_once = lambda _yf: None  # type: ignore[assignment]

    feed.start()
    first_thread = feed._poll_thread
    feed.start()  # second call — no new thread should be created
    assert feed._poll_thread is first_thread
    feed.stop()
    # Give the thread a moment to exit.
    time.sleep(0.05)
