"""Tests for run_continuous.ContinuousRunner.

These tests use a fake clock and a fake :class:`MarketScanner` to drive the
loop deterministically without sleeping or touching the network.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest

from market_config import MARKETS, MarketCategory, MarketConfig
from run_continuous import (
    DEFAULT_CADENCE_SECONDS,
    ContinuousRunner,
    _resolve_cadence,
    main,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeEngine:
    """Substitute paper-trading engine that returns canned portfolios."""

    def __init__(self, portfolios: Optional[Dict[str, Dict]] = None):
        self._portfolios = portfolios or {}
        self.snapshot_calls: List[str] = []

    def get_portfolio(self, symbol: str) -> Optional[Dict]:
        return self._portfolios.get(symbol)

    def take_snapshot(self, symbol: str) -> None:
        self.snapshot_calls.append(symbol)


class FakeScanner:
    """Substitute MarketScanner that records every cycle."""

    def __init__(self, signals: int = 0, trades: int = 0, raise_for: Optional[str] = None):
        self.engine = FakeEngine()
        self.cycles: List[List[str]] = []
        self._signals = signals
        self._trades = trades
        self._raise_for = raise_for

    def run_scan_cycle(self, symbols: Optional[List[str]] = None) -> Dict:
        self.cycles.append(list(symbols or []))
        if self._raise_for and self._raise_for in (symbols or []):
            raise RuntimeError("simulated failure")
        return {
            "signals_found": self._signals,
            "trades_opened": self._trades,
            "stops_triggered": 0,
        }


class FakeClock:
    """A deterministic clock that advances on every tick()."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, delta: float = 1.0) -> None:
        self.now += delta


# ---------------------------------------------------------------------------
# _resolve_cadence
# ---------------------------------------------------------------------------


def test_resolve_cadence_for_crypto_clamps_to_default():
    cfg = MarketConfig(
        symbol="BTC-USD", display_name="x", category=MarketCategory.CRYPTO,
        timeframes=["4h"], scan_interval_hours=4.0,
    )
    assert _resolve_cadence(cfg) == DEFAULT_CADENCE_SECONDS[MarketCategory.CRYPTO]


def test_resolve_cadence_respects_subhour_for_crypto():
    cfg = MarketConfig(
        symbol="X", display_name="x", category=MarketCategory.CRYPTO,
        timeframes=["1h"], scan_interval_hours=0.25,  # 15 min
    )
    assert _resolve_cadence(cfg) == int(0.25 * 3600)


def test_resolve_cadence_for_stock():
    cfg = MarketConfig(
        symbol="SPY", display_name="x", category=MarketCategory.STOCKS,
        timeframes=["1d"], scan_interval_hours=24.0,
    )
    assert _resolve_cadence(cfg) == int(24 * 3600)


def test_resolve_cadence_zero_uses_category_default():
    cfg = MarketConfig(
        symbol="X", display_name="x", category=MarketCategory.STOCKS,
        timeframes=["1d"], scan_interval_hours=0.0,
    )
    assert _resolve_cadence(cfg) == DEFAULT_CADENCE_SECONDS[MarketCategory.STOCKS]


# ---------------------------------------------------------------------------
# ContinuousRunner basics
# ---------------------------------------------------------------------------


def test_runner_initialises_schedules_for_known_symbols():
    runner = ContinuousRunner(
        scanner=FakeScanner(),
        symbols=["BTC-USD", "SPY"],
        clock=FakeClock(),
        sleeper=lambda _x: None,
    )
    assert set(runner._schedules.keys()) == {"BTC-USD", "SPY"}


def test_runner_skips_unknown_symbols(caplog):
    runner = ContinuousRunner(
        scanner=FakeScanner(),
        symbols=["NOPE", "BTC-USD"],
        clock=FakeClock(),
        sleeper=lambda _x: None,
    )
    assert "BTC-USD" in runner._schedules
    assert "NOPE" not in runner._schedules


def test_runner_run_once_executes_each_symbol():
    scanner = FakeScanner(signals=2, trades=1)
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=["BTC-USD", "ETH-USD"],
        clock=FakeClock(),
        sleeper=lambda _x: None,
    )
    runner.run_once()
    assert scanner.cycles == [["BTC-USD"], ["ETH-USD"]]
    for sched in runner._schedules.values():
        assert sched.total_cycles == 1
        assert sched.total_signals == 2
        assert sched.total_trades == 1


def test_runner_max_cycles_stops_loop():
    scanner = FakeScanner(signals=0, trades=0)
    clock = FakeClock()
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=["BTC-USD", "ETH-USD"],
        clock=clock,
        sleeper=lambda _x: None,
        summary_every_seconds=10**9,
    )
    # Set cadence to 0 so the schedule is immediately ready again after each run.
    for sched in runner._schedules.values():
        sched.cadence_seconds = 0
    runner.run(max_cycles=3)
    assert len(scanner.cycles) == 3


def test_runner_respects_cadence_between_cycles():
    scanner = FakeScanner()
    clock = FakeClock()
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=["BTC-USD"],
        clock=clock,
        sleeper=lambda _x: clock.tick(60),
        summary_every_seconds=10**9,
    )
    # Force the BTC schedule to 30s for this test.
    runner._schedules["BTC-USD"].cadence_seconds = 30
    runner._schedules["BTC-USD"].next_run_at = 0.0
    runner.run(max_cycles=2)
    assert len(scanner.cycles) == 2
    assert runner._schedules["BTC-USD"].next_run_at >= 30


def test_runner_records_errors_per_symbol():
    scanner = FakeScanner(raise_for="ETH-USD")
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=["BTC-USD", "ETH-USD"],
        clock=FakeClock(),
        sleeper=lambda _x: None,
    )
    runner.run_once()
    assert runner._schedules["BTC-USD"].last_error is None
    assert "simulated failure" in (runner._schedules["ETH-USD"].last_error or "")


def test_runner_continues_after_individual_failures():
    scanner = FakeScanner(raise_for="BTC-USD")
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=["BTC-USD", "ETH-USD"],
        clock=FakeClock(),
        sleeper=lambda _x: None,
    )
    runner.run_once()
    # ETH-USD still got scanned.
    assert ["ETH-USD"] in scanner.cycles


def test_runner_request_stop_short_circuits_loop():
    scanner = FakeScanner()
    clock = FakeClock()

    def stopper(_):
        runner.request_stop()
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=["BTC-USD"],
        clock=clock,
        sleeper=stopper,
    )
    runner._schedules["BTC-USD"].cadence_seconds = 60
    runner._schedules["BTC-USD"].next_run_at = 999.0  # so we never run; sleeper trips
    runner.run()
    assert runner._stop is True


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------


def test_performance_summary_shape():
    scanner = FakeScanner()
    scanner.engine = FakeEngine(portfolios={
        "BTC-USD": {
            "current_balance": 10500.0, "total_pnl": 500.0,
            "total_trades": 4, "win_rate": 0.5,
        }
    })
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=["BTC-USD"],
        clock=FakeClock(),
        sleeper=lambda _x: None,
    )
    summary = runner.performance_summary()
    assert summary["totals"]["balance"] == pytest.approx(10500.0)
    assert summary["totals"]["pnl"] == pytest.approx(500.0)
    assert summary["totals"]["n_markets"] == 1
    assert summary["portfolios"][0]["symbol"] == "BTC-USD"
    assert "schedules" in summary
    assert "BTC-USD" in summary["schedules"]


def test_performance_summary_handles_missing_engine_methods():
    runner = ContinuousRunner(
        scanner=FakeScanner(),
        symbols=["BTC-USD"],
        clock=FakeClock(),
        sleeper=lambda _x: None,
    )
    summary = runner.performance_summary()
    assert summary["strategies"] == {}


def test_summary_emitted_when_interval_elapsed(caplog):
    scanner = FakeScanner()
    clock = FakeClock()
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=["BTC-USD"],
        clock=clock,
        sleeper=lambda _x: None,
        summary_every_seconds=1,
    )
    clock.tick(5)
    with caplog.at_level("INFO"):
        runner._maybe_print_summary()
    assert any("Performance summary" in rec.message for rec in caplog.records)


def test_summary_skipped_when_interval_not_elapsed(caplog):
    runner = ContinuousRunner(
        scanner=FakeScanner(),
        symbols=["BTC-USD"],
        clock=FakeClock(),
        sleeper=lambda _x: None,
        summary_every_seconds=10**9,
    )
    with caplog.at_level("INFO"):
        runner._maybe_print_summary()
    assert not any("Performance summary" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_once_flag_runs_single_pass(monkeypatch):
    scanner = FakeScanner()

    def fake_scanner_ctor(*args, **kwargs):
        return scanner

    monkeypatch.setattr("run_continuous.MarketScanner", fake_scanner_ctor)
    rc = main(["--symbols", "BTC-USD", "--once"])
    assert rc == 0
    assert scanner.cycles == [["BTC-USD"]]


def test_main_max_cycles(monkeypatch):
    scanner = FakeScanner()

    def fake_scanner_ctor(*args, **kwargs):
        return scanner

    monkeypatch.setattr("run_continuous.MarketScanner", fake_scanner_ctor)
    rc = main(["--symbols", "BTC-USD", "ETH-USD", "--max-cycles", "2"])
    assert rc == 0
    assert len(scanner.cycles) == 2


def test_main_no_kronos_flag_disables_kronos(monkeypatch):
    captured = {}

    class Capture(FakeScanner):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__()

    monkeypatch.setattr("run_continuous.MarketScanner", Capture)
    rc = main(["--symbols", "BTC-USD", "--once", "--no-kronos"])
    assert rc == 0
    assert captured.get("use_kronos") is False
