"""
Continuous paper-trading runner.

Spins up a :class:`MarketScanner` (with Kronos enabled), then schedules per-market
scan cycles at the cadence specified by each ``MarketConfig``. Crypto markets
default to a 15-minute cadence and stocks/commodities/forex default to 1 hour
unless their ``scan_interval_hours`` says otherwise.

Logs every cycle and prints a periodic performance summary across all markets
and strategies. Runs until interrupted (Ctrl-C) or until ``max_cycles`` cycles
have elapsed (useful for tests).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from market_config import MARKETS, MarketCategory, MarketConfig
from scanner import MarketScanner

logger = logging.getLogger(__name__)


# Default cadence per category when ``scan_interval_hours`` isn't specified
# in ``MarketConfig``.
DEFAULT_CADENCE_SECONDS: Dict[MarketCategory, int] = {
    MarketCategory.CRYPTO: 15 * 60,         # 15 minutes
    MarketCategory.STOCKS: 60 * 60,         # 1 hour
    MarketCategory.COMMODITIES: 60 * 60,    # 1 hour
    MarketCategory.FOREX: 60 * 60,          # 1 hour
}


def _resolve_cadence(config: MarketConfig) -> int:
    """Pick the scan cadence (in seconds) for a single market."""
    # If MarketConfig is configured with a sub-hour interval, honour it.
    raw = float(config.scan_interval_hours or 0.0)
    if raw <= 0:
        return DEFAULT_CADENCE_SECONDS.get(config.category, 60 * 60)
    # Crypto: drop to 15 minutes to keep up with intraday flow even when
    # scan_interval_hours says 4h.
    if config.category == MarketCategory.CRYPTO and raw >= 1.0:
        return DEFAULT_CADENCE_SECONDS[MarketCategory.CRYPTO]
    return int(raw * 3600)


@dataclass
class _MarketSchedule:
    symbol: str
    cadence_seconds: int
    next_run_at: float
    last_signals: int = 0
    last_trades: int = 0
    total_signals: int = 0
    total_trades: int = 0
    total_cycles: int = 0
    last_error: Optional[str] = None
    last_regime: Optional[str] = None
    last_improvement_levels: List[str] = field(default_factory=list)


@dataclass
class ContinuousRunner:
    """Schedule and execute scan cycles per-market on a continuous loop."""

    scanner: MarketScanner
    symbols: List[str] = field(default_factory=lambda: list(MARKETS.keys()))
    summary_every_seconds: int = 30 * 60     # print performance summary every 30 min
    sleep_seconds: float = 5.0
    clock: Callable[[], float] = time.time
    sleeper: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self._stop = False
        self._schedules: Dict[str, _MarketSchedule] = {}
        for symbol in self.symbols:
            cfg = MARKETS.get(symbol)
            if cfg is None:
                logger.warning("Unknown market %s — skipping", symbol)
                continue
            cadence = _resolve_cadence(cfg)
            self._schedules[symbol] = _MarketSchedule(
                symbol=symbol,
                cadence_seconds=cadence,
                next_run_at=self.clock(),
            )
        self._last_summary_at: float = self.clock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def request_stop(self, *_args) -> None:
        """Trigger a graceful shutdown; safe to call from a signal handler."""
        logger.info("Stop requested — finishing pending work and exiting.")
        self._stop = True

    def install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)
        except (ValueError, OSError):
            # Signal handlers are not available off the main thread; ignore.
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_cycles: Optional[int] = None) -> Dict[str, _MarketSchedule]:
        """Run the continuous loop. Returns the final per-symbol schedules."""
        cycles = 0
        logger.info(
            "Continuous runner starting — symbols=%s, summary_every=%ds",
            list(self._schedules.keys()),
            self.summary_every_seconds,
        )
        while not self._stop:
            if max_cycles is not None and cycles >= max_cycles:
                break
            now = self.clock()
            ran_any = False
            for symbol, sched in self._schedules.items():
                if max_cycles is not None and cycles >= max_cycles:
                    break
                if now < sched.next_run_at:
                    continue
                ran_any = True
                self._run_one(symbol, sched)
                cycles += 1
            self._maybe_print_summary()
            if not ran_any:
                # Sleep until at least one schedule is ready. If our sleeper
                # is a fake that does not advance the clock (as in tests),
                # this naturally short-circuits when ``max_cycles`` is hit.
                next_due = min((s.next_run_at for s in self._schedules.values()), default=0.0)
                wait = max(0.0, min(self.sleep_seconds, next_due - now))
                self.sleeper(wait)
                # If the sleeper didn't advance the clock and we have
                # ``max_cycles`` requested, exit to avoid spinning.
                if self.clock() == now and max_cycles is not None:
                    break
        self._maybe_print_summary(force=True)
        return dict(self._schedules)

    def run_once(self) -> Dict[str, _MarketSchedule]:
        """Run a single cycle for each market regardless of cadence."""
        for symbol, sched in self._schedules.items():
            self._run_one(symbol, sched, force=True)
        self._maybe_print_summary(force=True)
        return dict(self._schedules)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_one(self, symbol: str, sched: _MarketSchedule, force: bool = False) -> None:
        start = self.clock()
        try:
            results = self.scanner.run_scan_cycle(symbols=[symbol])
            sched.last_signals = int(results.get("signals_found", 0))
            sched.last_trades = int(results.get("trades_opened", 0))
            sched.total_signals += sched.last_signals
            sched.total_trades += sched.last_trades
            sched.total_cycles += 1
            sched.last_error = None
            # Self-improvement summary: surface regime + which levels ran.
            regimes = results.get("regimes") or {}
            if isinstance(regimes, dict):
                sched.last_regime = regimes.get(symbol)
            imp = results.get("improvement") or {}
            if isinstance(imp, dict):
                levels = imp.get("levels_run") or []
                if isinstance(levels, list):
                    sched.last_improvement_levels = [str(x) for x in levels]
            elapsed = self.clock() - start
            logger.info(
                "[scan] %s — signals=%d trades=%d stops=%d regime=%s levels=%s (%.2fs)",
                symbol,
                sched.last_signals,
                sched.last_trades,
                int(results.get("stops_triggered", 0)),
                sched.last_regime or "-",
                ",".join(sched.last_improvement_levels) or "-",
                elapsed,
            )
        except Exception as exc:
            sched.last_error = str(exc)
            logger.exception("Scan failed for %s: %s", symbol, exc)
        finally:
            sched.next_run_at = self.clock() + (0 if force else sched.cadence_seconds)

    def _maybe_print_summary(self, force: bool = False) -> None:
        now = self.clock()
        if not force and (now - self._last_summary_at) < self.summary_every_seconds:
            return
        self._last_summary_at = now
        try:
            summary = self.performance_summary()
        except Exception as exc:
            logger.warning("Failed to build performance summary: %s", exc)
            return
        logger.info("Performance summary:\n%s", json.dumps(summary, indent=2, default=str))

    # ------------------------------------------------------------------
    # Performance reporting
    # ------------------------------------------------------------------

    def performance_summary(self) -> Dict:
        """Return aggregate paper-trading performance across all markets.

        Totals come from the unified MASTER portfolio; the ``portfolios`` list
        reports per-symbol realised P&L for analytics.
        """
        engine = self.scanner.engine
        symbols = list(self._schedules.keys()) or list(MARKETS.keys())

        # Master portfolio (unified totals)
        try:
            master = engine.get_master_portfolio() if hasattr(engine, "get_master_portfolio") else None
        except Exception as exc:
            logger.warning("Could not pull master portfolio: %s", exc)
            master = None

        # Equity = master cash + all open position values
        try:
            total_exposure = engine.get_total_exposure() if hasattr(engine, "get_total_exposure") else 0.0
        except Exception:
            total_exposure = 0.0

        if master is not None:
            total_balance = float(master.get("current_balance") or 0.0) + float(total_exposure or 0.0)
            total_pnl = float(master.get("total_pnl") or 0.0)
        else:
            total_balance = 0.0
            total_pnl = 0.0

        portfolios: List[Dict] = []
        for symbol in symbols:
            try:
                p = engine.get_portfolio(symbol)
                if not p:
                    continue
                portfolios.append({
                    "symbol": symbol,
                    "balance": p.get("current_balance"),
                    "total_pnl": p.get("total_pnl"),
                    "trades": p.get("total_trades"),
                    "win_rate": p.get("win_rate"),
                })
            except Exception as exc:
                logger.warning("Could not pull portfolio for %s: %s", symbol, exc)

        # Per-strategy aggregates
        strategy_stats: Dict[str, Dict] = {}
        try:
            rows = engine.get_strategy_performance() if hasattr(engine, "get_strategy_performance") else []
        except Exception:
            rows = []
        for row in rows or []:
            strat = row.get("strategy") if isinstance(row, dict) else None
            if not strat:
                continue
            agg = strategy_stats.setdefault(strat, {"trades": 0, "wins": 0, "pnl": 0.0})
            agg["trades"] += int(row.get("total_trades", 0))
            agg["wins"] += int(row.get("winning_trades", 0))
            agg["pnl"] += float(row.get("total_pnl", 0.0))

        scheduler_summary = {
            symbol: {
                "cadence_seconds": s.cadence_seconds,
                "total_cycles": s.total_cycles,
                "total_signals": s.total_signals,
                "total_trades": s.total_trades,
                "last_error": s.last_error,
                "last_regime": s.last_regime,
                "last_improvement_levels": list(s.last_improvement_levels),
            }
            for symbol, s in self._schedules.items()
        }

        # Pull latest self-improvement snapshot (best-effort, never fatal).
        improvement_snapshot: Optional[Dict[str, object]] = None
        improver = getattr(self.scanner, "self_improver", None)
        if improver is not None:
            try:
                improvement_snapshot = {
                    "strategy_weights": improver.get_strategy_weights(),
                    "disabled_strategies": improver.get_disabled_strategies(),
                    "kronos_accuracy": improver.evaluate_kronos_accuracy(),
                }
            except Exception as exc:
                logger.debug("improvement snapshot failed: %s", exc)
                improvement_snapshot = {"error": str(exc)}

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "master": {
                "balance": float(master.get("current_balance") or 0.0) if master else 0.0,
                "equity": total_balance,
                "total_pnl": total_pnl,
                "total_trades": int(master.get("total_trades") or 0) if master else 0,
                "winning_trades": int(master.get("winning_trades") or 0) if master else 0,
                "losing_trades": int(master.get("losing_trades") or 0) if master else 0,
                "consecutive_losses": int(master.get("consecutive_losses") or 0) if master else 0,
                "open_exposure": float(total_exposure or 0.0),
            } if master else None,
            "portfolios": portfolios,
            "totals": {
                "balance": total_balance,
                "pnl": total_pnl,
                "n_markets": len(portfolios),
            },
            "strategies": strategy_stats,
            "schedules": scheduler_summary,
            "improvement": improvement_snapshot,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantAgent continuous paper trading")
    parser.add_argument("--db", default=None, help="Path to SQLite database (optional)")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to scan (default: every market in market_config.MARKETS)",
    )
    parser.add_argument(
        "--summary-every-minutes",
        type=int,
        default=30,
        help="Minutes between performance summaries.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after N total scan cycles (useful for smoke tests).",
    )
    parser.add_argument(
        "--no-kronos",
        action="store_true",
        help="Disable Kronos forecasting (Kronos strategies will be skipped).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan for every market and exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    scanner = MarketScanner(db_path=args.db, use_kronos=not args.no_kronos)
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=args.symbols or list(MARKETS.keys()),
        summary_every_seconds=args.summary_every_minutes * 60,
    )
    runner.install_signal_handlers()

    if args.once:
        runner.run_once()
    else:
        runner.run(max_cycles=args.max_cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
