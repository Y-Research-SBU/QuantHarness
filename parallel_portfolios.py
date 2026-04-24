"""
Parallel paper-trading portfolios — A/B/C/D/E parameter exploration.

Spins up five independent PaperTradingEngine instances, each with its own
SQLite file and its own parameter / strategy preset. The same scanner runs
against all five so they see identical signals; what differs is which
signals each one accepts and how its stops are sized.

Why this exists: the live runner uses one set of weights at a time. With
several portfolios in parallel we can find out which preset survives best
*without* a six-month live A/B test.

Configurations:
    A — Aggressive RSI (60/40), tight stops (0.75x ATR SL)
    B — Conservative RSI (80/20), wide stops (2.0x ATR SL)
    C — Default params, Kronos-only strategies (no pure technical)
    D — Default params, no Kronos (pure technical only)
    E — Adaptive (live self-improvement weights — control)

Usage:
    python3 parallel_portfolios.py
    python3 parallel_portfolios.py --trades 200 --merge-best
    python3 parallel_portfolios.py --markets BTC-USD,ETH-USD --max-cycles 50
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_params import AdaptiveParams, DEFAULT_PARAMS
from db_schema import get_connection, init_db
from market_config import MARKETS, StrategyType
from scanner import MarketScanner
from self_improvement_schema import apply_self_improvement_schema
from self_improver import SelfImprover

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio presets
# ---------------------------------------------------------------------------


KRONOS_STRATEGIES = (
    StrategyType.KRONOS_MOMENTUM_CONFIRM,
    StrategyType.KRONOS_DIVERGENCE,
    StrategyType.MULTI_TIMEFRAME_KRONOS,
)
TECHNICAL_STRATEGIES = (
    StrategyType.MOMENTUM,
    StrategyType.MEAN_REVERSION,
    StrategyType.BREAKOUT,
    StrategyType.MULTI_FACTOR,
)


@dataclass
class PortfolioPreset:
    name: str
    db_path: str
    description: str
    # Parameters seeded into adaptive_params for every (strategy, symbol) pair.
    seed_params: Dict[str, float] = field(default_factory=dict)
    # Restrict signals to these strategies (None = all).
    allowed_strategies: Optional[Tuple[StrategyType, ...]] = None
    # When True, the scanner runs Kronos forecasting; otherwise skip it.
    use_kronos: bool = True
    # When True, this portfolio uses the live self-improvement weights instead
    # of seeded params (the control group).
    use_self_improvement: bool = True


def default_presets(db_dir: str = ".") -> List[PortfolioPreset]:
    """Return the canonical A–E portfolios."""
    def _path(name: str) -> str:
        return os.path.join(db_dir, f"portfolio_{name.lower()}.db")

    return [
        PortfolioPreset(
            name="A",
            db_path=_path("a"),
            description="Aggressive RSI (60/40), tight ATR stops",
            seed_params={
                "rsi_overbought": 60.0,
                "rsi_oversold": 40.0,
                "sl_atr_mult": 0.75,
                "tp_atr_mult": 1.5,
            },
            use_kronos=True,
            use_self_improvement=False,
        ),
        PortfolioPreset(
            name="B",
            db_path=_path("b"),
            description="Conservative RSI (80/20), wide ATR stops",
            seed_params={
                "rsi_overbought": 80.0,
                "rsi_oversold": 20.0,
                "sl_atr_mult": 2.0,
                "tp_atr_mult": 3.0,
            },
            use_kronos=True,
            use_self_improvement=False,
        ),
        PortfolioPreset(
            name="C",
            db_path=_path("c"),
            description="Defaults, Kronos-only strategies",
            seed_params=dict(DEFAULT_PARAMS),
            allowed_strategies=KRONOS_STRATEGIES,
            use_kronos=True,
            use_self_improvement=False,
        ),
        PortfolioPreset(
            name="D",
            db_path=_path("d"),
            description="Defaults, pure technical strategies (no Kronos)",
            seed_params=dict(DEFAULT_PARAMS),
            allowed_strategies=TECHNICAL_STRATEGIES,
            use_kronos=False,
            use_self_improvement=False,
        ),
        PortfolioPreset(
            name="E",
            db_path=_path("e"),
            description="Adaptive (live self-improvement weights — control)",
            seed_params={},
            use_kronos=True,
            use_self_improvement=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


@dataclass
class PortfolioPerformance:
    name: str
    description: str
    db_path: str
    total_trades: int = 0
    closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    final_equity: float = 0.0


def _sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    std = arr.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(arr.mean() / std * math.sqrt(len(arr)))


def _max_drawdown_pct(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    arr = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    valid = peak > 0
    if not valid.any():
        return 0.0
    drawdowns = (peak - arr) / np.where(valid, peak, 1.0)
    return float(drawdowns.max())


def measure_performance(preset: PortfolioPreset) -> PortfolioPerformance:
    """Pull aggregate stats for one portfolio's DB."""
    perf = PortfolioPerformance(
        name=preset.name,
        description=preset.description,
        db_path=preset.db_path,
    )
    if not os.path.exists(preset.db_path):
        return perf

    with get_connection(preset.db_path) as conn:
        master = conn.execute(
            "SELECT * FROM portfolios WHERE symbol = '__MASTER__'"
        ).fetchone()
        if master is not None:
            master = dict(master)
            perf.total_trades = int(master.get("total_trades") or 0)
            perf.winning_trades = int(master.get("winning_trades") or 0)
            perf.losing_trades = int(master.get("losing_trades") or 0)
            perf.total_pnl = float(master.get("total_pnl") or 0.0)
            perf.final_equity = float(master.get("current_balance") or 0.0)

        closed = conn.execute(
            "SELECT pnl_pct FROM trades WHERE status IN ('CLOSED','STOPPED') ORDER BY exit_time"
        ).fetchall()
        returns = [float(r["pnl_pct"] or 0.0) for r in closed]
        perf.closed_trades = len(returns)
        perf.sharpe_ratio = _sharpe(returns)
        if perf.closed_trades:
            perf.win_rate = perf.winning_trades / perf.closed_trades

        # Reconstruct equity curve from snapshots (master rows only).
        snapshots = conn.execute(
            "SELECT balance FROM portfolio_snapshots WHERE symbol = '__MASTER__' ORDER BY snapshot_time"
        ).fetchall()
        curve = [float(s["balance"]) for s in snapshots]
        if curve:
            perf.max_drawdown_pct = _max_drawdown_pct(curve)
            if not perf.final_equity:
                perf.final_equity = curve[-1]

    return perf


# ---------------------------------------------------------------------------
# Portfolio runner
# ---------------------------------------------------------------------------


def _seed_adaptive_params(preset: PortfolioPreset, symbols: List[str]) -> None:
    """Insert the preset's parameter set into adaptive_params for every (strategy, symbol)."""
    if not preset.seed_params:
        return
    apply_self_improvement_schema(preset.db_path)
    ap = AdaptiveParams(db_path=preset.db_path)
    strategies = list(StrategyType)
    for strat in strategies:
        for symbol in symbols:
            for param_name, value in preset.seed_params.items():
                ap.set_param(
                    strategy=strat.value,
                    symbol=symbol,
                    param_name=param_name,
                    param_value=float(value),
                    sample_size=0,
                    improvement_pct=0.0,
                )


def _build_scanner(preset: PortfolioPreset) -> MarketScanner:
    """Construct a scanner with the preset's restrictions applied."""
    init_db(preset.db_path)
    apply_self_improvement_schema(preset.db_path)

    scanner = MarketScanner(
        db_path=preset.db_path,
        use_kronos=preset.use_kronos,
        use_self_improvement=preset.use_self_improvement,
    )

    # Apply allowed-strategies restriction by mutating the in-memory MarketConfig
    # objects on the scanner side. We never write this back to disk.
    if preset.allowed_strategies is not None:
        allowed = set(preset.allowed_strategies)
        # Patch MARKETS for this scanner instance via a side-channel: the
        # scanner reads ``MARKETS`` at scan time, so we wrap run_scan_cycle.
        original_scan_market = scanner.scan_market

        def _scan_market_filtered(symbol: str, config):
            filtered = type(config)(
                symbol=config.symbol,
                display_name=config.display_name,
                category=config.category,
                timeframes=list(config.timeframes),
                enabled_strategies=[s for s in config.enabled_strategies if s in allowed],
                scan_interval_hours=config.scan_interval_hours,
                initial_balance=config.initial_balance,
                max_position_pct=config.max_position_pct,
                correlation_group=config.correlation_group,
            )
            return original_scan_market(symbol, filtered)

        scanner.scan_market = _scan_market_filtered  # type: ignore[assignment]

    return scanner


def run_portfolio_cycle(
    preset: PortfolioPreset,
    symbols: List[str],
) -> Dict[str, Any]:
    """Run one scan cycle for a portfolio. Returns the cycle result dict."""
    scanner = _build_scanner(preset)
    return scanner.run_scan_cycle(symbols=symbols)


def run_until(
    presets: List[PortfolioPreset],
    target_trades: int,
    symbols: List[str],
    max_cycles: int = 200,
) -> Dict[str, PortfolioPerformance]:
    """Run scan cycles until each portfolio has at least ``target_trades`` closed trades.

    Returns the performance snapshot per portfolio.
    """
    scanners: Dict[str, MarketScanner] = {}
    for preset in presets:
        _seed_adaptive_params(preset, symbols)
        scanners[preset.name] = _build_scanner(preset)

    perf: Dict[str, PortfolioPerformance] = {p.name: measure_performance(p) for p in presets}

    for cycle in range(max_cycles):
        all_done = True
        for preset in presets:
            current = measure_performance(preset)
            if current.closed_trades < target_trades:
                all_done = False
                try:
                    scanners[preset.name].run_scan_cycle(symbols=symbols)
                except Exception as exc:
                    logger.warning("portfolio %s cycle %d failed: %s", preset.name, cycle, exc)
            perf[preset.name] = current
        if all_done:
            break

    return {name: measure_performance(preset) for name, preset in zip([p.name for p in presets], presets)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_comparison_table(perfs: Dict[str, PortfolioPerformance]) -> str:
    """Render a tabular performance comparison."""
    header = (
        f"{'Portfolio':<10} {'Description':<48} "
        f"{'Trades':>7} {'WinRt':>7} {'Sharpe':>7} {'MaxDD%':>7} {'P&L':>10} {'Equity':>10}"
    )
    rows: List[str] = [header, "-" * len(header)]
    for name in sorted(perfs.keys()):
        p = perfs[name]
        rows.append(
            f"{p.name:<10} {p.description[:46]:<48} "
            f"{p.closed_trades:>7} {p.win_rate * 100:>6.1f}% "
            f"{p.sharpe_ratio:>7.2f} {p.max_drawdown_pct * 100:>6.1f}% "
            f"{p.total_pnl:>+10.2f} {p.final_equity:>10.2f}"
        )
    return "\n".join(rows)


def pick_best(perfs: Dict[str, PortfolioPerformance]) -> Optional[PortfolioPerformance]:
    """Return the highest-Sharpe portfolio that opened at least one closed trade."""
    candidates = [p for p in perfs.values() if p.closed_trades > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.sharpe_ratio, p.total_pnl))


def merge_best_into_live(
    best: PortfolioPerformance,
    live_db: Optional[str] = None,
) -> Dict[str, int]:
    """Copy the winning portfolio's adaptive_params + strategy_evolution into live.

    Only the per-symbol adaptive parameters move over — we never overwrite the
    live trades or portfolio rows. Returns counts copied.
    """
    live_db = live_db or os.environ.get("DB_PATH") or (
        "/data/paper_trades.db" if os.path.isdir("/data") else "paper_trades.db"
    )
    apply_self_improvement_schema(live_db)
    counts: Dict[str, int] = {}

    src = sqlite3.connect(best.db_path)
    src.row_factory = sqlite3.Row
    try:
        for table in ("adaptive_params", "strategy_evolution"):
            try:
                rows = src.execute(f"SELECT * FROM {table}").fetchall()
            except sqlite3.OperationalError:
                counts[table] = 0
                continue
            if not rows:
                counts[table] = 0
                continue
            cols = [c for c in rows[0].keys() if c != "id"]
            placeholders = ", ".join("?" for _ in cols)
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
            with get_connection(live_db) as dst:
                for r in rows:
                    dst.execute(sql, tuple(r[c] for c in cols))
            counts[table] = len(rows)
    finally:
        src.close()
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel paper-trading portfolios A/B/C/D/E.")
    parser.add_argument(
        "--trades",
        type=int,
        default=100,
        help="Target closed trades per portfolio before reporting.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=200,
        help="Hard cap on scan cycles per portfolio.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols. Default: every market in market_config.",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=".",
        help="Directory in which to place portfolio_*.db files.",
    )
    parser.add_argument(
        "--live-db",
        type=str,
        default=None,
        help="Path to the live DB used by --merge-best.",
    )
    parser.add_argument(
        "--merge-best",
        action="store_true",
        help="After comparison, copy the winning portfolio's adaptive_params into live DB.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to dump performance results as JSON.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else list(MARKETS.keys())
    )

    presets = default_presets(db_dir=args.db_dir)

    perfs = run_until(
        presets=presets,
        target_trades=args.trades,
        symbols=symbols,
        max_cycles=args.max_cycles,
    )

    print()
    print(render_comparison_table(perfs))
    best = pick_best(perfs)
    if best is None:
        print("\nNo portfolio produced any closed trades — nothing to merge.")
    else:
        print(
            f"\nBest portfolio: {best.name} "
            f"(Sharpe={best.sharpe_ratio:.2f}, P&L=${best.total_pnl:+.2f}, "
            f"trades={best.closed_trades})"
        )
        if args.merge_best:
            counts = merge_best_into_live(best, live_db=args.live_db)
            print(f"Merged into live DB: {counts}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(
                {
                    name: {
                        "name": p.name,
                        "description": p.description,
                        "db_path": p.db_path,
                        "total_trades": p.total_trades,
                        "closed_trades": p.closed_trades,
                        "winning_trades": p.winning_trades,
                        "losing_trades": p.losing_trades,
                        "win_rate": p.win_rate,
                        "total_pnl": p.total_pnl,
                        "sharpe_ratio": p.sharpe_ratio,
                        "max_drawdown_pct": p.max_drawdown_pct,
                        "final_equity": p.final_equity,
                    }
                    for name, p in perfs.items()
                },
                fh,
                indent=2,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
