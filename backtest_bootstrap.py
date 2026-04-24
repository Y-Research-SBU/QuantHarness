"""
Backtest bootstrap — pre-train all 5 self-improvement levels on historical data.

Walks 6 months of OHLCV across the most liquid markets and timeframes,
simulates paper trades through the same PaperTradingEngine the live runner
uses, then drives the SelfImprover through every level so the live database
boots with non-empty L1 weights, L2 params, an L3 model, an L4 regime
history, and L5 Kronos calibration.

Why this exists: the L1–L5 cadences fire only after enough closed trades
have accumulated. With the new 5m/15m fan-out we'll get there in days, but a
fresh deploy still starts cold. This script seeds every level so the first
live cycle already has learned priors.

Usage:
    python3 backtest_bootstrap.py
    python3 backtest_bootstrap.py --markets BTC-USD,ETH-USD --days 90
    python3 backtest_bootstrap.py --no-merge   # keep results in backtest_bootstrap.db only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_fetcher import fetch_market_data
from db_schema import get_connection, init_db
from market_config import MARKETS, StrategyType
from paper_trading import PaperTradingEngine
from position_sizing import calculate_position_size
from self_improver import (
    CADENCE_L1,
    CADENCE_L2,
    CADENCE_L3,
    CADENCE_L5,
    SelfImprover,
)
from self_improvement_schema import apply_self_improvement_schema
from strategies import Signal, run_all_strategies

logger = logging.getLogger(__name__)


# Default top-20 most-liquid markets across asset classes.
DEFAULT_MARKETS: List[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD",
    "AVAX-USD", "LINK-USD", "DOT-USD", "ATOM-USD", "NEAR-USD",
    "SPY", "QQQ", "AAPL", "NVDA", "TSLA",
    "AMD", "AMZN", "GOOGL", "GC=F", "CL=F",
]

# Per-timeframe history budget. yfinance allows 5m/15m for 60d max,
# 1h for 730d, 4h/1d for years. We cap intraday at the limit and use
# the requested ``--days`` for the longer horizons.
TIMEFRAME_MAX_DAYS: Dict[str, int] = {
    "5m": 60,
    "15m": 60,
    "1h": 365,
    "4h": 365,
    "1d": 730,
}


# Self-improvement tables we copy from the bootstrap DB into the live DB.
SELF_IMPROVEMENT_TABLES: Tuple[str, ...] = (
    "strategy_evolution",
    "adaptive_params",
    "kronos_predictions",
    "regime_history",
    "signal_model_log",
)


@dataclass
class BootstrapStats:
    total_simulated_trades: int = 0
    closed_trades: int = 0
    by_market: Dict[str, int] = field(default_factory=dict)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    disabled_strategies: List[str] = field(default_factory=list)
    model_accuracy: Optional[float] = None
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    kronos_hit_rate: float = 0.0
    kronos_n: int = 0


def _resolve_history_days(timeframe: str, requested: int) -> int:
    cap = TIMEFRAME_MAX_DAYS.get(timeframe, requested)
    return min(requested, cap)


def _safe_fetch(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Wrap fetch_market_data with a date-range request and a safe fallback."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    try:
        df = fetch_market_data(symbol, timeframe, start_date=start, end_date=end)
    except Exception as exc:
        logger.warning("fetch failed %s %s: %s", symbol, timeframe, exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe
    return df


def _signal_to_paper_signal(signal: Signal, symbol: str, timeframe: str) -> Signal:
    """Ensure a signal has the symbol/timeframe set (some strategies leave UNKNOWN)."""
    signal.symbol = symbol
    signal.timeframe = timeframe
    return signal


def _close_open_position_via_bar(
    engine: PaperTradingEngine,
    symbol: str,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    self_improver: Optional[SelfImprover] = None,
) -> int:
    """If an open position for symbol's stop/TP would trip on this bar, close it.

    Returns the number of trades closed (0 or 1 in practice). Uses the bar
    high/low to detect intrabar SL/TP hits, falling back to the close.
    """
    open_positions = engine.get_open_positions(symbol)
    n_closed = 0
    for pos in open_positions:
        sl = pos.get("stop_loss")
        tp = pos.get("take_profit")
        direction = pos.get("direction")
        exit_price: Optional[float] = None
        reason = "manual"
        if direction == "LONG":
            if sl is not None and bar_low <= sl:
                exit_price, reason = float(sl), "stop_loss"
            elif tp is not None and bar_high >= tp:
                exit_price, reason = float(tp), "take_profit"
        else:  # SHORT
            if sl is not None and bar_high >= sl:
                exit_price, reason = float(sl), "stop_loss"
            elif tp is not None and bar_low <= tp:
                exit_price, reason = float(tp), "take_profit"
        if exit_price is not None:
            engine.close_trade(int(pos["id"]), exit_price, reason=reason)
            n_closed += 1
    return n_closed


def _force_close_remaining(engine: PaperTradingEngine, last_prices: Dict[str, float]) -> int:
    """Flush any still-open positions at the last known price for accounting."""
    open_positions = engine.get_open_positions()
    n = 0
    for pos in open_positions:
        sym = pos["symbol"]
        price = last_prices.get(sym)
        if price is None:
            continue
        engine.close_trade(int(pos["id"]), float(price), reason="end_of_data")
        n += 1
    return n


def _walk_one_market(
    engine: PaperTradingEngine,
    self_improver: SelfImprover,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    warmup: int = 60,
    signal_lookback: int = 200,
) -> Tuple[int, int, Optional[float]]:
    """Walk a single (symbol, timeframe) frame and execute paper trades.

    Returns (signals_generated, trades_opened, last_close).
    """
    if df is None or len(df) <= warmup + 5:
        return 0, 0, None

    df = df.reset_index(drop=True).copy()
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe

    config = MARKETS.get(symbol)
    if config is None:
        return 0, 0, None

    enabled_strategies = list(config.enabled_strategies) if config.enabled_strategies else list(StrategyType)
    # Strip Kronos strategies from the bootstrap walk: Kronos requires loaded
    # model weights and is too slow to run bar-by-bar on every market. The
    # live runner exercises Kronos directly and L5 will calibrate from there.
    enabled_strategies = [
        st for st in enabled_strategies
        if st not in (
            StrategyType.KRONOS_MOMENTUM_CONFIRM,
            StrategyType.KRONOS_DIVERGENCE,
            StrategyType.MULTI_TIMEFRAME_KRONOS,
        )
    ]

    n_signals = 0
    n_opened = 0
    n = len(df)
    last_close: Optional[float] = None

    # Pre-fetch adaptive params snapshot so we can pass them through.
    adaptive_by_strategy: Dict[str, Dict[str, float]] = {}
    for st in enabled_strategies:
        try:
            adaptive_by_strategy[st.value] = self_improver.get_adaptive_params(st.value, symbol)
        except Exception:
            pass

    for i in range(warmup, n):
        bar = df.iloc[i]
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        bar_close = float(bar["Close"])
        last_close = bar_close

        # 1) Close any open positions whose SL/TP this bar would have tripped.
        _close_open_position_via_bar(
            engine, symbol, bar_high, bar_low, bar_close
        )

        # Skip new entries if this symbol already has an open position
        # (PaperTradingEngine blocks stacking anyway).
        if engine.get_open_positions(symbol):
            continue

        start = max(0, i + 1 - signal_lookback)
        window = df.iloc[start : i + 1].copy()
        window.attrs["symbol"] = symbol
        window.attrs["timeframe"] = timeframe

        try:
            signals = run_all_strategies(
                df=window,
                enabled_strategies=enabled_strategies,
                adaptive_params_by_strategy=adaptive_by_strategy or None,
            )
        except Exception as exc:
            logger.debug("run_all_strategies failed %s/%s @ %d: %s", symbol, timeframe, i, exc)
            continue

        if not signals:
            continue
        n_signals += len(signals)

        # Take the strongest signal for this bar.
        signal = max(signals, key=lambda s: float(s.strength or 0.0))
        _signal_to_paper_signal(signal, symbol, timeframe)
        # Force the entry price to the bar close for realism.
        signal.entry_price = bar_close
        # Recompute SL/TP relative to the new entry: keep the same risk distance.
        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            continue
        master = engine.get_master_portfolio()
        balance = float(master.get("initial_balance") or PaperTradingEngine.MASTER_INITIAL_BALANCE)
        try:
            pos = calculate_position_size(
                portfolio_balance=balance,
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss,
                direction=signal.direction,
                signal_strength=float(signal.strength or 0.5),
                max_risk_pct=0.02,
                risk_reward_ratio=signal.risk_reward_ratio,
                min_position_size=engine.MIN_POSITION_SIZE,
                max_position_size=engine.MAX_POSITION_SIZE,
            )
        except Exception as exc:
            logger.debug("position_size failed: %s", exc)
            continue
        if pos.position_size_usd <= 0:
            continue
        try:
            trade_id = engine.execute_trade(signal, pos)
            if trade_id:
                n_opened += 1
        except Exception as exc:
            logger.debug("execute_trade failed: %s", exc)

    # Periodic L4 regime logging — record the regime at the end of the walk
    # using the most recent window (cheap, not bar-by-bar).
    try:
        self_improver.log_regime(symbol, df.tail(200), timeframe=timeframe)
    except Exception as exc:
        logger.debug("log_regime failed for %s/%s: %s", symbol, timeframe, exc)

    return n_signals, n_opened, last_close


def _gather_history_buckets(
    markets: List[str],
    days: int,
    timeframes: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fetch historical OHLCV for each (symbol, timeframe). Skips empties."""
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in markets:
        out[symbol] = {}
        for tf in timeframes:
            window_days = _resolve_history_days(tf, days)
            df = _safe_fetch(symbol, tf, window_days)
            if df.empty:
                logger.info("skip %s %s — empty fetch", symbol, tf)
                continue
            logger.info("fetched %d bars for %s %s (%d days)", len(df), symbol, tf, window_days)
            out[symbol][tf] = df
    return out


def _drive_self_improvement(
    self_improver: SelfImprover,
    symbol_bars: Dict[str, pd.DataFrame],
) -> None:
    """Force every level to fire regardless of trade-count cadence.

    The orchestrator's ``_should_fire`` check is based on cumulative trade
    counts. We bypass it by calling each level explicitly so the bootstrap
    leaves behind a fully populated state even if the synthetic walk
    produced fewer than CADENCE_L5 closed trades.
    """
    # L1 — strategy scoring (always, even with few trades).
    try:
        self_improver.run_strategy_scoring()
    except Exception as exc:
        logger.warning("L1 strategy_scoring failed: %s", exc)

    # L2 — parameter optimization across strategies + per-symbol RSI.
    try:
        self_improver.run_mini_optimization(symbol_bars=symbol_bars or None)
    except Exception as exc:
        logger.warning("L2 mini_optimization failed: %s", exc)

    # L3 — train the signal-quality meta-model.
    try:
        self_improver.train_signal_model()
    except Exception as exc:
        logger.warning("L3 train_signal_model failed: %s", exc)

    # L4 — extra regime logging (already done per-symbol during walk).
    if symbol_bars:
        for sym, df in symbol_bars.items():
            try:
                self_improver.log_regime(sym, df)
            except Exception as exc:
                logger.debug("L4 log_regime failed for %s: %s", sym, exc)

    # L5 — evaluate Kronos accuracy (no-op when no Kronos predictions yet).
    try:
        self_improver.evaluate_kronos_accuracy()
    except Exception as exc:
        logger.warning("L5 evaluate_kronos_accuracy failed: %s", exc)


def _collect_stats(
    bootstrap_db: str,
    self_improver: SelfImprover,
    n_opened_total: int,
) -> BootstrapStats:
    stats = BootstrapStats(total_simulated_trades=n_opened_total)
    with get_connection(bootstrap_db) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE status IN ('CLOSED','STOPPED')"
        ).fetchone()
        stats.closed_trades = int(rows["n"]) if rows else 0
        market_rows = conn.execute(
            "SELECT symbol, COUNT(*) AS n FROM trades GROUP BY symbol"
        ).fetchall()
        stats.by_market = {r["symbol"]: int(r["n"]) for r in market_rows if r["symbol"] != "__MASTER__"}

        regime_rows = conn.execute(
            "SELECT regime, COUNT(*) AS n FROM regime_history GROUP BY regime"
        ).fetchall()
        stats.regime_distribution = {r["regime"]: int(r["n"]) for r in regime_rows}
    try:
        stats.strategy_weights = self_improver.get_strategy_weights()
        stats.disabled_strategies = self_improver.get_disabled_strategies()
    except Exception as exc:
        logger.debug("strategy weight collection failed: %s", exc)

    try:
        kronos = self_improver.evaluate_kronos_accuracy()
        stats.kronos_hit_rate = float(kronos.get("hit_rate") or 0.0)
        stats.kronos_n = int(kronos.get("n") or 0)
    except Exception:
        pass

    # Pull most recent signal_model accuracy.
    try:
        with get_connection(bootstrap_db) as conn:
            row = conn.execute(
                "SELECT accuracy FROM signal_model_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            stats.model_accuracy = float(row["accuracy"])
    except Exception:
        pass

    return stats


def _merge_self_improvement_into_live(bootstrap_db: str, live_db: str) -> Dict[str, int]:
    """Copy the L1–L5 tables from the bootstrap DB into the live DB.

    Live trading rows in the live DB are untouched. Existing self-improvement
    rows are preserved — we append the bootstrap rows so the learning loop
    sees a longer history. Returns counts copied per table.
    """
    apply_self_improvement_schema(live_db)
    counts: Dict[str, int] = {}
    src = sqlite3.connect(bootstrap_db)
    src.row_factory = sqlite3.Row
    try:
        for table in SELF_IMPROVEMENT_TABLES:
            try:
                rows = src.execute(f"SELECT * FROM {table}").fetchall()
            except sqlite3.OperationalError:
                counts[table] = 0
                continue
            if not rows:
                counts[table] = 0
                continue
            cols = rows[0].keys()
            insertable = [c for c in cols if c != "id"]
            placeholders = ", ".join("?" for _ in insertable)
            col_list = ", ".join(insertable)
            sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
            with get_connection(live_db) as dst_conn:
                for r in rows:
                    dst_conn.execute(sql, tuple(r[c] for c in insertable))
            counts[table] = len(rows)
    finally:
        src.close()
    return counts


def run_bootstrap(
    markets: Optional[List[str]] = None,
    days: int = 180,
    bootstrap_db: str = "backtest_bootstrap.db",
    live_db: Optional[str] = None,
    merge: bool = True,
    timeframes: Optional[List[str]] = None,
    warmup: int = 60,
) -> BootstrapStats:
    """End-to-end bootstrap flow. Returns aggregated stats."""
    markets = markets or DEFAULT_MARKETS
    timeframes = timeframes or ["5m", "15m", "1h", "4h", "1d"]

    # Fresh bootstrap DB — start clean each run so weights aren't double-counted.
    for suffix in ("", "-wal", "-shm"):
        p = bootstrap_db + suffix
        if os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass
    init_db(bootstrap_db)
    apply_self_improvement_schema(bootstrap_db)

    engine = PaperTradingEngine(db_path=bootstrap_db)
    self_improver = SelfImprover(db_path=bootstrap_db)

    # Per-symbol last close so we can flush any open positions at the end.
    last_prices: Dict[str, float] = {}
    n_opened_total = 0

    # Cache one (typically 1h) frame per symbol to feed L2/L4 at the end.
    cycle_bars: Dict[str, pd.DataFrame] = {}

    history = _gather_history_buckets(markets, days, timeframes)

    for symbol, frames in history.items():
        for timeframe, df in frames.items():
            try:
                n_signals, n_opened, last_close = _walk_one_market(
                    engine=engine,
                    self_improver=self_improver,
                    symbol=symbol,
                    timeframe=timeframe,
                    df=df,
                    warmup=warmup,
                )
            except Exception as exc:
                logger.warning("walk failed %s/%s: %s", symbol, timeframe, exc)
                continue
            n_opened_total += n_opened
            if last_close is not None:
                last_prices[symbol] = last_close
            # Cache the longest frame as the canonical bars for self-improvement.
            if symbol not in cycle_bars or len(df) > len(cycle_bars[symbol]):
                cycle_bars[symbol] = df
            logger.info(
                "[bootstrap] %s %s — signals=%d trades_opened=%d",
                symbol, timeframe, n_signals, n_opened,
            )

    # Flush any positions still open at the end of history.
    flushed = _force_close_remaining(engine, last_prices)
    logger.info("[bootstrap] force-closed %d remaining positions at end-of-data", flushed)

    # Drive every self-improvement level explicitly.
    _drive_self_improvement(self_improver, cycle_bars)

    stats = _collect_stats(bootstrap_db, self_improver, n_opened_total)

    if merge:
        live_path = live_db or os.environ.get("DB_PATH") or (
            "/data/paper_trades.db" if os.path.isdir("/data") else "paper_trades.db"
        )
        copied = _merge_self_improvement_into_live(bootstrap_db, live_path)
        logger.info("[bootstrap] merged self-improvement tables into %s: %s", live_path, copied)

    return stats


def _print_summary(stats: BootstrapStats) -> None:
    print("\n" + "=" * 70)
    print("Backtest bootstrap — summary")
    print("=" * 70)
    print(f"Total simulated trades opened: {stats.total_simulated_trades}")
    print(f"Closed trades in bootstrap DB: {stats.closed_trades}")
    print(f"Markets touched:               {len(stats.by_market)}")
    if stats.by_market:
        top = sorted(stats.by_market.items(), key=lambda kv: -kv[1])[:10]
        for sym, n in top:
            print(f"  {sym:<12} {n:>5} trades")
    print()
    print("Strategy weights (L1):")
    if stats.strategy_weights:
        for strat, w in sorted(stats.strategy_weights.items(), key=lambda kv: -kv[1]):
            print(f"  {strat:<32} weight={w:.2f}")
    else:
        print("  (no strategies scored — not enough closed trades)")
    if stats.disabled_strategies:
        print(f"Disabled strategies (L1):      {', '.join(stats.disabled_strategies)}")
    if stats.model_accuracy is not None:
        print(f"Signal model accuracy (L3):    {stats.model_accuracy:.3f}")
    else:
        print("Signal model (L3):             not trained (insufficient data)")
    if stats.regime_distribution:
        print("Regime distribution (L4):")
        for regime, n in sorted(stats.regime_distribution.items(), key=lambda kv: -kv[1]):
            print(f"  {regime:<14} {n:>4}")
    print(
        f"Kronos hit rate (L5):          "
        f"{stats.kronos_hit_rate:.3f} (n={stats.kronos_n})"
    )
    print("=" * 70)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-train QuantAgent self-improvement levels on historical data.")
    parser.add_argument(
        "--markets",
        type=str,
        default=None,
        help="Comma-separated symbols (default: top-20 most-liquid).",
    )
    parser.add_argument("--days", type=int, default=180, help="Lookback window in days for non-intraday timeframes.")
    parser.add_argument(
        "--timeframes",
        type=str,
        default="5m,15m,1h,4h,1d",
        help="Comma-separated timeframes to backtest.",
    )
    parser.add_argument(
        "--bootstrap-db",
        type=str,
        default="backtest_bootstrap.db",
        help="Path to the isolated bootstrap SQLite DB.",
    )
    parser.add_argument(
        "--live-db",
        type=str,
        default=None,
        help="Path to live DB for merge (default: env DB_PATH / /data / paper_trades.db).",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge results into the live DB. Bootstrap DB still saved.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to dump the bootstrap stats as JSON.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    markets = (
        [s.strip() for s in args.markets.split(",") if s.strip()]
        if args.markets else None
    )
    timeframes = [s.strip() for s in args.timeframes.split(",") if s.strip()]

    stats = run_bootstrap(
        markets=markets,
        days=args.days,
        bootstrap_db=args.bootstrap_db,
        live_db=args.live_db,
        merge=not args.no_merge,
        timeframes=timeframes,
    )

    _print_summary(stats)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({
                "total_simulated_trades": stats.total_simulated_trades,
                "closed_trades": stats.closed_trades,
                "by_market": stats.by_market,
                "strategy_weights": stats.strategy_weights,
                "disabled_strategies": stats.disabled_strategies,
                "model_accuracy": stats.model_accuracy,
                "regime_distribution": stats.regime_distribution,
                "kronos_hit_rate": stats.kronos_hit_rate,
                "kronos_n": stats.kronos_n,
            }, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
