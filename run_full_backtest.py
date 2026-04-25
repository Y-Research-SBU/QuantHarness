"""
Driver script for the comprehensive 68-symbol backtest.

Runs the 7 fast (non-Kronos) strategies on all 68 symbols at 5-year daily
resolution, then optionally adds Kronos strategies on a small representative
subset (Kronos is ~68s/symbol/strategy and intractable on the full universe).
Saves a single JSON file under backtest_results/ for later analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Sequence

from backtest import Backtester, BacktestResult, save_results
from data_fetcher import fetch_market_data
from market_config import StrategyType
from strategies import STRATEGIES

logger = logging.getLogger(__name__)


ALL_SYMBOLS = [
    # crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "AVAX-USD",
    "LINK-USD", "DOT-USD", "ATOM-USD", "AAVE-USD", "OP-USD", "SEI-USD",
    "TIA-USD", "NEAR-USD", "FET-USD", "WLD-USD", "AR-USD", "PENDLE-USD",
    "ENA-USD", "ONDO-USD", "DYDX-USD", "RUNE-USD", "EIGEN-USD", "WIF-USD",
    "BONK-USD", "FLOKI-USD", "SHIB-USD", "TURBO-USD", "LDO-USD", "MKR-USD",
    "CRV-USD", "HBAR-USD",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV",
    # mega-cap stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD",
    "AVGO", "TSM", "ARM", "MRVL", "COIN", "MSTR", "PLTR", "SOFI",
    "SNOW", "NET", "CRWD", "JPM", "GS", "XOM", "LLY", "BA",
    # commodities + forex
    "GC=F", "CL=F", "EURUSD=X", "GBPUSD=X",
]

NON_KRONOS_STRATEGIES = [
    StrategyType.MOMENTUM,
    StrategyType.MEAN_REVERSION,
    StrategyType.BREAKOUT,
    StrategyType.MULTI_FACTOR,
    StrategyType.VWAP_REVERSION,
    StrategyType.BB_SQUEEZE,
    StrategyType.EMA_CROSSOVER,
]

KRONOS_STRATEGIES = [
    StrategyType.KRONOS_MOMENTUM_CONFIRM,
    StrategyType.KRONOS_DIVERGENCE,
    StrategyType.MULTI_TIMEFRAME_KRONOS,
]

# Representative subset for Kronos (one per asset class)
KRONOS_SUBSET = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "AAPL", "NVDA", "GC=F", "EURUSD=X"]


def run_for_symbols(
    symbols: Sequence[str],
    strategies: Sequence[StrategyType],
    years: float,
    interval: str,
    backtester: Backtester,
) -> List[BacktestResult]:
    """Fetch data once per symbol, run every strategy on it."""
    end = datetime.utcnow()
    start = end - timedelta(days=int(years * 365))

    results: List[BacktestResult] = []
    for i, sym in enumerate(symbols, 1):
        t_fetch = time.time()
        df = fetch_market_data(sym, interval=interval, start_date=start, end_date=end)
        if df is None or df.empty or len(df) < 100:
            print(f"[{i}/{len(symbols)}] {sym:<12} SKIP (no data, len={0 if df is None else len(df)})")
            continue
        print(f"[{i}/{len(symbols)}] {sym:<12} bars={len(df)} fetched in {time.time()-t_fetch:.1f}s", flush=True)

        for st in strategies:
            strategy = STRATEGIES.get(st)
            if strategy is None:
                continue
            t0 = time.time()
            try:
                res = backtester.run(df.copy(), strategy, symbol=sym, timeframe=interval)
            except Exception as exc:
                print(f"    {st.value:32s} ERROR: {exc}", flush=True)
                continue
            results.append(res)
            print(
                f"    {st.value:32s} trades={res.total_trades:4d} "
                f"wr={res.win_rate*100:5.1f}% ret={res.total_return_pct:+7.2f}% "
                f"sharpe={res.sharpe_ratio:+5.2f} mdd={res.max_drawdown_pct:5.2f}% "
                f"pf={res.profit_factor:5.2f} ({time.time()-t0:.1f}s)",
                flush=True,
            )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--include-kronos", action="store_true",
                        help="Also run Kronos strategies on the small representative subset")
    parser.add_argument("--kronos-only", action="store_true",
                        help="Run ONLY Kronos strategies (on subset)")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Override symbol list (default: all 68)")
    parser.add_argument("--tag", default="full",
                        help="Tag added to the output JSON filename")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    symbols = args.symbols if args.symbols else ALL_SYMBOLS

    bt = Backtester(starting_capital=10_000.0)
    all_results: List[BacktestResult] = []

    if not args.kronos_only:
        print("=" * 80)
        print(f"PHASE: Non-Kronos strategies on {len(symbols)} symbols ({args.interval}, {args.years}y)")
        print("=" * 80, flush=True)
        all_results.extend(
            run_for_symbols(symbols, NON_KRONOS_STRATEGIES, args.years, args.interval, bt)
        )

    if args.include_kronos or args.kronos_only:
        kronos_syms = symbols if args.kronos_only else KRONOS_SUBSET
        print()
        print("=" * 80)
        print(f"PHASE: Kronos strategies on {len(kronos_syms)} symbols")
        print("=" * 80, flush=True)
        all_results.extend(
            run_for_symbols(kronos_syms, KRONOS_STRATEGIES, args.years, args.interval, bt)
        )

    # Save
    os.makedirs("backtest_results", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("backtest_results", f"backtest_{args.tag}_{timestamp}.json")
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "args": vars(args),
        "results": [r.to_dict() for r in all_results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
