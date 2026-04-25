"""Portfolio-level analyzer: aggregate trades across all (strategy, symbol)
backtests into a single equity curve and compute Sharpe / DD / win-rate /
profit factor for the portfolio as a whole.

Optionally restrict to a whitelist of strategies via --strategies.
Optionally drop strategy/symbol combos with avg Sharpe < min via --min-sharpe.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def _load(paths: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        out.extend(data["results"])
    return out


def _portfolio_metrics(
    rows: List[Dict[str, Any]],
    starting_capital: float = 10_000.0,
    strategy_whitelist: Optional[Iterable[str]] = None,
    drop_combos: Optional[Iterable[tuple]] = None,
) -> Dict[str, float]:
    """
    Time-bucketed portfolio aggregation.

    Trades from many (strategy, symbol) combos are bucketed by their *exit
    date* (UTC day) and summed into a daily portfolio PnL. The resulting
    equity curve has all trades that closed on the same day netted together,
    which models concurrent multi-strategy trading more realistically than
    sequencing every trade in isolation.
    """
    import pandas as pd

    drop_combos = set(drop_combos or [])
    whitelist = set(strategy_whitelist or [])

    trades: List[Dict[str, Any]] = []
    n_runs = 0
    for r in rows:
        if whitelist and r["strategy"] not in whitelist:
            continue
        if (r["strategy"], r["symbol"]) in drop_combos:
            continue
        n_runs += 1
        for t in r.get("trades", []):
            if t.get("exit_time"):
                trades.append({
                    "exit_time": t["exit_time"],
                    "pnl": float(t["pnl"]),
                })

    if not trades:
        return {"n_runs": n_runs, "n_trades": 0, "sharpe": 0.0,
                "win_rate": 0.0, "max_dd": 0.0, "profit_factor": 0.0,
                "total_return": 0.0, "ending_capital": starting_capital}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Treat the portfolio as ``n_runs`` × ``starting_capital`` of total
    # capital — each (strategy, symbol) combo is funded equally. Sum daily
    # PnLs and divide by total capital to get a return series that has the
    # right *shape* and the right *scale* for a multi-strategy portfolio.
    portfolio_capital = max(n_runs, 1) * starting_capital
    df = pd.DataFrame(trades)
    df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["exit_dt"])
    df["exit_date"] = df["exit_dt"].dt.date

    daily_pnl = df.groupby("exit_date")["pnl"].sum().sort_index()
    equity = portfolio_capital + daily_pnl.cumsum()
    equity_arr = np.concatenate([[portfolio_capital], equity.values])

    daily_returns = np.diff(equity_arr) / np.where(equity_arr[:-1] == 0, 1.0, equity_arr[:-1])
    mu = float(np.mean(daily_returns))
    sigma = float(np.std(daily_returns))
    sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else 0.0

    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / np.where(running_max == 0, 1.0, running_max)
    max_dd = float(abs(np.min(drawdowns)) * 100)

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    return {
        "n_runs": n_runs,
        "n_trades": len(trades),
        "sharpe": float(sharpe),
        "win_rate": len(wins) / len(trades),
        "max_dd": max_dd,
        "profit_factor": pf if np.isfinite(pf) else 0.0,
        "total_return": float((equity_arr[-1] - portfolio_capital) / portfolio_capital * 100.0),
        "ending_capital": float(equity_arr[-1]),
        "portfolio_capital": float(portfolio_capital),
        "n_days": int(len(daily_pnl)),
    }


def _per_strategy_combos(rows: List[Dict[str, Any]],
                         min_sharpe: float = 0.0) -> List[tuple]:
    """Return combos with avg sharpe < min_sharpe across the row set."""
    by_combo: Dict[tuple, float] = {}
    for r in rows:
        by_combo[(r["strategy"], r["symbol"])] = float(r["sharpe_ratio"])
    return [k for k, v in by_combo.items() if v < min_sharpe]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--strategies", nargs="*",
                    help="Whitelist of strategies to include.")
    ap.add_argument("--drop-negative", action="store_true",
                    help="Drop (strategy, symbol) combos with negative Sharpe.")
    ap.add_argument("--starting-capital", type=float, default=10_000.0)
    ap.add_argument("--min-sharpe", type=float, default=0.0,
                    help="With --drop-negative, minimum sharpe to keep a combo.")
    args = ap.parse_args()

    rows = _load(args.files)

    print(f"\nPortfolio analysis from {len(rows)} runs\n")

    # All combos
    metrics_all = _portfolio_metrics(rows, args.starting_capital,
                                     strategy_whitelist=args.strategies)
    print("ALL combos:")
    for k, v in metrics_all.items():
        if isinstance(v, float):
            print(f"  {k:<18} {v:+.4f}")
        else:
            print(f"  {k:<18} {v}")

    if args.drop_negative:
        drop = _per_strategy_combos(rows, args.min_sharpe)
        metrics_filtered = _portfolio_metrics(rows, args.starting_capital,
                                              strategy_whitelist=args.strategies,
                                              drop_combos=drop)
        print(f"\nFiltered (drop {len(drop)} combos with sharpe < {args.min_sharpe}):")
        for k, v in metrics_filtered.items():
            if isinstance(v, float):
                print(f"  {k:<18} {v:+.4f}")
            else:
                print(f"  {k:<18} {v}")


if __name__ == "__main__":
    main()
