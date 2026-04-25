"""Aggregate per-strategy and per-symbol metrics across one or more backtest result JSONs.

Usage:
    python backtest_results/analyze.py FILE1.json [FILE2.json ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List


def _load(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        rows.extend(data["results"])
    return rows


def _agg_strategy(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_strategy: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "losses": 0,
                 "total_pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
                 "returns": [], "sharpes": [], "mdds": [], "n_runs": 0,
                 "n_loss_runs": 0, "best_symbols": [], "worst_symbols": []}
    )
    for r in rows:
        s = r["strategy"]
        b = by_strategy[s]
        b["n_runs"] += 1
        if r["sharpe_ratio"] < 0:
            b["n_loss_runs"] += 1
        b["trades"] += r["total_trades"]
        b["wins"] += r["winning_trades"]
        b["losses"] += r["losing_trades"]
        b["sharpes"].append(r["sharpe_ratio"])
        b["mdds"].append(r["max_drawdown_pct"])
        b["returns"].append(r["total_return_pct"])
        for t in r.get("trades", []):
            pnl = t["pnl"]
            b["total_pnl"] += pnl
            if pnl > 0:
                b["gross_win"] += pnl
            else:
                b["gross_loss"] += abs(pnl)
        # Track best/worst symbol returns for context
        b.setdefault("by_symbol", []).append((r["symbol"], r["total_return_pct"], r["sharpe_ratio"]))

    out: Dict[str, Dict[str, float]] = {}
    for s, b in by_strategy.items():
        wr = b["wins"] / b["trades"] if b["trades"] else 0.0
        pf = b["gross_win"] / b["gross_loss"] if b["gross_loss"] > 0 else (
            float("inf") if b["gross_win"] > 0 else 0.0)
        out[s] = {
            "n_runs": b["n_runs"],
            "n_loss_runs": b["n_loss_runs"],
            "trades": b["trades"],
            "win_rate": wr,
            "avg_sharpe": sum(b["sharpes"]) / len(b["sharpes"]) if b["sharpes"] else 0.0,
            "avg_return": sum(b["returns"]) / len(b["returns"]) if b["returns"] else 0.0,
            "avg_mdd": sum(b["mdds"]) / len(b["mdds"]) if b["mdds"] else 0.0,
            "profit_factor": pf if pf != float("inf") else 99.0,
            "total_pnl": b["total_pnl"],
            "by_symbol": b["by_symbol"],
        }
    return out


def main(paths: List[str]) -> None:
    rows = _load(paths)
    by_strategy = _agg_strategy(rows)

    print()
    print(f"Loaded {len(rows)} results from {len(paths)} files")
    print()
    print(f"{'Strategy':<28} {'Runs':>4} {'Trades':>6} {'WR':>6} "
          f"{'AvgSharpe':>10} {'AvgRet%':>8} {'AvgMDD%':>8} {'PF':>6} "
          f"{'NegRuns':>8} {'TotalPnl':>10}")
    print("-" * 110)
    rows_sorted = sorted(by_strategy.items(), key=lambda kv: kv[1]["avg_sharpe"], reverse=True)
    for s, m in rows_sorted:
        print(
            f"{s:<28} {m['n_runs']:>4} {m['trades']:>6} {m['win_rate']*100:>5.1f}% "
            f"{m['avg_sharpe']:>+10.3f} {m['avg_return']:>+7.2f}% {m['avg_mdd']:>7.2f}% "
            f"{m['profit_factor']:>6.2f} {m['n_loss_runs']:>8} {m['total_pnl']:>+10.2f}"
        )

    print()
    print("Per-strategy: best/worst symbols (by sharpe)")
    for s, m in rows_sorted:
        sym_sorted = sorted(m["by_symbol"], key=lambda x: x[2], reverse=True)
        best = sym_sorted[:3]
        worst = sym_sorted[-3:]
        print(f"\n  {s}:")
        print(f"    best:  {[f'{sym}({sh:+.2f})' for sym, _, sh in best]}")
        print(f"    worst: {[f'{sym}({sh:+.2f})' for sym, _, sh in worst]}")


if __name__ == "__main__":
    main(sys.argv[1:])
