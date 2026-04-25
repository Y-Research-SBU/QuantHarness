"""Export the best (strategy, symbol) combos sorted by Sharpe.

Usage: python backtest_results/best_combos.py FILE1.json [FILE2.json ...]
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List


def main(paths: List[str]) -> None:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with open(p) as fh:
            rows.extend(json.load(fh)["results"])

    rows.sort(key=lambda r: r["sharpe_ratio"], reverse=True)

    print(f"\n{'Rank':>4} {'Strategy':<26} {'Symbol':<10} "
          f"{'Trades':>7} {'WR%':>6} {'Sharpe':>8} {'Ret%':>8} "
          f"{'MDD%':>7} {'PF':>6}")
    print("-" * 95)

    rank = 0
    for r in rows:
        if r["total_trades"] < 5:
            continue
        rank += 1
        if rank > 40:
            break
        print(f"{rank:>4} {r['strategy']:<26} {r['symbol']:<10} "
              f"{r['total_trades']:>7} {r['win_rate']*100:>6.1f} "
              f"{r['sharpe_ratio']:>+8.2f} {r['total_return_pct']:>+8.2f} "
              f"{r['max_drawdown_pct']:>7.2f} {r['profit_factor']:>6.2f}")

    print(f"\n--- Worst (sharpe < 0, with > 5 trades) ---")
    print(f"{'Strategy':<26} {'Symbol':<10} {'Trades':>7} {'WR%':>6} "
          f"{'Sharpe':>8} {'Ret%':>8}")
    print("-" * 80)
    bad = [r for r in rows if r["sharpe_ratio"] < 0 and r["total_trades"] >= 5]
    bad.sort(key=lambda r: r["sharpe_ratio"])
    for r in bad[:20]:
        print(f"{r['strategy']:<26} {r['symbol']:<10} "
              f"{r['total_trades']:>7} {r['win_rate']*100:>6.1f} "
              f"{r['sharpe_ratio']:>+8.2f} {r['total_return_pct']:>+8.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
