"""
Analyze a backtest results JSON and produce per-strategy / per-symbol /
strategy-symbol heatmap summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence


def _safe(v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def _category(symbol: str) -> str:
    if symbol.endswith("-USD"):
        return "crypto"
    if symbol in {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV"}:
        return "etf"
    if symbol.endswith("=F"):
        return "commodity"
    if symbol.endswith("=X"):
        return "forex"
    return "stock"


def per_strategy(results: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across all symbols for each strategy."""
    by_strat: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_strat[r["strategy"]].append(r)

    summary: Dict[str, Dict[str, float]] = {}
    for strat, items in by_strat.items():
        # Aggregate by pooling all trades across symbols.
        total_trades = sum(it["total_trades"] for it in items)
        wins = sum(it["winning_trades"] for it in items)
        # Pool returns
        returns = [_safe(it["total_return_pct"]) for it in items]
        sharpes = [_safe(it["sharpe_ratio"]) for it in items]
        mdds = [_safe(it["max_drawdown_pct"]) for it in items]
        pfs = [_safe(it["profit_factor"]) for it in items if _safe(it["profit_factor"]) > 0]
        # Symbols where this strategy made money
        profitable = sum(1 for it in items if _safe(it["total_return_pct"]) > 0)
        traded = sum(1 for it in items if it["total_trades"] > 0)

        summary[strat] = {
            "symbols": len(items),
            "symbols_traded": traded,
            "symbols_profitable": profitable,
            "total_trades": total_trades,
            "win_rate": wins / total_trades if total_trades else 0.0,
            "mean_return_pct": sum(returns) / len(returns) if returns else 0.0,
            "median_return_pct": sorted(returns)[len(returns) // 2] if returns else 0.0,
            "mean_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0.0,
            "median_sharpe": sorted(sharpes)[len(sharpes) // 2] if sharpes else 0.0,
            "pos_sharpe_pct": sum(1 for s in sharpes if s > 0) / len(sharpes) * 100.0 if sharpes else 0.0,
            "mean_mdd": sum(mdds) / len(mdds) if mdds else 0.0,
            "mean_pf": sum(pfs) / len(pfs) if pfs else 0.0,
        }
    return summary


def per_strategy_by_category(results: Sequence[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Heatmap: strategy -> category -> aggregate."""
    by: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by[r["strategy"]][_category(r["symbol"])].append(r)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for strat, cats in by.items():
        out[strat] = {}
        for cat, items in cats.items():
            total_trades = sum(it["total_trades"] for it in items)
            wins = sum(it["winning_trades"] for it in items)
            returns = [_safe(it["total_return_pct"]) for it in items]
            sharpes = [_safe(it["sharpe_ratio"]) for it in items]
            traded = sum(1 for it in items if it["total_trades"] > 0)
            out[strat][cat] = {
                "symbols": len(items),
                "symbols_traded": traded,
                "total_trades": total_trades,
                "win_rate": wins / total_trades if total_trades else 0.0,
                "mean_return_pct": sum(returns) / len(returns) if returns else 0.0,
                "mean_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0.0,
            }
    return out


def per_symbol(results: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    by_sym: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_sym[r["symbol"]].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for sym, items in by_sym.items():
        total_trades = sum(it["total_trades"] for it in items)
        wins = sum(it["winning_trades"] for it in items)
        returns = [_safe(it["total_return_pct"]) for it in items]
        sharpes = [_safe(it["sharpe_ratio"]) for it in items]
        out[sym] = {
            "category": _category(sym),
            "strategies": len(items),
            "total_trades": total_trades,
            "win_rate": wins / total_trades if total_trades else 0.0,
            "mean_return_pct": sum(returns) / len(returns) if returns else 0.0,
            "best_sharpe": max(sharpes) if sharpes else 0.0,
            "best_strategy": items[
                sharpes.index(max(sharpes))
            ]["strategy"] if sharpes else "",
        }
    return out


def portfolio_metrics(results: Sequence[Dict]) -> Dict[str, float]:
    """Equally weighted portfolio: average across all (sym,strat) cells."""
    returns = [_safe(it["total_return_pct"]) for it in results if it["total_trades"] > 0]
    sharpes = [_safe(it["sharpe_ratio"]) for it in results if it["total_trades"] > 0]
    mdds = [_safe(it["max_drawdown_pct"]) for it in results if it["total_trades"] > 0]
    n = len(returns)
    total_trades = sum(it["total_trades"] for it in results)
    wins = sum(it["winning_trades"] for it in results)
    pfs = [_safe(it["profit_factor"]) for it in results if it["total_trades"] > 0 and _safe(it["profit_factor"]) > 0]
    return {
        "active_cells": n,
        "total_trades": total_trades,
        "portfolio_win_rate": wins / total_trades if total_trades else 0.0,
        "mean_return_pct": sum(returns) / n if n else 0.0,
        "mean_sharpe": sum(sharpes) / n if n else 0.0,
        "median_sharpe": sorted(sharpes)[n // 2] if n else 0.0,
        "pct_positive_sharpe": sum(1 for s in sharpes if s > 0) / n * 100.0 if n else 0.0,
        "mean_mdd": sum(mdds) / n if n else 0.0,
        "mean_pf": sum(pfs) / len(pfs) if pfs else 0.0,
    }


def print_summary(path: str) -> None:
    with open(path) as f:
        d = json.load(f)
    results = d["results"]

    print(f"\n{'='*80}\nFile: {path}\nTotal cells: {len(results)}\n{'='*80}")

    pm = portfolio_metrics(results)
    print("\nPORTFOLIO (equally-weighted across active cells)")
    print(f"  Active cells:         {pm['active_cells']}")
    print(f"  Total trades:         {pm['total_trades']}")
    print(f"  Portfolio win rate:   {pm['portfolio_win_rate']*100:.1f}%")
    print(f"  Mean return per cell: {pm['mean_return_pct']:+.2f}%")
    print(f"  Mean Sharpe:          {pm['mean_sharpe']:+.2f}")
    print(f"  Median Sharpe:        {pm['median_sharpe']:+.2f}")
    print(f"  % cells +Sharpe:      {pm['pct_positive_sharpe']:.1f}%")
    print(f"  Mean MDD:             {pm['mean_mdd']:.2f}%")
    print(f"  Mean profit factor:   {pm['mean_pf']:.2f}")

    ps = per_strategy(results)
    print("\nPER-STRATEGY")
    header = f"  {'strategy':<28} {'syms':>4} {'trd':>4} {'prof':>4} {'trades':>7} {'wr%':>6} {'mret%':>7} {'mSh':>6} {'+Sh%':>5} {'mMDD':>6} {'mPF':>5}"
    print(header)
    for strat, m in sorted(ps.items(), key=lambda kv: -kv[1]["mean_sharpe"]):
        print(
            f"  {strat:<28} {m['symbols']:>4} {m['symbols_traded']:>4} "
            f"{m['symbols_profitable']:>4} {m['total_trades']:>7} "
            f"{m['win_rate']*100:>6.1f} {m['mean_return_pct']:>+7.2f} "
            f"{m['mean_sharpe']:>+6.2f} {m['pos_sharpe_pct']:>5.1f} "
            f"{m['mean_mdd']:>6.2f} {m['mean_pf']:>5.2f}"
        )

    print("\nPER-STRATEGY × CATEGORY (mean Sharpe)")
    sc = per_strategy_by_category(results)
    cats = sorted({c for v in sc.values() for c in v.keys()})
    print(f"  {'strategy':<28} " + " ".join(f"{c:>10}" for c in cats))
    for strat in sorted(sc.keys()):
        row = sc[strat]
        cells = []
        for c in cats:
            v = row.get(c)
            if v is None or v["symbols_traded"] == 0:
                cells.append("        --")
            else:
                cells.append(f"{v['mean_sharpe']:>+10.2f}")
        print(f"  {strat:<28} " + " ".join(cells))

    print("\nTOP 15 (sym, strat) cells by Sharpe")
    keep = [r for r in results if r["total_trades"] >= 5]
    keep.sort(key=lambda r: -_safe(r["sharpe_ratio"]))
    for r in keep[:15]:
        print(
            f"  {r['symbol']:<10} {r['strategy']:<28} "
            f"trades={r['total_trades']:>3} wr={r['win_rate']*100:5.1f}% "
            f"ret={r['total_return_pct']:+7.2f}% sharpe={r['sharpe_ratio']:+5.2f} "
            f"mdd={r['max_drawdown_pct']:5.2f}% pf={_safe(r['profit_factor']):.2f}"
        )

    print("\nWORST 15 (sym, strat) cells by Sharpe")
    keep.sort(key=lambda r: _safe(r["sharpe_ratio"]))
    for r in keep[:15]:
        print(
            f"  {r['symbol']:<10} {r['strategy']:<28} "
            f"trades={r['total_trades']:>3} wr={r['win_rate']*100:5.1f}% "
            f"ret={r['total_return_pct']:+7.2f}% sharpe={r['sharpe_ratio']:+5.2f} "
            f"mdd={r['max_drawdown_pct']:5.2f}% pf={_safe(r['profit_factor']):.2f}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args(argv)
    print_summary(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
