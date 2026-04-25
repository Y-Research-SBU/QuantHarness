"""
REL-376: Kronos horizon sweep across intraday timeframes.

For each (timeframe, horizon) combination, evaluate Kronos forecasts on
BTC-USD, ETH-USD, SOL-USD, SPY, AAPL using yfinance data, walking
forward and comparing predicted close at t+h vs actual close at t+h.

Output: aggregate metrics per (tf, horizon) cell.

Optimization: for each (symbol, timeframe, evaluation_point), we run
ONE Kronos prediction at the maximum horizon for that timeframe, and
read off intermediate horizons from the same predicted path.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kronos_agent import KronosForecastAgent  # noqa: E402
from data_fetcher import fetch_market_data  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("kronos_sweep")
logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────
# Sweep config
# ─────────────────────────────────────────────────────────────────────

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "AAPL"]

# (timeframe, list_of_horizons). Horizons must be sorted; we run prediction
# at the max horizon and read intermediate horizons from the same path.
TF_HORIZONS: Dict[str, List[int]] = {
    "1h": [4, 6, 8, 12, 24],
    "4h": [3, 6, 8, 12],
    "15m": [12, 24, 48],
    "5m": [24, 48],
}

# Per-tf data window (yfinance limits + 90-day target).
# yfinance: 1m=7d, 5m/15m/30m=60d, 1h/4h/1d=730d.
TF_LOOKBACK_DAYS: Dict[str, int] = {
    "1h": 90,
    "4h": 180,   # need more bars at 4h to get many evaluation points
    "15m": 55,   # under 60d cap
    "5m": 55,    # under 60d cap
}

# Context bars given to Kronos. Predictor max_context = 512.
CONTEXT_BARS = 400

# Number of evaluation points per (symbol, timeframe). We sample
# evenly-spaced points so Kronos sees diverse market regimes.
EVAL_POINTS_PER_SYMBOL = 30

# Crypto trades 24/7; equities don't. SPY/AAPL won't have many bars
# at sub-daily timeframes (≈ 6.5h × 5d/week). We accept smaller n there.

OUT_PATH = Path(__file__).resolve().parent.parent / "kronos_horizon_sweep_2026-04-25.md"
RAW_JSON = Path(__file__).resolve().parent.parent / "kronos_horizon_sweep_2026-04-25.json"


# ─────────────────────────────────────────────────────────────────────
# Per-prediction record
# ─────────────────────────────────────────────────────────────────────


@dataclass
class PredRow:
    symbol: str
    timeframe: str
    eval_idx: int           # bar index of the prediction time t
    horizon: int            # h
    last_close: float
    predicted_close: float  # at t+h
    actual_close: float     # at t+h
    confidence: float
    pred_pct: float
    actual_pct: float
    pred_dir: str           # UP / DOWN / FLAT
    actual_dir: str
    direction_hit: bool


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _direction(pct: float, eps: float = 0.0) -> str:
    if pct > eps:
        return "UP"
    if pct < -eps:
        return "DOWN"
    return "FLAT"


def fetch_full_history(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch as many bars as possible for the lookback window."""
    days = TF_LOOKBACK_DAYS[timeframe]
    end = datetime.now()
    start = end - timedelta(days=days)
    df = fetch_market_data(
        symbol=symbol,
        interval=timeframe,
        start_date=start,
        end_date=end,
        use_cache=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    df.attrs["timeframe"] = timeframe
    return df


def evaluation_indices(n_bars: int, max_h: int, n_points: int) -> List[int]:
    """Pick n_points indices evenly distributed in [CONTEXT_BARS, n_bars - max_h - 1]."""
    lo = CONTEXT_BARS
    hi = n_bars - max_h - 1
    if hi <= lo:
        return []
    if n_points >= (hi - lo):
        return list(range(lo, hi))
    step = max(1, (hi - lo) // n_points)
    return list(range(lo, hi, step))[:n_points]


# ─────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────


def run_sweep() -> List[PredRow]:
    rows: List[PredRow] = []
    agent = KronosForecastAgent(default_horizon=48, sample_count=1)

    # Force load now so failures are loud.
    predictor = agent._get_predictor()
    if predictor is None:
        raise RuntimeError(
            f"Kronos predictor failed to load: {KronosForecastAgent._LOAD_ERROR}"
        )

    for tf, horizons in TF_HORIZONS.items():
        max_h = max(horizons)
        logger.info("─" * 60)
        logger.info("Timeframe %s — horizons %s — max_h %d", tf, horizons, max_h)
        for symbol in SYMBOLS:
            try:
                full_df = fetch_full_history(symbol, tf)
            except Exception as exc:
                logger.warning("fetch failed %s %s: %s", symbol, tf, exc)
                continue
            if full_df.empty or len(full_df) < CONTEXT_BARS + max_h + 5:
                logger.info(
                    "  %s skipped: only %d bars (need ≥ %d)",
                    symbol,
                    len(full_df),
                    CONTEXT_BARS + max_h + 5,
                )
                continue

            indices = evaluation_indices(len(full_df), max_h, EVAL_POINTS_PER_SYMBOL)
            logger.info(
                "  %s: %d bars, %d eval points",
                symbol,
                len(full_df),
                len(indices),
            )

            for eval_idx in indices:
                # Context = bars[0 : eval_idx+1] (inclusive of t)
                ctx = full_df.iloc[: eval_idx + 1].copy()
                ctx.attrs["timeframe"] = tf
                last_close = float(ctx["Close"].iloc[-1])
                # Run a single prediction at max_h.
                try:
                    forecast = agent.predict(ctx, horizon=max_h, timeframe=tf)
                except Exception as exc:
                    logger.warning(
                        "predict failed %s %s idx=%d: %s", symbol, tf, eval_idx, exc
                    )
                    continue
                if forecast.source != "kronos":
                    # fallback path — skip, we want to evaluate Kronos itself
                    continue
                path = forecast.metadata.get("predicted_path") or []
                if len(path) < max_h:
                    continue

                for h in horizons:
                    pred_close = float(path[h - 1])
                    actual_idx = eval_idx + h
                    if actual_idx >= len(full_df):
                        continue
                    actual_close = float(full_df["Close"].iloc[actual_idx])
                    pred_pct = (pred_close - last_close) / last_close * 100.0
                    actual_pct = (actual_close - last_close) / last_close * 100.0
                    pred_dir = _direction(pred_pct)
                    actual_dir = _direction(actual_pct)
                    hit = (pred_dir == actual_dir) and pred_dir != "FLAT"
                    # If pred is FLAT, count as miss (no signal); if actual is
                    # FLAT, also miss. We want strict directional accuracy.
                    if pred_dir == "FLAT" or actual_dir == "FLAT":
                        hit = False

                    rows.append(
                        PredRow(
                            symbol=symbol,
                            timeframe=tf,
                            eval_idx=eval_idx,
                            horizon=h,
                            last_close=last_close,
                            predicted_close=pred_close,
                            actual_close=actual_close,
                            confidence=float(forecast.confidence),
                            pred_pct=pred_pct,
                            actual_pct=actual_pct,
                            pred_dir=pred_dir,
                            actual_dir=actual_dir,
                            direction_hit=hit,
                        )
                    )

    return rows


# ─────────────────────────────────────────────────────────────────────
# Aggregation + reporting
# ─────────────────────────────────────────────────────────────────────


def aggregate(rows: List[PredRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in rows])
    grp = df.groupby(["timeframe", "horizon"])

    agg = grp.apply(
        lambda g: pd.Series({
            "n": len(g),
            "hit_rate": g["direction_hit"].mean(),
            "mean_pred_pct": g["pred_pct"].mean(),
            "mean_actual_pct": g["actual_pct"].mean(),
            "mean_signed_error": (g["pred_pct"] - g["actual_pct"]).mean(),
            "mean_abs_error": (g["pred_pct"] - g["actual_pct"]).abs().mean(),
            "pred_down_pct": (g["pred_dir"] == "DOWN").mean(),
            "pred_up_pct": (g["pred_dir"] == "UP").mean(),
            "actual_down_pct": (g["actual_dir"] == "DOWN").mean(),
            "actual_up_pct": (g["actual_dir"] == "UP").mean(),
        })
    ).reset_index()

    # Bearish bias %: how much more often Kronos predicts DOWN than reality.
    agg["bearish_bias_pct"] = (agg["pred_down_pct"] - agg["actual_down_pct"]) * 100.0
    return agg


def format_report(agg: pd.DataFrame, n_rows: int) -> str:
    lines: List[str] = []
    lines.append("# Kronos Horizon Sweep — REL-376 (2026-04-25)")
    lines.append("")
    lines.append(
        "Investigation of whether Kronos can be salvaged on intraday timeframes "
        "with shorter horizons. Prior finding: 1d h=5 trustworthy (54.55%), "
        "1h h=24 broken (27.9%, systematic bearish bias)."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        f"For each (symbol ∈ {SYMBOLS}, timeframe, evaluation point) we fed Kronos "
        f"a context of up to {CONTEXT_BARS} bars and asked it to predict the "
        "maximum horizon for that timeframe. Intermediate horizons were read "
        "off the same predicted path (no recomputation, no look-ahead). "
        f"Eval points: {EVAL_POINTS_PER_SYMBOL} evenly spaced per (symbol, tf)."
    )
    lines.append("")
    lines.append(f"Total prediction rows: **{n_rows}**")
    lines.append("")

    if agg.empty:
        lines.append("⚠️ No data collected.")
        return "\n".join(lines)

    # Heatmap-style table
    lines.append("## Heatmap: hit-rate by (timeframe, horizon)")
    lines.append("")
    lines.append("| timeframe | horizon | n | hit_rate | mean_pred_% | mean_actual_% | signed_error | bearish_bias_% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in agg.iterrows():
        marker = ""
        if r["n"] >= 50 and r["hit_rate"] >= 0.45:
            marker = " ✅"
        elif r["hit_rate"] < 0.40:
            marker = " ❌"
        lines.append(
            f"| {r['timeframe']} | {int(r['horizon'])} | {int(r['n'])} | "
            f"{r['hit_rate']*100:.1f}%{marker} | "
            f"{r['mean_pred_pct']:+.2f}% | {r['mean_actual_pct']:+.2f}% | "
            f"{r['mean_signed_error']:+.2f}% | {r['bearish_bias_pct']:+.1f}% |"
        )
    lines.append("")

    # Per-tf recommendation
    lines.append("## Recommendation per timeframe")
    lines.append("")
    for tf in TF_HORIZONS.keys():
        sub = agg[agg["timeframe"] == tf]
        if sub.empty:
            lines.append(f"- **{tf}**: no data → untrustworthy")
            continue
        passed = sub[(sub["n"] >= 50) & (sub["hit_rate"] >= 0.45)]
        if not passed.empty:
            best = passed.sort_values("hit_rate", ascending=False).iloc[0]
            lines.append(
                f"- **{tf}**: trust at horizon={int(best['horizon'])} "
                f"({best['hit_rate']*100:.1f}% over n={int(best['n'])})"
            )
        else:
            best = sub.sort_values("hit_rate", ascending=False).iloc[0]
            lines.append(
                f"- **{tf}**: untrustworthy — best is h={int(best['horizon'])} "
                f"@ {best['hit_rate']*100:.1f}% (n={int(best['n'])})"
            )
    lines.append("")

    # Bearish bias note
    lines.append("## Systematic bias")
    lines.append("")
    lines.append(
        "If `bearish_bias_%` is large positive, Kronos predicted DOWN more often "
        "than actually happened (the 1h h=24 failure mode). Magnitudes near 0 "
        "imply directional balance."
    )
    return "\n".join(lines)


def main() -> int:
    rows = run_sweep()
    logger.info("Collected %d prediction rows", len(rows))

    # Persist raw JSON
    with RAW_JSON.open("w") as fh:
        json.dump([asdict(r) for r in rows], fh, indent=2, default=str)

    agg = aggregate(rows)
    report = format_report(agg, len(rows))
    OUT_PATH.write_text(report)

    print(report)
    print()
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {RAW_JSON}")

    if not agg.empty:
        passed = agg[(agg["n"] >= 50) & (agg["hit_rate"] >= 0.45)]
        if not passed.empty:
            print()
            print("PASSED CELLS:")
            print(passed[["timeframe", "horizon", "n", "hit_rate"]].to_string(index=False))
        else:
            print()
            print("NO CELL PASSED ≥45% / n≥50 — Kronos remains restricted to 1d.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
