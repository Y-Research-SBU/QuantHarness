"""
Parameter adaptation (Level 2 of the self-improvement system).

Stores per-strategy, per-symbol tuned parameters in the ``adaptive_params``
table and falls back to ``DEFAULT_PARAMS`` when not enough evidence exists.

Design:
    - ``DEFAULT_PARAMS`` holds the system-wide baseline.
    - ``optimize_rsi_thresholds`` scans candidate (overbought, oversold)
      combinations on historical bars and picks the combo with the best
      simulated expectancy.
    - ``optimize_stop_distances`` scans ATR multipliers on closed trades and
      picks the combo with the best risk-adjusted return.
    - ``get_params`` returns the most recent stored value per param name,
      falling back to the default.
    - All values are clamped to safe bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from db_schema import get_connection
from self_improvement_schema import apply_self_improvement_schema


DEFAULT_PARAMS: Dict[str, float] = {
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 2.0,
    "kronos_min_confidence": 0.6,
    "min_signal_strength": 0.3,
}


# Hard safety bounds — optimizer may never return a value outside these.
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "rsi_overbought": (50.0, 90.0),
    "rsi_oversold": (10.0, 50.0),
    "sl_atr_mult": (0.5, 3.0),
    "tp_atr_mult": (0.5, 5.0),
    "kronos_min_confidence": (0.3, 0.95),
    "min_signal_strength": (0.0, 1.0),
}


def _clamp(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS.get(name, (-1e18, 1e18))
    return float(max(lo, min(hi, value)))


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic RSI (Wilder smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    alt = name.lower() if name[0].isupper() else name.capitalize()
    if alt in df.columns:
        return df[alt]
    raise KeyError(f"Column {name!r} / {alt!r} not in DataFrame")


class AdaptiveParams:
    """Look up + persist tuned parameter values."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path
        apply_self_improvement_schema(db_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_params(self, strategy: str, symbol: str) -> Dict[str, float]:
        """Return merged params: defaults overlaid with latest stored values."""
        params = dict(DEFAULT_PARAMS)
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT param_name, param_value
                FROM adaptive_params
                WHERE strategy = ? AND symbol = ?
                ORDER BY optimized_at DESC, id DESC
                """,
                (strategy, symbol),
            ).fetchall()
        seen: set = set()
        for row in rows:
            name = row["param_name"]
            if name in seen:
                continue
            seen.add(name)
            params[name] = _clamp(name, float(row["param_value"]))
        return params

    def set_param(
        self,
        strategy: str,
        symbol: str,
        param_name: str,
        param_value: float,
        sample_size: int = 0,
        improvement_pct: float = 0.0,
    ) -> None:
        """Persist a new param value (clamped to safe bounds)."""
        value = _clamp(param_name, float(param_value))
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO adaptive_params
                    (strategy, symbol, param_name, param_value, sample_size, improvement_pct)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (strategy, symbol, param_name, value, int(sample_size), float(improvement_pct)),
            )

    # ------------------------------------------------------------------
    # Optimization — RSI thresholds for mean reversion
    # ------------------------------------------------------------------

    def optimize_rsi_thresholds(
        self,
        bars: pd.DataFrame,
        overbought_grid: Iterable[float] = (65, 70, 75, 80, 85),
        oversold_grid: Iterable[float] = (15, 20, 25, 30, 35),
        forward_bars: int = 5,
    ) -> Dict[str, float]:
        """Sweep (overbought, oversold) pairs and pick the most profitable.

        Simple simulation: whenever RSI crosses above ``overbought``, assume a
        SHORT entry and evaluate return ``forward_bars`` later. Likewise for
        oversold → LONG. Picks the combo with the highest total simulated
        return across both sides.
        """
        if bars is None or len(bars) < 50:
            return {
                "rsi_overbought": DEFAULT_PARAMS["rsi_overbought"],
                "rsi_oversold": DEFAULT_PARAMS["rsi_oversold"],
            }

        close = _col(bars, "Close").astype(float).reset_index(drop=True)
        rsi = _rsi_series(close).reset_index(drop=True)

        best = {
            "overbought": DEFAULT_PARAMS["rsi_overbought"],
            "oversold": DEFAULT_PARAMS["rsi_oversold"],
            "score": -np.inf,
        }

        for ob in overbought_grid:
            for os in oversold_grid:
                if os >= ob:
                    continue
                # SHORT entries when RSI crosses above ob
                short_entries = (rsi.shift(1) <= ob) & (rsi > ob)
                # LONG entries when RSI crosses below os
                long_entries = (rsi.shift(1) >= os) & (rsi < os)

                total_return = 0.0
                n_trades = 0
                for idx in np.where(short_entries)[0]:
                    exit_idx = idx + forward_bars
                    if exit_idx >= len(close):
                        continue
                    entry_p = close.iloc[idx]
                    exit_p = close.iloc[exit_idx]
                    if entry_p <= 0:
                        continue
                    total_return += (entry_p - exit_p) / entry_p
                    n_trades += 1
                for idx in np.where(long_entries)[0]:
                    exit_idx = idx + forward_bars
                    if exit_idx >= len(close):
                        continue
                    entry_p = close.iloc[idx]
                    exit_p = close.iloc[exit_idx]
                    if entry_p <= 0:
                        continue
                    total_return += (exit_p - entry_p) / entry_p
                    n_trades += 1

                if n_trades < 3:
                    continue
                score = total_return / n_trades
                if score > best["score"]:
                    best = {"overbought": ob, "oversold": os, "score": score}

        return {
            "rsi_overbought": _clamp("rsi_overbought", float(best["overbought"])),
            "rsi_oversold": _clamp("rsi_oversold", float(best["oversold"])),
        }

    # ------------------------------------------------------------------
    # Optimization — stop / take-profit ATR multipliers
    # ------------------------------------------------------------------

    def optimize_stop_distances(
        self,
        trades: List[Dict[str, Any]],
        sl_grid: Iterable[float] = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
        tp_grid: Iterable[float] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
    ) -> Dict[str, float]:
        """Pick SL/TP ATR multipliers that maximize Sharpe on the trade set.

        Each trade must contain ``pnl_pct`` and ``decision_json`` with keys
        ``atr`` and ``entry_price`` (or just ``atr_pct`` as a fallback).
        Trades missing this info are skipped. We simulate the same signals
        under different multipliers by scaling the realized % return by the
        ratio of (new SL distance) / (actual stop distance).

        When no data is available, returns defaults.
        """
        if not trades:
            return {
                "sl_atr_mult": DEFAULT_PARAMS["sl_atr_mult"],
                "tp_atr_mult": DEFAULT_PARAMS["tp_atr_mult"],
            }

        samples: List[Tuple[float, float, float]] = []  # (atr_pct, realized_return, sign)
        for t in trades:
            pnl_pct = t.get("pnl_pct")
            if pnl_pct is None:
                continue
            atr_pct = t.get("atr_pct")
            if atr_pct is None:
                # Try decoding from decision_json
                md = t.get("metadata") or {}
                atr_pct = md.get("atr_pct")
                if atr_pct is None and md.get("atr") and md.get("entry_price"):
                    try:
                        atr_pct = float(md["atr"]) / float(md["entry_price"])
                    except (TypeError, ValueError, ZeroDivisionError):
                        atr_pct = None
            if atr_pct is None or atr_pct <= 0:
                continue
            samples.append((float(atr_pct), float(pnl_pct), 1.0 if pnl_pct >= 0 else -1.0))

        if len(samples) < 5:
            return {
                "sl_atr_mult": DEFAULT_PARAMS["sl_atr_mult"],
                "tp_atr_mult": DEFAULT_PARAMS["tp_atr_mult"],
            }

        best = {"sl": DEFAULT_PARAMS["sl_atr_mult"], "tp": DEFAULT_PARAMS["tp_atr_mult"], "score": -np.inf}
        for sl in sl_grid:
            for tp in tp_grid:
                if tp <= sl:
                    continue
                returns: List[float] = []
                for atr_pct, pnl_pct, _sign in samples:
                    # Cap the realized pnl_pct by the new SL/TP bands.
                    max_gain = tp * atr_pct
                    max_loss = -sl * atr_pct
                    capped = max(min(pnl_pct, max_gain), max_loss)
                    returns.append(capped)
                arr = np.asarray(returns)
                if arr.std(ddof=1) == 0 or np.isnan(arr.std(ddof=1)):
                    continue
                sharpe = arr.mean() / arr.std(ddof=1) * np.sqrt(len(arr))
                if sharpe > best["score"]:
                    best = {"sl": sl, "tp": tp, "score": sharpe}

        return {
            "sl_atr_mult": _clamp("sl_atr_mult", float(best["sl"])),
            "tp_atr_mult": _clamp("tp_atr_mult", float(best["tp"])),
        }

    # ------------------------------------------------------------------
    # High-level optimize — stash results in DB
    # ------------------------------------------------------------------

    def optimize(
        self,
        strategy: str,
        symbol: str,
        trades: Optional[List[Dict[str, Any]]] = None,
        bars: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Run all optimizers relevant to this strategy and persist results.

        Returns the final merged params for the (strategy, symbol) pair.
        """
        trades = trades or []
        changes: Dict[str, float] = {}

        if bars is not None and len(bars) >= 50:
            rsi_out = self.optimize_rsi_thresholds(bars)
            for name, value in rsi_out.items():
                self.set_param(strategy, symbol, name, value, sample_size=len(bars))
                changes[name] = value

        if trades:
            stop_out = self.optimize_stop_distances(trades)
            for name, value in stop_out.items():
                self.set_param(strategy, symbol, name, value, sample_size=len(trades))
                changes[name] = value

        return self.get_params(strategy, symbol)
