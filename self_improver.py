"""
Self-improvement engine for QuantAgent — orchestrates Levels 1-5.

Levels:
    L1 — Strategy performance tracking (rolling Sharpe, weight, disable)
    L2 — Parameter adaptation (RSI thresholds, stop distances)
    L3 — Signal quality scoring (sklearn meta-model)
    L4 — Regime detection (ADX / ATR / BB / SMA slope)
    L5 — Kronos accuracy tracking (forecast hit-rate calibration)

This module is read-only to the existing trading code: it queries the
trades table, reads OHLCV frames passed to it, and writes only to the new
self-improvement tables. Integration with scanner / paper_trading happens
later via explicit calls to the methods here.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from adaptive_params import AdaptiveParams, DEFAULT_PARAMS, PARAM_BOUNDS
from db_schema import get_connection
from regime_detector import RegimeDetector
from self_improvement_schema import apply_self_improvement_schema
from signal_scorer import SignalScorer, extract_features, feature_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Rolling window for strategy scoring (most recent N closed trades per strategy).
ROLLING_WINDOW = 200

# Disable a strategy if sharpe < 0 after this many trades.
MIN_TRADES_FOR_DISABLE = 10

# Emergency disable: if win rate is below this after MIN_TRADES_FOR_DISABLE trades,
# disable regardless of sharpe (catches strategies that win big rarely but bleed)
EMERGENCY_DISABLE_WIN_RATE = 0.15

# Minimum win-rate floor for high-sharpe boost: a strategy must achieve at
# least this win rate to qualify for WEIGHT_HIGH_SHARPE.  Prevents a few
# outsized winners from masking a chronic losing streak.
HIGH_SHARPE_MIN_WIN_RATE = 0.25

# Zero-win emergency: disable if win rate is exactly 0% after this many trades
# (lower threshold than MIN_TRADES_FOR_DISABLE to catch total-loss strategies faster)
ZERO_WIN_MIN_TRADES = 5

# Strategy sharpe → weight mapping.
WEIGHT_HIGH_SHARPE = 2.0
WEIGHT_NORMAL = 1.0
WEIGHT_REDUCED = 0.5
WEIGHT_DISABLED = 0.0

# Level cadences (in # of closed trades between runs). Tightened so the
# 5m/15m crypto fan-out actually feeds the learning loop within hours
# rather than days.
CADENCE_L1 = 20
CADENCE_L2 = 50
CADENCE_L3 = 100
CADENCE_L5 = 200

# Regime × strategy affinity priors — used before we have enough data to
# learn them from outcomes.
DEFAULT_REGIME_AFFINITY: Dict[str, Dict[str, float]] = {
    "momentum": {"trending_up": 1.5, "trending_down": 1.5, "ranging": 0.4, "volatile": 0.7},
    "mean_reversion": {"trending_up": 0.3, "trending_down": 0.3, "ranging": 1.5, "volatile": 0.8},
    "breakout": {"trending_up": 1.2, "trending_down": 1.2, "ranging": 0.6, "volatile": 1.3},
    "multi_factor": {"trending_up": 1.0, "trending_down": 1.0, "ranging": 1.0, "volatile": 1.0},
    "kronos_momentum_confirm": {"trending_up": 1.4, "trending_down": 1.4, "ranging": 0.6, "volatile": 0.9},
    "kronos_divergence": {"trending_up": 0.7, "trending_down": 0.7, "ranging": 1.3, "volatile": 1.1},
    "multi_timeframe_kronos": {"trending_up": 1.2, "trending_down": 1.2, "ranging": 0.8, "volatile": 1.0},
}


@dataclass
class ImprovementCycleResult:
    """Summary of a single ``run_improvement_cycle`` invocation."""
    total_trades: int
    levels_run: List[str] = field(default_factory=list)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    disabled_strategies: List[str] = field(default_factory=list)
    params_updated: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_trained: bool = False
    model_accuracy: Optional[float] = None
    kronos_stats: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "levels_run": list(self.levels_run),
            "strategy_weights": dict(self.strategy_weights),
            "disabled_strategies": list(self.disabled_strategies),
            "params_updated": dict(self.params_updated),
            "model_trained": self.model_trained,
            "model_accuracy": self.model_accuracy,
            "kronos_stats": self.kronos_stats,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sharpe(returns: List[float]) -> float:
    """Annualized-agnostic Sharpe: mean / std * sqrt(n). Returns 0 for empty."""
    if not returns:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    std = arr.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(arr.mean() / std * math.sqrt(len(arr)))


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class SelfImprover:
    """Brain of the self-improvement system."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        rolling_window: int = ROLLING_WINDOW,
        min_trades_for_disable: int = MIN_TRADES_FOR_DISABLE,
    ) -> None:
        self.db_path = db_path
        self.rolling_window = rolling_window
        self.min_trades_for_disable = min_trades_for_disable

        # Ensure our own tables exist; never touches the trading schema.
        apply_self_improvement_schema(db_path)

        self.regime_detector = RegimeDetector()
        self.signal_scorer = SignalScorer()
        self.params = AdaptiveParams(db_path=db_path)
        self._last_cycle_triggers: Dict[str, int] = {"L1": 0, "L2": 0, "L3": 0, "L5": 0}

    # ==================================================================
    # LEVEL 1: Strategy performance tracking
    # ==================================================================

    def _fetch_closed_trades(self, strategy: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the most recent closed/stopped trades, newest first."""
        sql = """
            SELECT id, symbol, timeframe, strategy, direction, entry_price, exit_price,
                   position_size, pnl, pnl_pct, status, decision_json, entry_time, exit_time
            FROM trades
            WHERE status IN ('CLOSED', 'STOPPED')
        """
        args: List[Any] = []
        if strategy is not None:
            sql += " AND strategy = ?"
            args.append(strategy)
        sql += " ORDER BY COALESCE(exit_time, entry_time) DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        with get_connection(self.db_path) as conn:
            rows = conn.execute(sql, tuple(args)).fetchall()
        return [dict(r) for r in rows]

    def score_strategies(self) -> Dict[str, float]:
        """Return {strategy: Sharpe} over rolling window of closed trades."""
        with get_connection(self.db_path) as conn:
            strategies = [
                r["strategy"]
                for r in conn.execute(
                    "SELECT DISTINCT strategy FROM trades WHERE status IN ('CLOSED','STOPPED')"
                ).fetchall()
            ]
        out: Dict[str, float] = {}
        for strat in strategies:
            trades = self._fetch_closed_trades(strategy=strat, limit=self.rolling_window)
            if not trades:
                continue
            returns = [float(t.get("pnl_pct") or 0.0) for t in trades]
            out[strat] = _sharpe(returns)
        return out

    def get_strategy_trade_counts(self) -> Dict[str, int]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT strategy, COUNT(*) AS n
                FROM trades
                WHERE status IN ('CLOSED','STOPPED')
                GROUP BY strategy
                """
            ).fetchall()
        return {r["strategy"]: int(r["n"]) for r in rows}

    def get_strategy_weights(self) -> Dict[str, float]:
        """Sharpe → weight multiplier per strategy."""
        sharpes = self.score_strategies()
        counts = self.get_strategy_trade_counts()
        weights: Dict[str, float] = {}
        for strat, sharpe in sharpes.items():
            n = counts.get(strat, 0)
            # Compute rolling win rate for emergency check
            trades = self._fetch_closed_trades(strategy=strat, limit=self.rolling_window)
            wins = sum(1 for t in trades if float(t.get("pnl") or 0.0) > 0)
            win_rate = wins / len(trades) if trades else 0.0

            if n >= ZERO_WIN_MIN_TRADES and win_rate == 0.0:
                weights[strat] = WEIGHT_DISABLED
            elif sharpe < 0 and n >= self.min_trades_for_disable:
                weights[strat] = WEIGHT_DISABLED
            elif n >= self.min_trades_for_disable and win_rate < EMERGENCY_DISABLE_WIN_RATE:
                weights[strat] = WEIGHT_DISABLED
            elif sharpe > 1.5 and win_rate >= HIGH_SHARPE_MIN_WIN_RATE:
                weights[strat] = WEIGHT_HIGH_SHARPE
            elif sharpe > 1.5 and win_rate < HIGH_SHARPE_MIN_WIN_RATE:
                # High Sharpe but poor win rate — demote to normal weight
                # to avoid over-leveraging a few lucky big wins.
                weights[strat] = WEIGHT_NORMAL
            elif sharpe >= 0.5:
                weights[strat] = WEIGHT_NORMAL
            elif sharpe >= 0:
                weights[strat] = WEIGHT_REDUCED
            else:
                # Negative but not enough trades yet → reduce but not disable.
                weights[strat] = WEIGHT_REDUCED
        return weights

    def get_disabled_strategies(self) -> List[str]:
        return [s for s, w in self.get_strategy_weights().items() if w == WEIGHT_DISABLED]

    def _log_strategy_evolution(
        self,
        strategy: str,
        sharpe: float,
        win_rate: float,
        total_trades: int,
        weight: float,
        enabled: bool,
        reason: str,
    ) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO strategy_evolution
                    (strategy, sharpe_ratio, win_rate, total_trades, weight, enabled, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (strategy, float(sharpe), float(win_rate), int(total_trades), float(weight), 1 if enabled else 0, reason),
            )

    def run_strategy_scoring(self) -> Dict[str, float]:
        """L1 main entry: score, weight, and log evolution for every strategy."""
        sharpes = self.score_strategies()
        counts = self.get_strategy_trade_counts()
        weights = self.get_strategy_weights()
        for strat, sharpe in sharpes.items():
            trades = self._fetch_closed_trades(strategy=strat, limit=self.rolling_window)
            n = len(trades)
            if n:
                wins = sum(1 for t in trades if float(t.get("pnl") or 0.0) > 0)
                win_rate = wins / n
            else:
                win_rate = 0.0
            w = weights.get(strat, WEIGHT_NORMAL)
            enabled = w != WEIGHT_DISABLED
            reason = (
                f"sharpe={sharpe:.3f} n={counts.get(strat, 0)} win_rate={win_rate:.3f}"
            )
            self._log_strategy_evolution(strat, sharpe, win_rate, counts.get(strat, 0), w, enabled, reason)
        return weights

    # ==================================================================
    # LEVEL 2: Parameter adaptation
    # ==================================================================

    def optimize_rsi_thresholds(self, symbol: str, bars: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Wrapper around AdaptiveParams — expects caller to supply bars."""
        if bars is None:
            return {
                "rsi_overbought": DEFAULT_PARAMS["rsi_overbought"],
                "rsi_oversold": DEFAULT_PARAMS["rsi_oversold"],
            }
        out = self.params.optimize_rsi_thresholds(bars)
        for name, value in out.items():
            self.params.set_param("mean_reversion", symbol, name, value, sample_size=len(bars))
        return out

    def optimize_stop_distances(self, strategy: str) -> Dict[str, float]:
        """Optimize SL/TP ATR multipliers using closed trades for this strategy."""
        trades = self._fetch_closed_trades(strategy=strategy, limit=self.rolling_window)
        # Convert decision_json → metadata for the optimizer
        normalized: List[Dict[str, Any]] = []
        for t in trades:
            md = _parse_metadata(t.get("decision_json"))
            normalized.append(
                {
                    "pnl_pct": t.get("pnl_pct"),
                    "atr_pct": md.get("atr_pct"),
                    "metadata": md,
                }
            )
        out = self.params.optimize_stop_distances(normalized)
        for name, value in out.items():
            self.params.set_param(strategy, "__ANY__", name, value, sample_size=len(normalized))
        return out

    def get_adaptive_params(self, strategy: str, symbol: str) -> Dict[str, float]:
        """Merge symbol-specific + strategy-wide + defaults."""
        symbol_params = self.params.get_params(strategy, symbol)
        any_params = self.params.get_params(strategy, "__ANY__")
        merged = dict(DEFAULT_PARAMS)
        # __ANY__ overrides defaults, symbol overrides __ANY__.
        for source in (any_params, symbol_params):
            for k, v in source.items():
                merged[k] = v
        return merged

    def run_mini_optimization(self, symbol_bars: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Dict[str, float]]:
        """Run L2 optimizers for each strategy that has enough data.

        ``symbol_bars`` is an optional mapping {symbol: ohlcv_df}. Supply it
        to also run the RSI threshold optimizer per symbol.
        """
        changes: Dict[str, Dict[str, float]] = {}
        counts = self.get_strategy_trade_counts()
        for strat, n in counts.items():
            if n < 20:  # Not enough closed trades to tune SL/TP
                continue
            out = self.optimize_stop_distances(strat)
            changes.setdefault(strat, {}).update(out)

        if symbol_bars:
            for symbol, bars in symbol_bars.items():
                if bars is None or len(bars) < 50:
                    continue
                rsi_out = self.optimize_rsi_thresholds(symbol, bars)
                changes.setdefault("mean_reversion", {}).update(
                    {f"{symbol}:{k}": v for k, v in rsi_out.items()}
                )
        return changes

    # ==================================================================
    # LEVEL 3: Signal quality scoring
    # ==================================================================

    def _trades_to_dataframe(self, limit: int = 2000) -> pd.DataFrame:
        """Fetch closed trades and normalize for the meta-model."""
        trades = self._fetch_closed_trades(limit=limit)
        records: List[Dict[str, Any]] = []
        for t in trades:
            md = _parse_metadata(t.get("decision_json"))
            records.append(
                {
                    "strategy": t.get("strategy") or "",
                    "category": md.get("category") or "unknown",
                    "pnl": float(t.get("pnl") or 0.0),
                    "metadata": md,
                }
            )
        return pd.DataFrame(records)

    def train_signal_model(self) -> Optional[SignalScorer]:
        """Train the meta-model on all available closed trades."""
        df = self._trades_to_dataframe()
        if df.empty:
            return None
        acc = self.signal_scorer.train(df)
        if acc is None:
            return None
        importances = self.signal_scorer.feature_importance()
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO signal_model_log
                    (model_version, accuracy, auc_roc, n_training_samples, feature_importances)
                VALUES (
                    COALESCE((SELECT MAX(model_version)+1 FROM signal_model_log), 1),
                    ?, ?, ?, ?
                )
                """,
                (
                    float(self.signal_scorer.accuracy),
                    float(self.signal_scorer.auc_roc),
                    int(self.signal_scorer.n_training_samples),
                    json.dumps(importances),
                ),
            )
        return self.signal_scorer

    def predict_signal_quality(self, features: Dict[str, Any], strategy: str = "", category: str = "") -> float:
        """Score a candidate signal 0-1. Neutral 0.5 if no model trained."""
        return self.signal_scorer.predict(features, strategy=strategy, category=category)

    def get_feature_importances(self) -> Dict[str, float]:
        return self.signal_scorer.feature_importance()

    # ==================================================================
    # LEVEL 4: Regime detection
    # ==================================================================

    def detect_regime(self, df: pd.DataFrame) -> str:
        return self.regime_detector.classify(df)

    def log_regime(self, symbol: str, df: pd.DataFrame, timeframe: Optional[str] = None) -> str:
        result = self.regime_detector.classify_with_features(df)
        feats = result.get("features", {}) or {}
        regime = result.get("regime", "unknown")
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO regime_history
                    (symbol, timeframe, regime, adx, atr_pct, bb_width, sma_slope)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    timeframe,
                    regime,
                    float(feats.get("adx")) if feats.get("adx") is not None and not np.isnan(feats.get("adx")) else None,
                    float(feats.get("atr_pct")) if feats.get("atr_pct") is not None and not np.isnan(feats.get("atr_pct")) else None,
                    float(feats.get("bb_width")) if feats.get("bb_width") is not None and not np.isnan(feats.get("bb_width")) else None,
                    float(feats.get("sma_slope")) if feats.get("sma_slope") is not None and not np.isnan(feats.get("sma_slope")) else None,
                ),
            )
        return regime

    def get_regime_strategy_affinity(self) -> Dict[str, Dict[str, float]]:
        """Learned (or prior) affinity matrix strategy → {regime: multiplier}."""
        learned = self._learn_regime_affinity()
        out = {k: dict(v) for k, v in DEFAULT_REGIME_AFFINITY.items()}
        for strat, regime_scores in learned.items():
            out.setdefault(strat, dict(DEFAULT_REGIME_AFFINITY.get(strat, {})))
            for regime, score in regime_scores.items():
                out[strat][regime] = score
        return out

    def _learn_regime_affinity(self) -> Dict[str, Dict[str, float]]:
        """Infer strategy/regime multipliers from trades and regime_history.

        For each (strategy, regime) pair with >= 10 trades, compute the
        avg pnl_pct. Convert to a multiplier centered at 1.0 (positive
        returns → > 1, negative → < 1).
        """
        with get_connection(self.db_path) as conn:
            # Match trades to the most-recent regime snapshot for their symbol
            # at the time of entry. For simplicity we take the regime nearest
            # the trade entry_time in regime_history.
            rows = conn.execute(
                """
                SELECT t.strategy AS strategy, t.pnl_pct AS pnl_pct,
                       (SELECT r.regime FROM regime_history r
                        WHERE r.symbol = t.symbol AND r.detected_at <= t.entry_time
                        ORDER BY r.detected_at DESC LIMIT 1) AS regime
                FROM trades t
                WHERE t.status IN ('CLOSED','STOPPED')
                """
            ).fetchall()
        buckets: Dict[Tuple[str, str], List[float]] = {}
        for row in rows:
            regime = row["regime"]
            if not regime:
                continue
            buckets.setdefault((row["strategy"], regime), []).append(float(row["pnl_pct"] or 0.0))
        out: Dict[str, Dict[str, float]] = {}
        for (strat, regime), returns in buckets.items():
            if len(returns) < 10:
                continue
            avg = np.mean(returns)
            # Map avg return to a multiplier: avg=0 → 1.0, avg=+0.01 → 1.5, avg=-0.01 → 0.5
            mult = max(0.1, min(2.5, 1.0 + avg * 50))
            out.setdefault(strat, {})[regime] = float(mult)
        return out

    def get_regime_adjusted_weight(self, strategy: str, regime: str) -> float:
        """Return strategy weight × regime affinity."""
        weights = self.get_strategy_weights()
        base = weights.get(strategy, WEIGHT_NORMAL)
        affinity = self.get_regime_strategy_affinity().get(strategy, {}).get(regime, 1.0)
        return float(base * affinity)

    # ==================================================================
    # LEVEL 5: Kronos accuracy tracking
    # ==================================================================

    def log_kronos_prediction(
        self,
        symbol: str,
        timeframe: str,
        predicted_direction: str,
        predicted_magnitude: float,
        confidence: float,
        horizon: int,
        predicted_price: Optional[float] = None,
    ) -> int:
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO kronos_predictions
                    (symbol, timeframe, horizon, predicted_direction, predicted_magnitude,
                     confidence, predicted_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    timeframe,
                    int(horizon),
                    predicted_direction,
                    float(predicted_magnitude),
                    float(confidence),
                    float(predicted_price) if predicted_price is not None else None,
                ),
            )
            return int(cursor.lastrowid)

    def record_kronos_outcome(
        self,
        prediction_id: int,
        actual_direction: str,
        actual_magnitude: float,
        actual_price: Optional[float] = None,
    ) -> None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT predicted_direction FROM kronos_predictions WHERE id = ?",
                (int(prediction_id),),
            ).fetchone()
            if not row:
                return
            correct = 1 if actual_direction == row["predicted_direction"] else 0
            conn.execute(
                """
                UPDATE kronos_predictions
                SET actual_direction = ?, actual_magnitude = ?, actual_price = ?,
                    evaluation_time = datetime('now'), correct = ?
                WHERE id = ?
                """,
                (
                    actual_direction,
                    float(actual_magnitude),
                    float(actual_price) if actual_price is not None else None,
                    correct,
                    int(prediction_id),
                ),
            )

    # ------------------------------------------------------------------
    # L5 — pending-prediction resolver (REL-376)
    # ------------------------------------------------------------------

    # Direction threshold MUST match KronosForecastAgent.UP_THRESHOLD_PCT so
    # we measure with the same yardstick the model uses.
    KRONOS_DIRECTION_THRESHOLD_PCT = 0.25

    # Per-timeframe bar-step. Aligned with kronos_agent._timeframe_to_timedelta
    # but expressed in pandas-compatible offsets.
    _TIMEFRAME_STEP = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
        "1w": pd.Timedelta(weeks=1),
    }

    @classmethod
    def _direction_from_pct(cls, pct: float) -> str:
        if pct > cls.KRONOS_DIRECTION_THRESHOLD_PCT:
            return "UP"
        if pct < -cls.KRONOS_DIRECTION_THRESHOLD_PCT:
            return "DOWN"
        return "NEUTRAL"

    @classmethod
    def _resolution_target(
        cls, prediction_time: pd.Timestamp, timeframe: str, horizon: int
    ) -> Optional[pd.Timestamp]:
        """Return the wall-clock time at which the prediction can be resolved.

        Returns ``prediction_time + horizon * step``. If the timeframe is
        unknown we return ``None`` (caller should skip the row).
        """
        step = cls._TIMEFRAME_STEP.get(str(timeframe or "").lower())
        if step is None or int(horizon) <= 0:
            return None
        return pd.Timestamp(prediction_time) + step * int(horizon)

    def evaluate_pending_kronos_predictions(
        self,
        now: Optional[pd.Timestamp] = None,
        max_rows: int = 500,
        price_lookup: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Resolve every pending Kronos prediction whose horizon has elapsed.

        For each row with ``correct IS NULL`` and ``prediction_time +
        horizon * step`` <= ``now``:

        1. Pick the anchor close at ``prediction_time`` (the bar at-or-just-
           before the prediction was logged).
        2. Pick the actual close at ``prediction_time + horizon * step``
           (the bar at-or-just-before that target).
        3. Compute ``actual_pct`` and ``actual_direction`` using the SAME
           ``±0.25%`` threshold the model uses for its UP/DOWN call.
        4. Write back ``actual_*``, ``evaluation_time``, ``correct``.

        ``price_lookup(symbol, interval, start, end) -> pd.DataFrame`` may be
        injected for tests. In production we use ``data_fetcher.fetch_market_data``
        with a tight date window so the cache deduplicates calls.

        Returns a stats dict ``{resolved, skipped_no_data, skipped_unknown_tf}``.
        """
        if now is None:
            now = pd.Timestamp.utcnow().tz_localize(None)
        else:
            now = pd.Timestamp(now)
            if now.tzinfo is not None:
                now = now.tz_localize(None)

        # Filter out symbols that are no longer in MARKETS (e.g. delisted) so we
        # don't spam yfinance with requests that always fail. Pending rows for
        # such symbols are marked correct=-1 to take them out of the queue
        # without losing the historical record.
        try:
            from market_config import MARKETS as _MARKETS  # local import to avoid cycle
            _active_symbols = set(_MARKETS.keys())
        except Exception:
            _active_symbols = None

        if price_lookup is None:
            try:
                from data_fetcher import fetch_market_data as _fmd  # local import to avoid cycle
            except Exception:
                _fmd = None

            def price_lookup(symbol, interval, start, end):  # noqa: E306
                if _fmd is None:
                    return pd.DataFrame()
                try:
                    return _fmd(symbol, interval=interval, start_date=start, end_date=end, use_cache=True)
                except Exception as exc:
                    logger.debug("price_lookup failed for %s/%s: %s", symbol, interval, exc)
                    return pd.DataFrame()

        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, timeframe, horizon, predicted_direction,
                       predicted_magnitude, predicted_price, prediction_time
                FROM kronos_predictions
                WHERE correct IS NULL
                ORDER BY prediction_time ASC
                LIMIT ?
                """,
                (int(max_rows),),
            ).fetchall()
        rows = [dict(r) for r in rows]

        stats = {
            "resolved": 0,
            "skipped_no_data": 0,
            "skipped_unknown_tf": 0,
            "skipped_not_due": 0,
            "skipped_delisted": 0,
        }

        # Bulk-mark delisted-symbol rows as correct=-1 so we stop polling them
        # every cycle. -1 is a sentinel meaning "unresolvable" (NULL = pending,
        # 0/1 = resolved). It excludes them from accuracy stats below.
        if _active_symbols is not None:
            with get_connection(self.db_path) as conn:
                placeholders = ",".join("?" * len(_active_symbols)) if _active_symbols else "''"
                if _active_symbols:
                    cur = conn.execute(
                        f"""
                        UPDATE kronos_predictions
                           SET correct = -1,
                               evaluation_time = datetime('now')
                         WHERE correct IS NULL
                           AND symbol NOT IN ({placeholders})
                        """,
                        tuple(_active_symbols),
                    )
                    stats["skipped_delisted"] = cur.rowcount or 0

        for row in rows:
            symbol = row["symbol"]
            if _active_symbols is not None and symbol not in _active_symbols:
                # Already marked above; skip to avoid the yfinance call.
                continue
            timeframe = (row["timeframe"] or "").lower()
            horizon = int(row["horizon"] or 0)
            try:
                pred_time = pd.Timestamp(row["prediction_time"])
                if pred_time.tzinfo is not None:
                    pred_time = pred_time.tz_localize(None)
            except Exception:
                stats["skipped_unknown_tf"] += 1
                continue

            target = self._resolution_target(pred_time, timeframe, horizon)
            if target is None:
                stats["skipped_unknown_tf"] += 1
                continue
            if target > now:
                stats["skipped_not_due"] += 1
                continue

            # Pull a window covering [pred_time - 1 step, target + 1 step].
            step = self._TIMEFRAME_STEP[timeframe]
            start = pred_time - step * 2
            end = target + step * 2
            df = price_lookup(symbol, timeframe, start, end)
            if df is None or len(df) == 0 or "Close" not in df.columns:
                stats["skipped_no_data"] += 1
                continue

            ts_col = "Datetime" if "Datetime" in df.columns else None
            if ts_col is None:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index().rename(columns={df.index.name or "index": "Datetime"})
                    ts_col = "Datetime"
                else:
                    stats["skipped_no_data"] += 1
                    continue

            ts = pd.to_datetime(df[ts_col], errors="coerce")
            try:
                ts = ts.dt.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

            anchor_idx = df["_ts"].searchsorted(pred_time, side="right") - 1
            target_idx = df["_ts"].searchsorted(target, side="right") - 1
            if anchor_idx < 0 or target_idx < 0 or target_idx <= anchor_idx:
                stats["skipped_no_data"] += 1
                continue

            anchor_close = float(df["Close"].iloc[anchor_idx])
            actual_close = float(df["Close"].iloc[target_idx])
            if anchor_close <= 0:
                stats["skipped_no_data"] += 1
                continue

            actual_pct = (actual_close - anchor_close) / anchor_close * 100.0
            actual_direction = self._direction_from_pct(actual_pct)
            correct = 1 if actual_direction == row["predicted_direction"] else 0

            with get_connection(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE kronos_predictions
                    SET actual_direction = ?, actual_magnitude = ?, actual_price = ?,
                        evaluation_time = datetime('now'), correct = ?
                    WHERE id = ?
                    """,
                    (actual_direction, float(actual_pct), float(actual_close), int(correct), int(row["id"])),
                )
            stats["resolved"] += 1

        return stats

    def evaluate_kronos_accuracy(self) -> Dict[str, Any]:
        """Summary stats over all evaluated Kronos predictions."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT symbol, predicted_direction, predicted_magnitude, confidence,
                       actual_direction, actual_magnitude, correct
                FROM kronos_predictions
                WHERE correct IS NOT NULL AND correct >= 0
                """
            ).fetchall()
        rows = [dict(r) for r in rows]
        if not rows:
            return {
                "hit_rate": 0.0,
                "avg_error": 0.0,
                "calibration_curve": {},
                "per_market_accuracy": {},
                "n": 0,
            }

        hits = sum(1 for r in rows if r["correct"])
        errors = [
            abs(float(r["predicted_magnitude"]) - float(r["actual_magnitude"] or 0.0))
            for r in rows
        ]

        # Calibration curve: bucket confidence into deciles, compute hit rate per bucket.
        buckets: Dict[float, List[int]] = {}
        for r in rows:
            conf = float(r["confidence"] or 0.0)
            bucket = round(conf, 1)
            buckets.setdefault(bucket, []).append(1 if r["correct"] else 0)
        calibration = {bucket: float(np.mean(vals)) for bucket, vals in sorted(buckets.items())}

        # Per-market accuracy.
        market_buckets: Dict[str, List[int]] = {}
        for r in rows:
            market_buckets.setdefault(r["symbol"], []).append(1 if r["correct"] else 0)
        per_market = {sym: float(np.mean(vals)) for sym, vals in market_buckets.items()}

        return {
            "hit_rate": hits / len(rows),
            "avg_error": float(np.mean(errors)) if errors else 0.0,
            "calibration_curve": calibration,
            "per_market_accuracy": per_market,
            "n": len(rows),
        }

    def get_kronos_confidence_adjustment(self, symbol: str) -> float:
        """Discount factor for Kronos confidence on this market.

        Compares claimed confidence to realized hit rate. Returns a
        multiplier in [0.5, 1.5] that callers can apply to raw Kronos
        confidence scores.
        """
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT AVG(confidence) AS avg_conf, AVG(correct) AS hit_rate, COUNT(*) AS n
                FROM kronos_predictions
                WHERE symbol = ? AND correct IS NOT NULL AND correct >= 0
                """,
                (symbol,),
            ).fetchone()
        if not row or not row["n"] or row["n"] < 10:
            return 1.0
        avg_conf = float(row["avg_conf"] or 0.0)
        hit_rate = float(row["hit_rate"] or 0.0)
        if avg_conf == 0:
            return 1.0
        ratio = hit_rate / avg_conf
        return max(0.5, min(1.5, ratio))

    # ==================================================================
    # Orchestration
    # ==================================================================

    def _should_fire(self, level: str, total_trades: int, cadence: int) -> bool:
        """Fire a level when total_trades crossed a new cadence boundary."""
        if total_trades < cadence:
            return False
        last = self._last_cycle_triggers.get(level, 0)
        next_boundary = ((last // cadence) + 1) * cadence
        return total_trades >= next_boundary

    def run_improvement_cycle(
        self,
        total_trades: Optional[int] = None,
        symbol_bars: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Orchestrate the cycle. Cheap levels (L4) always run; others gated."""
        if total_trades is None:
            total_trades = sum(self.get_strategy_trade_counts().values())

        result = ImprovementCycleResult(total_trades=total_trades)

        # L4 — regime detection runs every call (cheap). Caller provides bars.
        if symbol_bars:
            for symbol, df in symbol_bars.items():
                try:
                    regime = self.log_regime(symbol, df)
                    result.notes.append(f"regime[{symbol}]={regime}")
                except Exception as e:  # pragma: no cover — log & continue
                    logger.warning("regime logging failed for %s: %s", symbol, e)
            result.levels_run.append("L4")

        # L1 — every CADENCE_L1 trades
        if self._should_fire("L1", total_trades, CADENCE_L1):
            result.strategy_weights = self.run_strategy_scoring()
            result.disabled_strategies = [s for s, w in result.strategy_weights.items() if w == WEIGHT_DISABLED]
            result.levels_run.append("L1")
            self._last_cycle_triggers["L1"] = total_trades

        # L2 — every CADENCE_L2 trades
        if self._should_fire("L2", total_trades, CADENCE_L2):
            result.params_updated = self.run_mini_optimization(symbol_bars=symbol_bars)
            result.levels_run.append("L2")
            self._last_cycle_triggers["L2"] = total_trades

        # L3 — every CADENCE_L3 trades
        if self._should_fire("L3", total_trades, CADENCE_L3):
            trained = self.train_signal_model()
            result.model_trained = trained is not None
            if trained is not None:
                result.model_accuracy = trained.accuracy
            result.levels_run.append("L3")
            self._last_cycle_triggers["L3"] = total_trades

        # L5 — every CADENCE_L5 trades; resolve pending predictions every cycle
        # (cheap: one bounded DB scan + cache-deduped price lookups).
        try:
            self.evaluate_pending_kronos_predictions()
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("evaluate_pending_kronos_predictions failed: %s", exc)
        if self._should_fire("L5", total_trades, CADENCE_L5):
            result.kronos_stats = self.evaluate_kronos_accuracy()
            result.levels_run.append("L5")
            self._last_cycle_triggers["L5"] = total_trades

        return result.to_dict()
