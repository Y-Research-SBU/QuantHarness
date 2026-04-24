"""
Signal quality meta-model (Level 3 of the self-improvement system).

Trains a logistic-regression classifier on closed-trade history to predict
the probability that a candidate signal will be profitable.

Features extracted from a trade's ``decision_json`` metadata:
    rsi, macd_hist, kronos_confidence, volume_ratio, atr_pct, signal_strength,
    strategy (one-hot), category (one-hot)

Target:
    1 if pnl > 0 else 0

The model is trained in-memory and can be serialized to disk with pickle
if desired. ``predict`` returns 0.5 (neutral) if no model has been trained
yet, so callers can safely use this before enough data exists.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Numeric feature keys — order matters for reproducibility.
NUMERIC_FEATURES: List[str] = [
    "rsi",
    "macd_hist",
    "kronos_confidence",
    "volume_ratio",
    "atr_pct",
    "signal_strength",
]

# Known strategy / category labels for one-hot encoding.
STRATEGY_LABELS: List[str] = [
    "momentum",
    "mean_reversion",
    "breakout",
    "multi_factor",
    "kronos_momentum_confirm",
    "kronos_divergence",
    "multi_timeframe_kronos",
]

CATEGORY_LABELS: List[str] = ["crypto", "stocks", "commodities", "forex", "unknown"]


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion (None / bad inputs → default)."""
    try:
        if value is None:
            return default
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def extract_features(metadata: Dict[str, Any], strategy: str = "", category: str = "") -> Dict[str, float]:
    """Pull model features from a trade's metadata dict + labels.

    Returns a flat dict of floats. Missing values default to 0.0 except
    ``rsi`` which defaults to 50 (neutral) and ``kronos_confidence`` which
    defaults to 0 (no kronos signal).
    """
    metadata = metadata or {}
    raw: Dict[str, float] = {}
    raw["rsi"] = _as_float(metadata.get("rsi"), 50.0)
    raw["macd_hist"] = _as_float(metadata.get("macd_hist") or metadata.get("macd_histogram"))
    # Kronos fields can be nested under "kronos"
    kronos = metadata.get("kronos") or {}
    raw["kronos_confidence"] = _as_float(
        metadata.get("kronos_confidence") or kronos.get("confidence")
    )
    raw["volume_ratio"] = _as_float(metadata.get("volume_ratio"), 1.0)
    raw["atr_pct"] = _as_float(metadata.get("atr_pct"))
    raw["signal_strength"] = _as_float(metadata.get("signal_strength") or metadata.get("strength"))

    feats: Dict[str, float] = dict(raw)
    strat_key = (strategy or metadata.get("strategy") or "").lower()
    for label in STRATEGY_LABELS:
        feats[f"strategy_{label}"] = 1.0 if strat_key == label else 0.0
    cat_key = (category or metadata.get("category") or "unknown").lower()
    for label in CATEGORY_LABELS:
        feats[f"category_{label}"] = 1.0 if cat_key == label else 0.0
    return feats


def feature_vector(feats: Dict[str, float]) -> List[float]:
    """Serialize the feature dict to a fixed-order vector."""
    vec = [feats.get(k, 0.0) for k in NUMERIC_FEATURES]
    vec += [feats.get(f"strategy_{label}", 0.0) for label in STRATEGY_LABELS]
    vec += [feats.get(f"category_{label}", 0.0) for label in CATEGORY_LABELS]
    return vec


def feature_names() -> List[str]:
    names = list(NUMERIC_FEATURES)
    names += [f"strategy_{label}" for label in STRATEGY_LABELS]
    names += [f"category_{label}" for label in CATEGORY_LABELS]
    return names


class SignalScorer:
    """Wraps a sklearn LogisticRegression meta-model for signal quality."""

    MIN_TRAINING_SAMPLES = 100

    def __init__(self) -> None:
        self.model = None
        self._accuracy: float = 0.0
        self._auc: float = 0.0
        self._n_samples: int = 0
        self._feature_names: List[str] = feature_names()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_training_set(self, trades_df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        X_rows: List[List[float]] = []
        y_rows: List[int] = []
        for _, row in trades_df.iterrows():
            md = row.get("metadata")
            if isinstance(md, str):
                try:
                    md = json.loads(md) if md else {}
                except json.JSONDecodeError:
                    md = {}
            if not isinstance(md, dict):
                md = {}
            feats = extract_features(md, strategy=row.get("strategy", ""), category=row.get("category", ""))
            X_rows.append(feature_vector(feats))
            y_rows.append(1 if float(row.get("pnl", 0.0)) > 0 else 0)
        return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=int)

    def train(self, trades_df: pd.DataFrame) -> Optional[float]:
        """Fit the meta-model. Returns accuracy on the training set (or None)."""
        if trades_df is None or len(trades_df) < self.MIN_TRAINING_SAMPLES:
            self.model = None
            return None

        X, y = self._build_training_set(trades_df)
        if len(X) < self.MIN_TRAINING_SAMPLES:
            self.model = None
            return None
        # Need at least two classes to train a classifier.
        if len(np.unique(y)) < 2:
            self.model = None
            return None

        # Local import so unit tests that don't exercise this path don't pay
        # the sklearn import cost.
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score

        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(X, y)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        acc = float(accuracy_score(y, preds))
        try:
            auc = float(roc_auc_score(y, probs))
        except ValueError:
            auc = 0.5

        self.model = model
        self._accuracy = acc
        self._auc = auc
        self._n_samples = int(len(X))
        return acc

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, Any], strategy: str = "", category: str = "") -> float:
        """Return probability that a trade with these features is profitable.

        If no model has been trained, returns 0.5 (neutral).
        """
        if self.model is None:
            return 0.5

        # Accept either a raw metadata dict or a fully extracted feature dict.
        feats = features
        if not all(k in feats for k in NUMERIC_FEATURES):
            feats = extract_features(features, strategy=strategy, category=category)

        vec = np.asarray([feature_vector(feats)], dtype=float)
        prob = float(self.model.predict_proba(vec)[0, 1])
        # Clamp to [0,1] defensively.
        return max(0.0, min(1.0, prob))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def feature_importance(self) -> Dict[str, float]:
        """Return |coef| normalized to sum to 1. Empty dict if no model."""
        if self.model is None:
            return {}
        coefs = np.abs(self.model.coef_[0])
        total = float(coefs.sum())
        if total == 0:
            return {name: 0.0 for name in self._feature_names}
        return {name: float(c / total) for name, c in zip(self._feature_names, coefs)}

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def auc_roc(self) -> float:
        return self._auc

    @property
    def n_training_samples(self) -> int:
        return self._n_samples

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save — call train() first.")
        payload = {
            "model": self.model,
            "accuracy": self._accuracy,
            "auc": self._auc,
            "n_samples": self._n_samples,
            "feature_names": self._feature_names,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    def load(self, path: str) -> None:
        """Load a previously saved model."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.model = payload["model"]
        self._accuracy = payload.get("accuracy", 0.0)
        self._auc = payload.get("auc", 0.0)
        self._n_samples = payload.get("n_samples", 0)
        self._feature_names = payload.get("feature_names", feature_names())
