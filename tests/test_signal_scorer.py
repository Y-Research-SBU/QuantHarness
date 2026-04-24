"""Tests for signal_scorer (Level 3 of the self-improvement system)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from signal_scorer import (
    CATEGORY_LABELS,
    NUMERIC_FEATURES,
    STRATEGY_LABELS,
    SignalScorer,
    extract_features,
    feature_names,
    feature_vector,
)


# ---------------------------------------------------------------------------
# Helpers — synthesize a closed-trade DataFrame where "good" features predict wins.
# ---------------------------------------------------------------------------


def _synth_trades(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic trades where higher kronos_confidence + signal_strength tend to win."""
    rng = np.random.default_rng(seed)
    records = []
    strategies = STRATEGY_LABELS
    for i in range(n):
        kronos_conf = float(rng.uniform(0.3, 0.95))
        strength = float(rng.uniform(0.0, 1.0))
        rsi = float(rng.uniform(20, 80))
        macd_hist = float(rng.normal(0, 0.5))
        volume_ratio = float(rng.uniform(0.5, 2.0))
        atr_pct = float(rng.uniform(0.005, 0.05))
        # Win probability is driven by kronos_conf + strength.
        p_win = max(0.05, min(0.95, 0.1 + 0.5 * kronos_conf + 0.4 * strength))
        is_win = rng.random() < p_win
        pnl = float(rng.uniform(0.5, 5.0)) if is_win else float(rng.uniform(-5.0, -0.5))
        strat = strategies[i % len(strategies)]
        records.append(
            {
                "strategy": strat,
                "category": "crypto",
                "pnl": pnl,
                "metadata": {
                    "rsi": rsi,
                    "macd_hist": macd_hist,
                    "kronos_confidence": kronos_conf,
                    "volume_ratio": volume_ratio,
                    "atr_pct": atr_pct,
                    "signal_strength": strength,
                },
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_extract_features_defaults_for_missing():
    feats = extract_features({})
    assert feats["rsi"] == 50.0  # neutral default
    assert feats["kronos_confidence"] == 0.0
    assert feats["signal_strength"] == 0.0


def test_extract_features_handles_nested_kronos():
    feats = extract_features({"kronos": {"confidence": 0.7}})
    assert feats["kronos_confidence"] == pytest.approx(0.7)


def test_extract_features_strategy_one_hot():
    feats = extract_features({}, strategy="momentum")
    assert feats["strategy_momentum"] == 1.0
    assert feats["strategy_breakout"] == 0.0


def test_feature_vector_length_matches_feature_names():
    feats = extract_features({}, strategy="momentum", category="crypto")
    vec = feature_vector(feats)
    assert len(vec) == len(feature_names())


def test_feature_names_cover_numeric_and_labels():
    names = feature_names()
    for n in NUMERIC_FEATURES:
        assert n in names
    for s in STRATEGY_LABELS:
        assert f"strategy_{s}" in names
    for c in CATEGORY_LABELS:
        assert f"category_{c}" in names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_with_no_data_returns_none():
    scorer = SignalScorer()
    out = scorer.train(pd.DataFrame())
    assert out is None
    assert scorer.model is None


def test_train_with_too_few_samples_returns_none():
    scorer = SignalScorer()
    df = _synth_trades(n=10)
    assert scorer.train(df) is None
    assert scorer.model is None


def test_train_with_100_trades_fits_model():
    scorer = SignalScorer()
    df = _synth_trades(n=200, seed=1)
    acc = scorer.train(df)
    assert acc is not None
    assert scorer.model is not None
    assert 0.0 <= acc <= 1.0
    assert scorer.n_training_samples == 200


def test_train_requires_both_classes():
    # All trades winners → only one class → should not train.
    scorer = SignalScorer()
    rows = []
    for _ in range(150):
        rows.append({"strategy": "momentum", "category": "crypto", "pnl": 1.0, "metadata": {"rsi": 50}})
    out = scorer.train(pd.DataFrame(rows))
    assert out is None
    assert scorer.model is None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_returns_0_5_when_no_model():
    scorer = SignalScorer()
    score = scorer.predict({"rsi": 60})
    assert score == 0.5


def test_predict_returns_0_to_1_after_training():
    scorer = SignalScorer()
    df = _synth_trades(n=200)
    scorer.train(df)
    score = scorer.predict({"kronos_confidence": 0.9, "signal_strength": 0.9}, strategy="momentum", category="crypto")
    assert 0.0 <= score <= 1.0


def test_predict_high_vs_low_confidence_differs():
    scorer = SignalScorer()
    df = _synth_trades(n=400, seed=3)
    scorer.train(df)
    high = scorer.predict(
        {"kronos_confidence": 0.95, "signal_strength": 0.9, "rsi": 50},
        strategy="momentum",
        category="crypto",
    )
    low = scorer.predict(
        {"kronos_confidence": 0.35, "signal_strength": 0.1, "rsi": 50},
        strategy="momentum",
        category="crypto",
    )
    # Model learned that higher confidence/strength → more likely win.
    assert high > low


def test_predict_clamped_to_unit_interval():
    scorer = SignalScorer()
    df = _synth_trades(n=200)
    scorer.train(df)
    for _ in range(20):
        f = {
            "rsi": float(np.random.uniform(0, 100)),
            "kronos_confidence": float(np.random.uniform(0, 1)),
            "signal_strength": float(np.random.uniform(0, 1)),
        }
        s = scorer.predict(f, strategy="momentum", category="crypto")
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def test_feature_importance_empty_when_no_model():
    scorer = SignalScorer()
    assert scorer.feature_importance() == {}


def test_feature_importance_sums_to_1():
    scorer = SignalScorer()
    df = _synth_trades(n=200)
    scorer.train(df)
    importances = scorer.feature_importance()
    total = sum(importances.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert len(importances) == len(feature_names())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_model_persistence_roundtrip(tmp_path):
    scorer = SignalScorer()
    df = _synth_trades(n=200, seed=5)
    scorer.train(df)
    path = str(tmp_path / "model.pkl")
    scorer.save(path)

    scorer2 = SignalScorer()
    scorer2.load(path)
    feats = {"kronos_confidence": 0.8, "signal_strength": 0.7}
    assert scorer2.predict(feats, strategy="momentum", category="crypto") == pytest.approx(
        scorer.predict(feats, strategy="momentum", category="crypto"),
        rel=1e-6,
    )
    assert scorer2.accuracy == pytest.approx(scorer.accuracy)


def test_save_without_training_raises():
    scorer = SignalScorer()
    with pytest.raises(RuntimeError):
        scorer.save("/tmp/should_not_exist.pkl")
