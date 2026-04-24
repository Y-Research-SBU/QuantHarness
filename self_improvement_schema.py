"""
Schema migration for the self-improvement module (Levels 1-5).

Adds tables:
  - strategy_evolution     (L1: rolling strategy scoring history)
  - adaptive_params        (L2: per-strategy/per-symbol optimized params)
  - kronos_predictions     (L5: forecast accuracy log)
  - regime_history         (L4: detected market regimes over time)
  - signal_model_log       (L3: trained meta-model metadata)

The migration is idempotent (CREATE TABLE IF NOT EXISTS) and intentionally
kept separate from ``db_schema.SCHEMA_SQL`` so the currently running trading
code is not touched.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from db_schema import get_connection


SELF_IMPROVEMENT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_evolution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    sharpe_ratio REAL NOT NULL DEFAULT 0.0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    total_trades INTEGER NOT NULL DEFAULT 0,
    weight REAL NOT NULL DEFAULT 1.0,
    enabled INTEGER NOT NULL DEFAULT 1,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_strategy_evolution_strategy ON strategy_evolution(strategy);
CREATE INDEX IF NOT EXISTS idx_strategy_evolution_timestamp ON strategy_evolution(timestamp);

CREATE TABLE IF NOT EXISTS adaptive_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    param_name TEXT NOT NULL,
    param_value REAL NOT NULL,
    optimized_at TEXT NOT NULL DEFAULT (datetime('now')),
    sample_size INTEGER NOT NULL DEFAULT 0,
    improvement_pct REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_adaptive_params_lookup
    ON adaptive_params(strategy, symbol, param_name, optimized_at);

CREATE TABLE IF NOT EXISTS kronos_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    predicted_direction TEXT NOT NULL,
    predicted_magnitude REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.0,
    actual_direction TEXT,
    actual_magnitude REAL,
    prediction_time TEXT NOT NULL DEFAULT (datetime('now')),
    evaluation_time TEXT,
    correct INTEGER,
    predicted_price REAL,
    actual_price REAL
);
CREATE INDEX IF NOT EXISTS idx_kronos_pred_symbol ON kronos_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_kronos_pred_time ON kronos_predictions(prediction_time);

CREATE TABLE IF NOT EXISTS regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT,
    regime TEXT NOT NULL,
    adx REAL,
    atr_pct REAL,
    bb_width REAL,
    sma_slope REAL,
    detected_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_regime_symbol ON regime_history(symbol);
CREATE INDEX IF NOT EXISTS idx_regime_time ON regime_history(detected_at);

CREATE TABLE IF NOT EXISTS signal_model_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version INTEGER NOT NULL DEFAULT 1,
    accuracy REAL NOT NULL DEFAULT 0.0,
    auc_roc REAL NOT NULL DEFAULT 0.0,
    n_training_samples INTEGER NOT NULL DEFAULT 0,
    feature_importances TEXT,
    trained_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signal_model_trained_at ON signal_model_log(trained_at);
"""


def apply_self_improvement_schema(db_path: Optional[str] = None) -> None:
    """Create self-improvement tables if they don't exist yet.

    Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS``.
    """
    with get_connection(db_path) as conn:
        conn.executescript(SELF_IMPROVEMENT_SCHEMA_SQL)


def self_improvement_tables_exist(db_path: Optional[str] = None) -> bool:
    """Return True if all self-improvement tables are present."""
    required = {
        "strategy_evolution",
        "adaptive_params",
        "kronos_predictions",
        "regime_history",
        "signal_model_log",
    }
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    existing = {r["name"] for r in rows}
    return required.issubset(existing)
