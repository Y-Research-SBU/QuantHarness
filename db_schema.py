"""
SQLite database schema and management for paper trading.
Tables: trades, positions, portfolios, portfolio_snapshots, strategy_performance, api_costs.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "paper_trades.db"

SCHEMA_SQL = """
-- Trades table: every trade (entry and exit)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
    entry_price REAL NOT NULL,
    exit_price REAL,
    position_size REAL NOT NULL,
    quantity REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    pnl REAL DEFAULT 0.0,
    pnl_pct REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'CLOSED', 'STOPPED')),
    agent_reasoning TEXT,
    indicator_report TEXT,
    pattern_report TEXT,
    trend_report TEXT,
    decision_json TEXT,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Open positions view helper
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);

-- Portfolios table: one per market
CREATE TABLE IF NOT EXISTS portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    initial_balance REAL NOT NULL DEFAULT 10000.0,
    current_balance REAL NOT NULL DEFAULT 10000.0,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    consecutive_losses INTEGER NOT NULL DEFAULT 0,
    max_drawdown REAL NOT NULL DEFAULT 0.0,
    peak_balance REAL NOT NULL DEFAULT 10000.0,
    is_circuit_breaker_active INTEGER NOT NULL DEFAULT 0,
    daily_pnl REAL NOT NULL DEFAULT 0.0,
    daily_pnl_reset_date TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Portfolio snapshots for equity curve tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    balance REAL NOT NULL,
    total_pnl REAL NOT NULL,
    open_positions INTEGER NOT NULL DEFAULT 0,
    drawdown_pct REAL NOT NULL DEFAULT 0.0,
    snapshot_time TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON portfolio_snapshots(symbol);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON portfolio_snapshots(snapshot_time);

-- Strategy performance tracking
CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    avg_win REAL NOT NULL DEFAULT 0.0,
    avg_loss REAL NOT NULL DEFAULT 0.0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    profit_factor REAL NOT NULL DEFAULT 0.0,
    sharpe_ratio REAL NOT NULL DEFAULT 0.0,
    max_drawdown REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(strategy, symbol, timeframe)
);

-- API cost tracking
CREATE TABLE IF NOT EXISTS api_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    operation TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_api_costs_symbol ON api_costs(symbol);
CREATE INDEX IF NOT EXISTS idx_api_costs_time ON api_costs(created_at);
"""


def get_db_path(db_path: Optional[str] = None) -> str:
    """Get database path, defaulting to DEFAULT_DB_PATH."""
    return db_path or DEFAULT_DB_PATH


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Initialize the database with schema.
    Returns a connection to the database.
    """
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


@contextmanager
def get_connection(db_path: Optional[str] = None):
    """Context manager for database connections."""
    conn = init_db(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reset_db(db_path: Optional[str] = None):
    """Drop all tables and recreate (for testing)."""
    path = get_db_path(db_path)
    if Path(path).exists():
        Path(path).unlink()
    init_db(db_path)
