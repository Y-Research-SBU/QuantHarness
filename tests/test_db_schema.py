"""Tests for db_schema.py — SQLite init, connection, reset."""

import os
import sqlite3

import pytest

from db_schema import DEFAULT_DB_PATH, get_connection, get_db_path, init_db, reset_db


def test_get_db_path_default():
    assert get_db_path() == DEFAULT_DB_PATH


def test_get_db_path_custom():
    assert get_db_path("/tmp/custom.db") == "/tmp/custom.db"


def test_init_db_creates_file(tmp_db_path):
    conn = init_db(tmp_db_path)
    assert isinstance(conn, sqlite3.Connection)
    conn.close()
    assert os.path.exists(tmp_db_path)


def test_init_db_creates_all_tables(tmp_db_path):
    conn = init_db(tmp_db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [r[0] for r in cursor.fetchall()]
    conn.close()
    for expected in [
        "api_costs",
        "portfolios",
        "portfolio_snapshots",
        "strategy_performance",
        "trades",
    ]:
        assert expected in tables


def test_init_db_is_idempotent(tmp_db_path):
    init_db(tmp_db_path).close()
    init_db(tmp_db_path).close()
    # Second call should not crash.


def test_get_connection_context_manager(tmp_db_path):
    with get_connection(tmp_db_path) as conn:
        conn.execute("INSERT INTO portfolios (symbol) VALUES ('TEST-CM')")
    # Data persists after context exit.
    with get_connection(tmp_db_path) as conn:
        row = conn.execute("SELECT * FROM portfolios WHERE symbol = 'TEST-CM'").fetchone()
    assert row is not None


def test_get_connection_rolls_back_on_exception(tmp_db_path):
    init_db(tmp_db_path).close()
    with pytest.raises(ValueError):
        with get_connection(tmp_db_path) as conn:
            conn.execute("INSERT INTO portfolios (symbol) VALUES ('TEST-ROLLBACK')")
            raise ValueError("test rollback")

    with get_connection(tmp_db_path) as conn:
        row = conn.execute("SELECT * FROM portfolios WHERE symbol = 'TEST-ROLLBACK'").fetchone()
    assert row is None


def test_row_factory_returns_dict_like(tmp_db_path):
    with get_connection(tmp_db_path) as conn:
        conn.execute("INSERT INTO portfolios (symbol) VALUES ('RF-TEST')")
        row = conn.execute("SELECT * FROM portfolios WHERE symbol = 'RF-TEST'").fetchone()
    # sqlite3.Row supports index AND name access.
    assert row["symbol"] == "RF-TEST"


def test_reset_db(tmp_db_path):
    with get_connection(tmp_db_path) as conn:
        conn.execute("INSERT INTO portfolios (symbol) VALUES ('RESET-ME')")

    reset_db(tmp_db_path)
    with get_connection(tmp_db_path) as conn:
        row = conn.execute("SELECT * FROM portfolios WHERE symbol = 'RESET-ME'").fetchone()
    assert row is None


def test_reset_db_handles_missing_file(tmp_db_path):
    # Remove the file first.
    os.unlink(tmp_db_path)
    # Should create a fresh DB without error.
    reset_db(tmp_db_path)
    assert os.path.exists(tmp_db_path)


def test_trades_table_check_constraints(tmp_db_path):
    """direction must be LONG or SHORT; status must be one of OPEN/CLOSED/STOPPED."""
    init_db(tmp_db_path).close()
    with pytest.raises(sqlite3.IntegrityError):
        with get_connection(tmp_db_path) as conn:
            conn.execute(
                """INSERT INTO trades
                   (symbol, timeframe, strategy, direction, entry_price, position_size,
                    quantity, entry_time)
                   VALUES ('X', '4h', 'momentum', 'INVALID', 100, 100, 1, '2024-01-01')""",
            )
