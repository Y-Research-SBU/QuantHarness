"""Tests for the standalone QuantAgent monitoring dashboard (dashboard.py)."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import dashboard
from db_schema import init_db


# ──────────────────────────── fixtures ────────────────────────────


@pytest.fixture
def seeded_db(tmp_db_path):
    """Populate the temp DB with a realistic slice of portfolios, trades,
    snapshots and strategy_performance rows — enough to exercise every API."""
    conn = init_db(tmp_db_path)

    now = datetime.utcnow()
    t_now = now.strftime("%Y-%m-%dT%H:%M:%S")

    # Portfolios (3 markets: one profitable, one losing, one neutral)
    portfolios = [
        # symbol, initial, current, pnl, trades, wins, losses, cons_loss,
        # max_dd, peak, breaker, daily_pnl
        ("BTC-USD", 10000.0, 11500.0, 1500.0, 5, 3, 2, 0, 0.05, 11500.0, 0, 0.0),
        ("ETH-USD", 10000.0, 8500.0, -1500.0, 4, 1, 3, 2, 0.15, 10200.0, 1, 0.0),
        ("SPY", 10000.0, 10100.0, 100.0, 2, 1, 1, 0, 0.02, 10200.0, 0, 0.0),
    ]
    for (sym, init, cur, pnl, n, w, l, cl, dd, peak, br, dpnl) in portfolios:
        conn.execute(
            """INSERT INTO portfolios
               (symbol, initial_balance, current_balance, total_pnl,
                total_trades, winning_trades, losing_trades, consecutive_losses,
                max_drawdown, peak_balance, is_circuit_breaker_active, daily_pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sym, init, cur, pnl, n, w, l, cl, dd, peak, br, dpnl),
        )

    # Trades (mix of OPEN / CLOSED / STOPPED across strategies)
    trades = [
        ("BTC-USD", "1h", "momentum", "LONG", 90000.0, 91000.0, 200.0, 2.2,
         "CLOSED", t_now, t_now),
        ("BTC-USD", "1h", "momentum", "LONG", 89500.0, 88900.0, -100.0, -1.1,
         "STOPPED", t_now, t_now),
        ("ETH-USD", "4h", "mean_reversion", "SHORT", 3500.0, 3400.0, 80.0, 2.3,
         "CLOSED", t_now, t_now),
        ("ETH-USD", "4h", "kronos_divergence", "LONG", 3200.0, None, 0.0, 0.0,
         "OPEN", t_now, None),
        ("SPY", "1d", "multi_factor", "LONG", 450.0, 452.0, 20.0, 0.44,
         "CLOSED", t_now, t_now),
    ]
    for (sym, tf, strat, dir_, entry, exit_, pnl, pnl_pct, status, etime, xtime) in trades:
        conn.execute(
            """INSERT INTO trades
               (symbol, timeframe, strategy, direction, entry_price, exit_price,
                position_size, quantity, pnl, pnl_pct, status, entry_time, exit_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sym, tf, strat, dir_, entry, exit_, 1000.0, 0.01, pnl, pnl_pct, status,
             etime, xtime),
        )

    # Portfolio snapshots — build sparkline points for BTC and ETH
    base_time = now - timedelta(hours=10)
    for i in range(10):
        ts = (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO portfolio_snapshots (symbol, balance, total_pnl, snapshot_time) "
            "VALUES (?, ?, ?, ?)",
            ("BTC-USD", 10000.0 + i * 150.0, i * 150.0, ts),
        )
        conn.execute(
            "INSERT INTO portfolio_snapshots (symbol, balance, total_pnl, snapshot_time) "
            "VALUES (?, ?, ?, ?)",
            ("ETH-USD", 10000.0 - i * 150.0, -i * 150.0, ts),
        )

    # Strategy performance
    perf = [
        ("momentum", "BTC-USD", "1h", 5, 3, 2, 500.0, 250.0, 100.0, 0.6, 2.5, 1.2, 0.05),
        ("mean_reversion", "ETH-USD", "4h", 4, 1, 3, -400.0, 100.0, 200.0, 0.25, 0.3, -0.5, 0.15),
        ("multi_factor", "SPY", "1d", 2, 1, 1, 20.0, 20.0, 0.0, 0.5, 1.0, 0.1, 0.02),
    ]
    for row in perf:
        conn.execute(
            """INSERT INTO strategy_performance
               (strategy, symbol, timeframe, total_trades, winning_trades, losing_trades,
                total_pnl, avg_win, avg_loss, win_rate, profit_factor, sharpe_ratio, max_drawdown)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            row,
        )

    conn.commit()
    conn.close()
    return tmp_db_path


@pytest.fixture
def backtest_dir(tmp_path):
    """Create a backtest_results directory with one sample JSON file."""
    d = tmp_path / "backtest_results"
    d.mkdir()
    payload = {
        "generated_at": "2026-04-23T05:19:27",
        "results": [
            {
                "symbol": "BTC-USD",
                "strategy": "momentum",
                "timeframe": "1h",
                "starting_capital": 10000.0,
                "ending_capital": 12000.0,
                "total_return_pct": 20.0,
                "total_trades": 50,
                "winning_trades": 30,
                "losing_trades": 20,
                "win_rate": 0.6,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": 8.0,
                "profit_factor": 1.8,
                "avg_win": 100.0,
                "avg_loss": -50.0,
            },
            {
                "symbol": "ETH-USD",
                "strategy": "mean_reversion",
                "timeframe": "4h",
                "total_return_pct": -5.0,
                "total_trades": 30,
                "winning_trades": 10,
                "losing_trades": 20,
                "win_rate": 0.33,
                "sharpe_ratio": -0.4,
                "max_drawdown_pct": 15.0,
                "profit_factor": 0.7,
            },
        ],
    }
    (d / "backtest_20260423_051927.json").write_text(json.dumps(payload))
    return str(d)


@pytest.fixture
def app(seeded_db, backtest_dir):
    app = dashboard.create_app(db_path=seeded_db, backtest_dir=backtest_dir)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


# ──────────────────────────── index page ────────────────────────────


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    assert "QuantAgent" in body
    assert "Market Grid" in body
    assert "Strategy Performance" in body
    assert "Trade Log" in body
    assert "Scanner Status" in body
    assert "Backtest Results" in body


def test_index_references_static_js(client):
    r = client.get("/")
    assert b"/static/dashboard.js" in r.data


def test_static_dashboard_js_served(client):
    r = client.get("/static/dashboard.js")
    assert r.status_code == 200
    assert b"refreshAll" in r.data


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["db_exists"] is True


# ──────────────────────────── /api/overview ────────────────────────────


def test_api_overview_aggregates_portfolios(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    data = r.get_json()

    # Cash: 11500 + 8500 + 10100 = 30100; plus 1 OPEN trade with position_size=1000
    # Equity = cash + open_value = 30100 + 1000 = 31100
    assert data["total_balance"] == pytest.approx(31100.0)
    assert data["total_initial"] == pytest.approx(30000.0)
    # total_pnl = realized (100) + unrealized (from open ETH trade)
    assert data["is_profitable"] is True
    assert data["open_positions"] == 1  # one OPEN trade above
    assert "markets_tracked" in data
    assert data["markets_tracked"] >= 3
    assert "uptime_seconds" in data
    assert "current_time" in data


def test_api_overview_empty_db(tmp_db_path):
    """With a fresh empty DB (schema present but no rows), the app should
    still return zeros rather than 500."""
    init_db(tmp_db_path)
    app = dashboard.create_app(db_path=tmp_db_path, backtest_dir="nonexistent_dir")
    client = app.test_client()

    r = client.get("/api/overview")
    assert r.status_code == 200
    data = r.get_json()
    assert data["total_balance"] == 0
    assert data["total_pnl"] == 0
    assert data["open_positions"] == 0


def test_api_overview_missing_db_file_is_graceful(tmp_path):
    """If the DB file doesn't exist, APIs should still respond 200 with zeros
    rather than crashing."""
    fake_path = str(tmp_path / "does_not_exist.db")
    app = dashboard.create_app(db_path=fake_path, backtest_dir=str(tmp_path))
    client = app.test_client()

    r = client.get("/api/overview")
    assert r.status_code == 200
    data = r.get_json()
    assert data["total_balance"] == 0
    assert data["open_positions"] == 0


# ──────────────────────────── /api/markets ────────────────────────────


def test_api_markets_returns_all_configured_markets(client):
    r = client.get("/api/markets")
    assert r.status_code == 200
    rows = r.get_json()
    assert isinstance(rows, list)
    assert len(rows) >= 3  # at least the seeded ones (usually all 12)

    symbols = {row["symbol"] for row in rows}
    for must_have in ("BTC-USD", "ETH-USD", "SPY"):
        assert must_have in symbols

    # Each card has required keys
    sample = rows[0]
    for key in (
        "symbol", "display_name", "category", "current_balance", "initial_balance",
        "total_pnl", "pnl_pct", "total_trades", "max_drawdown_pct",
        "circuit_breaker", "sparkline", "latest_signal",
    ):
        assert key in sample


def test_api_markets_flags_circuit_breaker(client):
    rows = client.get("/api/markets").get_json()
    eth = next(r for r in rows if r["symbol"] == "ETH-USD")
    assert eth["circuit_breaker"] is True


def test_api_markets_sparkline_populated(client):
    rows = client.get("/api/markets").get_json()
    btc = next(r for r in rows if r["symbol"] == "BTC-USD")
    assert isinstance(btc["sparkline"], list)
    assert len(btc["sparkline"]) == 10
    # should be ordered oldest→newest; we wrote 10k, 10150, 10300, ... 11350
    assert btc["sparkline"][0] < btc["sparkline"][-1]


def test_api_markets_latest_signal(client):
    rows = client.get("/api/markets").get_json()
    btc = next(r for r in rows if r["symbol"] == "BTC-USD")
    assert btc["latest_signal"] is not None
    assert btc["latest_signal"]["symbol"] == "BTC-USD"
    assert btc["latest_signal"]["strategy"] == "momentum"


def test_api_markets_no_trades_yields_null_signal(client):
    # find any symbol that wasn't seeded with a trade
    rows = client.get("/api/markets").get_json()
    unseeded = [r for r in rows if r["symbol"] not in ("BTC-USD", "ETH-USD", "SPY")]
    if unseeded:
        assert unseeded[0]["latest_signal"] is None


# ──────────────────────────── /api/strategies ────────────────────────────


def test_api_strategies_aggregates(client):
    r = client.get("/api/strategies")
    assert r.status_code == 200
    rows = r.get_json()
    assert isinstance(rows, list)
    assert len(rows) == 3  # momentum, mean_reversion, multi_factor

    strategies = {row["strategy"]: row for row in rows}
    mo = strategies["momentum"]
    assert mo["total_trades"] == 5
    assert mo["winning_trades"] == 3
    assert mo["losing_trades"] == 2
    assert mo["total_pnl"] == pytest.approx(500.0)
    assert mo["win_rate"] == pytest.approx(0.6)
    assert mo["avg_win"] == pytest.approx(250.0)
    assert mo["avg_loss"] == pytest.approx(100.0)


def test_api_strategies_flags_best_worst(client):
    rows = client.get("/api/strategies").get_json()
    flagged_best = [r for r in rows if r.get("is_best")]
    flagged_worst = [r for r in rows if r.get("is_worst")]
    assert len(flagged_best) == 1
    assert len(flagged_worst) == 1
    assert flagged_best[0]["strategy"] == "momentum"       # highest P&L
    assert flagged_worst[0]["strategy"] == "mean_reversion"  # lowest P&L


def test_api_strategies_empty(tmp_db_path):
    init_db(tmp_db_path)  # no strategy_performance rows
    app = dashboard.create_app(db_path=tmp_db_path, backtest_dir="nonexistent")
    rows = app.test_client().get("/api/strategies").get_json()
    assert rows == []


# ──────────────────────────── /api/trades ────────────────────────────


def test_api_trades_returns_recent(client):
    r = client.get("/api/trades")
    assert r.status_code == 200
    rows = r.get_json()
    assert isinstance(rows, list)
    assert len(rows) == 5  # we seeded 5
    # Every row has the expected slim shape
    for t in rows:
        for key in ("id", "symbol", "direction", "strategy",
                    "entry_price", "pnl", "status", "entry_time"):
            assert key in t


def test_api_trades_limit_clamp(client):
    rows = client.get("/api/trades?limit=2").get_json()
    assert len(rows) == 2


def test_api_trades_invalid_limit_falls_back(client):
    rows = client.get("/api/trades?limit=not-a-number").get_json()
    assert len(rows) == 5


def test_api_trades_statuses_present(client):
    rows = client.get("/api/trades").get_json()
    statuses = {r["status"] for r in rows}
    assert "OPEN" in statuses
    assert "CLOSED" in statuses
    assert "STOPPED" in statuses


# ──────────────────────────── /api/scanner ────────────────────────────


def test_api_scanner_status(client):
    r = client.get("/api/scanner")
    assert r.status_code == 200
    data = r.get_json()
    for key in ("last_scan_time", "recent_scans", "total_snapshots",
                "snapshots_last_hour", "signals_last_hour",
                "markets_scanned_last_hour"):
        assert key in data

    assert data["total_snapshots"] == 20  # 10 BTC + 10 ETH
    assert isinstance(data["recent_scans"], list)
    assert len(data["recent_scans"]) >= 2
    # Recent scans should include both seeded symbols
    seen = {r["symbol"] for r in data["recent_scans"]}
    assert "BTC-USD" in seen
    assert "ETH-USD" in seen


# ──────────────────────────── /api/backtests ────────────────────────────


def test_api_backtests_reads_newest_file(client):
    r = client.get("/api/backtests")
    assert r.status_code == 200
    rows = r.get_json()
    assert isinstance(rows, list)
    assert len(rows) == 2
    symbols = {row["symbol"] for row in rows}
    assert symbols == {"BTC-USD", "ETH-USD"}


def test_api_backtests_sorted_by_return(client):
    rows = client.get("/api/backtests").get_json()
    # Should be sorted descending by total_return_pct
    returns = [r["total_return_pct"] for r in rows]
    assert returns == sorted(returns, reverse=True)


def test_api_backtests_missing_dir(tmp_db_path, tmp_path):
    init_db(tmp_db_path)
    fake_dir = str(tmp_path / "no_such_dir")
    app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=fake_dir)
    rows = app.test_client().get("/api/backtests").get_json()
    assert rows == []


def test_api_backtests_malformed_file_ignored(tmp_db_path, tmp_path):
    init_db(tmp_db_path)
    d = tmp_path / "backtest_results"
    d.mkdir()
    (d / "bad.json").write_text("not valid json {")
    (d / "empty.json").write_text("{}")  # no "results"
    app = dashboard.create_app(db_path=tmp_db_path, backtest_dir=str(d))
    r = app.test_client().get("/api/backtests")
    assert r.status_code == 200
    assert r.get_json() == []


# ──────────────────────────── helper-level tests ────────────────────────────


def test_build_overview_math(seeded_db):
    ov = dashboard.build_overview(seeded_db)
    # Equity = cash (30100) + open position value (1000) = 31100
    assert ov["total_balance"] == pytest.approx(31100.0)
    # total_pnl includes realized + unrealized
    assert ov["open_positions"] == 1
    assert ov["is_profitable"] is True


def test_build_market_grid_structure(seeded_db):
    grid = dashboard.build_market_grid(seeded_db)
    assert len(grid) >= 3
    btc = next(c for c in grid if c["symbol"] == "BTC-USD")
    assert btc["current_balance"] == pytest.approx(11500.0)
    assert btc["pnl_pct"] == pytest.approx(15.0)  # 1500 / 10000
    assert btc["latest_signal"]["strategy"] == "momentum"


def test_build_strategy_table_aggregates(seeded_db):
    rows = dashboard.build_strategy_table(seeded_db)
    strategies = {r["strategy"] for r in rows}
    assert strategies == {"momentum", "mean_reversion", "multi_factor"}


def test_build_trade_log_limits(seeded_db):
    rows = dashboard.build_trade_log(seeded_db, limit=2)
    assert len(rows) == 2
    # Ordered by entry_time DESC — all have same t_now, but we still get 2
    for r in rows:
        assert "pnl" in r and "status" in r


def test_load_markets_meta_returns_dict():
    meta = dashboard._load_markets_meta()
    assert isinstance(meta, dict)
    assert len(meta) >= 3


def test_create_app_uses_defaults(tmp_path, monkeypatch):
    # create_app with no args should still succeed (won't crash even if default
    # DB doesn't exist — queries degrade gracefully).
    monkeypatch.chdir(tmp_path)
    app = dashboard.create_app()
    assert app is not None
    assert app.config["DB_PATH"] == dashboard.DEFAULT_DB_PATH
    assert app.config["BACKTEST_DIR"] == dashboard.DEFAULT_BACKTEST_DIR
