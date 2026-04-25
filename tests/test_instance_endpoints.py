"""Tests for /api/instances and /api/instance/<name>/portfolio endpoints.

Cross-instance DB isolation is the key invariant: data written for one
instance must not bleed into another.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

import dashboard
import profile_loader
from db_schema import init_db


@pytest.fixture
def isolated_instances(tmp_path, monkeypatch):
    """Create two fully isolated tournament instances on disk + DBs."""
    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()

    def _seed_db(path: str, equity_pnl: float, n_trades: int, sym: str):
        # Master + per-symbol analytics row + N trades
        conn = init_db(path)
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        conn.execute(
            """INSERT INTO portfolios
               (symbol, initial_balance, current_balance, total_pnl,
                total_trades, winning_trades, losing_trades, consecutive_losses,
                max_drawdown, peak_balance, is_circuit_breaker_active, daily_pnl)
               VALUES ('__MASTER__', 10000.0, ?, ?, ?, ?, 0, 0, 0.0, ?, 0, 0.0)""",
            (10000.0 + equity_pnl, equity_pnl, n_trades, n_trades, 10000.0 + equity_pnl),
        )
        conn.execute(
            """INSERT INTO portfolios
               (symbol, initial_balance, current_balance, total_pnl,
                total_trades, winning_trades, losing_trades, consecutive_losses,
                max_drawdown, peak_balance, is_circuit_breaker_active, daily_pnl)
               VALUES (?, 0.0, 0.0, ?, ?, ?, 0, 0, 0.0, 0.0, 0, 0.0)""",
            (sym, equity_pnl, n_trades, n_trades),
        )
        for i in range(n_trades):
            conn.execute(
                """INSERT INTO trades
                   (symbol, timeframe, strategy, direction, entry_price, exit_price,
                    position_size, quantity, pnl, pnl_pct, status, entry_time, exit_time)
                   VALUES (?, '1h', 'momentum', 'LONG', 100.0, 110.0, 1000.0, 10.0,
                           ?, 10.0, 'CLOSED', ?, ?)""",
                (sym, equity_pnl / max(n_trades, 1), now, now),
            )
        conn.commit()
        conn.close()

    def _make_instance(name: str, sym: str, pnl: float, trades: int):
        d = instances_dir / name
        d.mkdir()
        db_file = tmp_path / f"paper_trades_{name}.db"
        body = f"""
from instances.profile import Profile
PROFILE = Profile(name='{name}', db_path='{db_file}', universe=['{sym}'])
"""
        (d / "profile.py").write_text(body)
        _seed_db(str(db_file), pnl, trades, sym)
        # Heartbeat (running == True if pid alive). Use the test runner's PID.
        (d / "state.json").write_text(json.dumps({
            "name": name,
            "pid": os.getpid(),
            "started_at": "2026-04-25T13:00:00",
            "last_heartbeat": "2026-04-25T13:05:00",
            "db_path": str(db_file),
            "equity": 10000.0 + pnl,
            "open_positions": 0,
            "realized_pnl": pnl,
            "total_trades": trades,
        }))
        return d, str(db_file)

    a_dir, a_db = _make_instance("alpha", "BTC-USD", 250.0, 3)
    b_dir, b_db = _make_instance("bravo", "AAPL", -100.0, 2)

    # Patch the loader to point at our temp dir.
    monkeypatch.setattr(profile_loader, "INSTANCES_DIR", instances_dir)

    yield {
        "instances_dir": instances_dir,
        "alpha_db": a_db,
        "bravo_db": b_db,
    }


@pytest.fixture
def client(isolated_instances, tmp_path):
    # Main app DB — separate from any instance.
    main_db = tmp_path / "paper_trades_main.db"
    init_db(str(main_db)).close()
    app = dashboard.create_app(db_path=str(main_db), backtest_dir=str(tmp_path))
    app.config["TESTING"] = True
    return app.test_client()


# ──────────────────────────────────────────────────────────────────────
# /api/instances
# ──────────────────────────────────────────────────────────────────────


def test_instances_endpoint_returns_list(client):
    r = client.get("/api/instances")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)
    names = {row["name"] for row in data}
    assert "alpha" in names
    assert "bravo" in names


def test_instances_endpoint_running_status_true_for_self_pid(client):
    r = client.get("/api/instances")
    rows = {row["name"]: row for row in r.get_json()}
    # We seeded each state.json with our own PID.
    assert rows["alpha"]["status"] == "running"
    assert rows["bravo"]["status"] == "running"


def test_instances_endpoint_includes_equity_and_pnl(client):
    rows = {r["name"]: r for r in client.get("/api/instances").get_json()}
    assert rows["alpha"]["equity"] == 10250.0
    assert rows["alpha"]["realized_pnl"] == 250.0
    assert rows["bravo"]["equity"] == 9900.0
    assert rows["bravo"]["realized_pnl"] == -100.0


def test_instances_endpoint_includes_db_path_and_started_at(client):
    rows = {r["name"]: r for r in client.get("/api/instances").get_json()}
    assert rows["alpha"]["db_path"].endswith("paper_trades_alpha.db")
    assert rows["alpha"]["started_at"] == "2026-04-25T13:00:00"


def test_instances_endpoint_profile_metadata(client):
    rows = {r["name"]: r for r in client.get("/api/instances").get_json()}
    assert "profile" in rows["alpha"]
    assert rows["alpha"]["profile"]["universe"] == ["BTC-USD"]


# ──────────────────────────────────────────────────────────────────────
# /api/instance/<name>/portfolio
# ──────────────────────────────────────────────────────────────────────


def test_portfolio_endpoint_returns_alpha_data(client):
    r = client.get("/api/instance/alpha/portfolio")
    assert r.status_code == 200
    data = r.get_json()
    assert data["name"] == "alpha"
    assert data["db_path"].endswith("paper_trades_alpha.db")
    # Trades should reflect alpha's seeded BTC-USD only.
    syms = {t["symbol"] for t in data["trades"]}
    assert syms == {"BTC-USD"}


def test_portfolio_endpoint_returns_bravo_data(client):
    r = client.get("/api/instance/bravo/portfolio")
    assert r.status_code == 200
    data = r.get_json()
    syms = {t["symbol"] for t in data["trades"]}
    assert syms == {"AAPL"}


def test_portfolio_endpoint_unknown_instance_404(client):
    r = client.get("/api/instance/zzz/portfolio")
    assert r.status_code == 404


def test_portfolio_endpoint_isolates_dbs(client):
    a = client.get("/api/instance/alpha/portfolio").get_json()
    b = client.get("/api/instance/bravo/portfolio").get_json()
    # No cross-pollination between alpha (BTC-USD) and bravo (AAPL).
    a_syms = {t["symbol"] for t in a["trades"]}
    b_syms = {t["symbol"] for t in b["trades"]}
    assert a_syms.isdisjoint(b_syms)


def test_portfolio_endpoint_includes_overview(client):
    data = client.get("/api/instance/alpha/portfolio").get_json()
    assert "overview" in data
    assert isinstance(data["overview"], dict)


def test_portfolio_endpoint_includes_positions_and_strategies(client):
    data = client.get("/api/instance/alpha/portfolio").get_json()
    assert "positions" in data
    assert "strategies" in data
    assert "markets" in data


def test_portfolio_endpoint_returns_profile_meta(client):
    data = client.get("/api/instance/alpha/portfolio").get_json()
    assert data["profile"]["universe"] == ["BTC-USD"]


# ──────────────────────────────────────────────────────────────────────
# Index page exposes the switcher
# ──────────────────────────────────────────────────────────────────────


def test_index_includes_instance_switcher(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b'id="instance-switcher"' in r.data


def test_index_includes_tournament_link(client):
    r = client.get("/")
    assert b'/tournament' in r.data


# ──────────────────────────────────────────────────────────────────────
# Loader unit checks alongside endpoints
# ──────────────────────────────────────────────────────────────────────


def test_loader_describe_all_yields_two(isolated_instances):
    from profile_loader import ProfileLoader
    loader = ProfileLoader(instances_dir=isolated_instances["instances_dir"])
    out = loader.describe_all()
    assert len(out) == 2


def test_loader_state_files_exist(isolated_instances):
    from profile_loader import ProfileLoader
    loader = ProfileLoader(instances_dir=isolated_instances["instances_dir"])
    assert loader.state_file("alpha").exists()
    assert loader.state_file("bravo").exists()
