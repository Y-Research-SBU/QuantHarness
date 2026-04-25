"""End-to-end tournament tests covering Mode A/B/C plumbing together.

These tests do NOT spawn real subprocesses — they reuse the existing
PaperTradingEngine + dashboard app to verify isolation + leaderboard
+ judging on a synthetic 4-instance tournament.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import pytest

import dashboard
import data_cache
import data_daemon
import profile_loader
import run_instance
import tournament_judge
from data_cache import OHLCVCache
from db_schema import init_db
from instances.profile import Profile


# ──────────────────────────────────────────────────────────────────────
# Tournament fixture: 4 isolated instances
# ──────────────────────────────────────────────────────────────────────


def _seed_db_with_trades(path: str, sym: str, equity_curve: List[float], trades_pnl: List[float]):
    """Schema mirror of paper_trading.py's relevant tables."""
    conn = init_db(path)
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    final_eq = equity_curve[-1] if equity_curve else 10000.0
    pnl_total = final_eq - 10000.0
    n_trades = len(trades_pnl)
    n_wins = sum(1 for p in trades_pnl if p > 0)
    n_losses = n_trades - n_wins

    conn.execute(
        """INSERT INTO portfolios
           (symbol, initial_balance, current_balance, total_pnl,
            total_trades, winning_trades, losing_trades, consecutive_losses,
            max_drawdown, peak_balance, is_circuit_breaker_active, daily_pnl)
           VALUES ('__MASTER__', 10000.0, ?, ?, ?, ?, ?, 0, 0.0, ?, 0, 0.0)""",
        (final_eq, pnl_total, n_trades, n_wins, n_losses, max(equity_curve, default=10000.0)),
    )
    conn.execute(
        """INSERT INTO portfolios
           (symbol, initial_balance, current_balance, total_pnl,
            total_trades, winning_trades, losing_trades, consecutive_losses,
            max_drawdown, peak_balance, is_circuit_breaker_active, daily_pnl)
           VALUES (?, 0.0, 0.0, ?, ?, ?, ?, 0, 0.0, 0.0, 0, 0.0)""",
        (sym, pnl_total, n_trades, n_wins, n_losses),
    )
    for i, p in enumerate(trades_pnl):
        conn.execute(
            """INSERT INTO trades
               (symbol, timeframe, strategy, direction, entry_price, exit_price,
                position_size, quantity, pnl, pnl_pct, status, entry_time, exit_time)
               VALUES (?, '1h', ?, 'LONG', 100.0, 110.0, 1000.0, 10.0, ?, ?, 'CLOSED', ?, ?)""",
            (sym, "momentum" if i % 2 == 0 else "ema_crossover", p, p / 100.0, now, now),
        )
    for i, e in enumerate(equity_curve):
        conn.execute(
            "INSERT INTO portfolio_snapshots "
            "(symbol, balance, total_pnl, open_positions, drawdown_pct, snapshot_time) "
            "VALUES ('__MASTER__', ?, ?, 0, 0.0, datetime('now', ?))",
            (e, e - 10000.0, f"-{len(equity_curve) - i} hour"),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def tournament(tmp_path, monkeypatch):
    """Build 4 isolated instances on disk."""
    inst = tmp_path / "instances"
    inst.mkdir()

    plan = [
        # name, symbol, equity_curve, trades_pnl
        ("alpha", "BTC-USD", [10000, 10080, 10180, 10260, 10340, 10410, 10500], [50, 100, -20, 80, 90]),
        ("bravo", "ETH-USD", [10000, 9950, 10000, 9900, 9950, 9870, 9800], [-50, 30, -80, -50]),
        ("charlie", "AAPL", [10000, 10010, 10020, 10005, 10030, 10025, 10050], [10, -5, 25, 20]),
        ("delta", "EURUSD=X", [10000, 9990, 10010, 9985, 10020, 9990, 10015], [-10, 20, -15, 20]),
    ]
    state_paths = {}
    for name, sym, eq, trades in plan:
        d = inst / name
        d.mkdir()
        db_file = tmp_path / f"paper_trades_{name}.db"
        (d / "profile.py").write_text(
            "from instances.profile import Profile\n"
            f"PROFILE = Profile(name='{name}', db_path='{db_file}', universe=['{sym}'])\n"
        )
        _seed_db_with_trades(str(db_file), sym, eq, trades)
        state_p = d / "state.json"
        state_p.write_text(json.dumps({
            "name": name,
            "pid": os.getpid(),
            "started_at": "2026-04-19",
            "last_heartbeat": "2026-04-25T13:00:00",
            "db_path": str(db_file),
            "equity": eq[-1],
            "open_positions": 0,
            "realized_pnl": eq[-1] - 10000.0,
            "total_trades": len(trades),
        }))
        state_paths[name] = state_p

    monkeypatch.setattr(profile_loader, "INSTANCES_DIR", inst)
    return {"dir": inst, "tmp_path": tmp_path, "state_paths": state_paths}


@pytest.fixture
def app(tournament, tmp_path):
    main_db = tmp_path / "paper_trades_main.db"
    init_db(str(main_db)).close()
    a = dashboard.create_app(db_path=str(main_db), backtest_dir=str(tmp_path))
    a.config["TESTING"] = True
    return a


@pytest.fixture
def client(app):
    return app.test_client()


# ──────────────────────────────────────────────────────────────────────
# 1. Multi-instance discovery + isolation
# ──────────────────────────────────────────────────────────────────────


def test_discover_all_four_instances(tournament):
    loader = profile_loader.ProfileLoader(instances_dir=tournament["dir"])
    names = loader.list_profiles()
    assert set(names) == {"alpha", "bravo", "charlie", "delta"}


def test_instances_endpoint_lists_four(client):
    rows = client.get("/api/instances").get_json()
    assert {r["name"] for r in rows} == {"alpha", "bravo", "charlie", "delta"}


def test_dbs_are_distinct_files(tournament):
    files = list(tournament["tmp_path"].glob("paper_trades_*.db"))
    assert len(files) == 4


def test_per_instance_portfolio_isolation(client):
    a = client.get("/api/instance/alpha/portfolio").get_json()
    b = client.get("/api/instance/bravo/portfolio").get_json()
    a_syms = {t["symbol"] for t in a["trades"]}
    b_syms = {t["symbol"] for t in b["trades"]}
    assert a_syms == {"BTC-USD"}
    assert b_syms == {"ETH-USD"}
    assert a["db_path"] != b["db_path"]


def test_unknown_instance_returns_404(client):
    r = client.get("/api/instance/zzzzzz/portfolio")
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# 2. Tournament leaderboard
# ──────────────────────────────────────────────────────────────────────


def test_tournament_leaderboard_endpoint(client):
    r = client.get("/api/tournament")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["days"] == 7
    assert len(payload["instances"]) == 4
    names = {row["name"] for row in payload["instances"]}
    assert names == {"alpha", "bravo", "charlie", "delta"}


def test_tournament_leaderboard_includes_sharpe(client):
    payload = client.get("/api/tournament").get_json()
    rows = {r["name"]: r for r in payload["instances"]}
    # alpha is the steady winner; bravo is declining.
    assert rows["alpha"]["sharpe_7d"] > rows["bravo"]["sharpe_7d"]


def test_tournament_leaderboard_includes_attribution(client):
    payload = client.get("/api/tournament").get_json()
    rows = {r["name"]: r for r in payload["instances"]}
    assert "attribution" in rows["alpha"]
    strategies = {a["strategy"] for a in rows["alpha"]["attribution"]}
    assert strategies <= {"momentum", "ema_crossover"}


def test_tournament_leaderboard_includes_equity_series(client):
    payload = client.get("/api/tournament").get_json()
    rows = {r["name"]: r for r in payload["instances"]}
    series = rows["alpha"]["equity_series"]
    assert len(series) == 7
    assert all("t" in pt and "equity" in pt for pt in series)


def test_tournament_view_renders(client):
    r = client.get("/tournament")
    assert r.status_code == 200
    assert b"Leaderboard" in r.data
    assert b"Equity Curves" in r.data


def test_tournament_view_has_chart_canvas(client):
    r = client.get("/tournament")
    assert b'id="equity-chart"' in r.data


# ──────────────────────────────────────────────────────────────────────
# 3. Cache integration
# ──────────────────────────────────────────────────────────────────────


def test_cache_disabled_via_env(monkeypatch):
    monkeypatch.setenv("QUANTAGENT_CACHE_DISABLED", "1")
    data_cache.reset_cache()
    c = data_cache.get_cache()
    assert c.is_connected() is False


def test_data_fetcher_falls_back_when_cache_disconnected(monkeypatch):
    """fetch_market_data must call yfinance when cache is offline."""
    import data_fetcher
    called = {"yf": 0}

    def fake_yf_download(*a, **kw):
        called["yf"] += 1
        return pd.DataFrame({
            "Datetime": pd.to_datetime(["2026-04-25 12:00:00"]),
            "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1.0],
        }).set_index("Datetime")

    # Force cache off, monkeypatch yfinance.
    monkeypatch.setenv("QUANTAGENT_CACHE_DISABLED", "1")
    data_cache.reset_cache()
    monkeypatch.setattr(data_fetcher.yf, "download", fake_yf_download)
    df = data_fetcher.fetch_market_data("BTC-USD", "1h", bars=1)
    assert called["yf"] == 1
    assert not df.empty


def test_data_fetcher_uses_cache_when_present(monkeypatch):
    import data_fetcher
    sample = pd.DataFrame({
        "Datetime": pd.to_datetime(["2026-04-25 12:00:00"]),
        "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1.0],
    })

    class _FakeCache:
        def __init__(self):
            self.gets = 0
        def is_connected(self):
            return True
        def get(self, *a, **kw):
            self.gets += 1
            return sample
        def set(self, *a, **kw):
            return True

    fc = _FakeCache()
    monkeypatch.setattr(data_fetcher, "get_cache", lambda: fc)
    yf_calls = {"n": 0}

    def fake_yf(*a, **kw):
        yf_calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(data_fetcher.yf, "download", fake_yf)
    out = data_fetcher.fetch_market_data("BTC-USD", "1h", bars=1)
    assert fc.gets == 1
    assert yf_calls["n"] == 0
    assert not out.empty


def test_daemon_run_once_writes_for_every_cell(monkeypatch):
    sample = pd.DataFrame({
        "Datetime": pd.to_datetime(["2026-04-25 12:00:00"]),
        "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1.0],
    })
    monkeypatch.setattr(data_daemon, "fetch_market_data", lambda *a, **kw: sample)

    class FakeClient:
        def __init__(self):
            self.store = {}
        def setex(self, k, t, v):
            self.store[k] = v
        def get(self, k):
            return self.store.get(k)
        def ping(self):
            return True

    cache = OHLCVCache(client=FakeClient())

    from dataclasses import dataclass, field
    @dataclass
    class _M:
        timeframes: list = field(default_factory=lambda: ["1h"])
    fake_markets = {"BTC-USD": _M(), "AAPL": _M()}

    daemon = data_daemon.DataDaemon(cache=cache, markets=fake_markets)
    stats = daemon.run_once()
    assert stats["success"] == 2


# ──────────────────────────────────────────────────────────────────────
# 4. Promotion (Mode C.3)
# ──────────────────────────────────────────────────────────────────────


def test_judge_picks_alpha_as_winner(tournament, tmp_path, monkeypatch):
    monkeypatch.setattr(tournament_judge, "TOURNAMENT_LOG", tmp_path / "log.json")
    monkeypatch.setattr(tournament_judge, "BRAIN_DIR", tmp_path / "brain")
    rec = tournament_judge.run(
        instances_dir=tournament["dir"],
        today=datetime(2026, 4, 30, tzinfo=timezone.utc),
    )
    assert rec["winner"]["name"] == "alpha"


def test_judge_creates_champion_directory(tournament, tmp_path, monkeypatch):
    monkeypatch.setattr(tournament_judge, "TOURNAMENT_LOG", tmp_path / "log.json")
    monkeypatch.setattr(tournament_judge, "BRAIN_DIR", tmp_path / "brain")
    today = datetime(2026, 5, 1, tzinfo=timezone.utc)
    tournament_judge.run(instances_dir=tournament["dir"], today=today)
    new = tournament["dir"] / "champion-20260501"
    assert new.exists()
    assert (new / "profile.py").exists()


def test_judge_logs_to_brain(tournament, tmp_path, monkeypatch):
    monkeypatch.setattr(tournament_judge, "TOURNAMENT_LOG", tmp_path / "log.json")
    brain = tmp_path / "brain"
    monkeypatch.setattr(tournament_judge, "BRAIN_DIR", brain)
    tournament_judge.run(
        instances_dir=tournament["dir"],
        today=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )
    files = list(brain.glob("*.json"))
    assert len(files) >= 1


def test_judge_logs_to_tournament_log(tournament, tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    monkeypatch.setattr(tournament_judge, "TOURNAMENT_LOG", log_path)
    monkeypatch.setattr(tournament_judge, "BRAIN_DIR", tmp_path / "brain")
    tournament_judge.run(
        instances_dir=tournament["dir"],
        today=datetime(2026, 5, 3, tzinfo=timezone.utc),
    )
    rows = json.loads(log_path.read_text())
    assert len(rows) == 1
    assert rows[0]["winner"]["name"] == "alpha"


# ──────────────────────────────────────────────────────────────────────
# 5. Profile + scanner integration smoke
# ──────────────────────────────────────────────────────────────────────


def test_profile_filter_blocks_disallowed_cells():
    p = Profile(name="x", cell_overrides=[("AAPL", "breakout")])
    assert p.is_cell_allowed("AAPL", "breakout") is True
    assert p.is_cell_allowed("AAPL", "momentum") is False
    assert p.is_cell_allowed("BTC-USD", "breakout") is False


def test_resolve_symbols_for_cell_overrides_profile():
    p = Profile(name="x", cell_overrides=[("AAPL", "breakout"), ("TSLA", "bb_squeeze")])
    out = run_instance.resolve_symbols(p)
    assert sorted(out) == ["AAPL", "TSLA"]


def test_state_files_exist_for_each_instance(tournament):
    for name in ["alpha", "bravo", "charlie", "delta"]:
        assert (tournament["dir"] / name / "state.json").exists()


def test_state_files_pids_alive(tournament):
    loader = profile_loader.ProfileLoader(instances_dir=tournament["dir"])
    for name in ["alpha", "bravo", "charlie", "delta"]:
        d = loader.describe(name)
        # Seeded with our own pid -> should be running.
        assert d["status"] == "running"


def test_leaderboard_handles_missing_db(tmp_path, monkeypatch):
    """Leaderboard must not crash when an instance's DB doesn't exist yet."""
    inst = tmp_path / "instances"
    inst.mkdir()
    d = inst / "ghost"
    d.mkdir()
    (d / "profile.py").write_text(
        "from instances.profile import Profile\n"
        "PROFILE = Profile(name='ghost', db_path='paper_trades_ghost.db')\n"
    )
    (d / "state.json").write_text(json.dumps({
        "name": "ghost", "pid": 0, "db_path": "/nonexistent/path.db",
        "equity": 10000.0, "open_positions": 0, "realized_pnl": 0.0,
    }))
    monkeypatch.setattr(profile_loader, "INSTANCES_DIR", inst)
    payload = dashboard.build_tournament_leaderboard(days=7)
    assert len(payload["instances"]) == 1
    row = payload["instances"][0]
    assert row["sharpe_7d"] == 0.0
    assert row["equity_series"] == []


def test_attribution_groups_by_strategy(client):
    rows = client.get("/api/tournament").get_json()["instances"]
    alpha = next(r for r in rows if r["name"] == "alpha")
    seen_strats = {a["strategy"] for a in alpha["attribution"]}
    assert seen_strats <= {"momentum", "ema_crossover"}


def test_leaderboard_winner_has_positive_pnl(client):
    rows = client.get("/api/tournament").get_json()["instances"]
    alpha = next(r for r in rows if r["name"] == "alpha")
    bravo = next(r for r in rows if r["name"] == "bravo")
    assert alpha["realized_pnl"] > 0
    assert bravo["realized_pnl"] < 0
