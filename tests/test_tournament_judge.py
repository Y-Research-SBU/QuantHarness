"""Tests for tournament_judge: metrics, selection, promotion, logging."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import tournament_judge


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _seed_db(path: str, equity_series):
    """Create a minimal portfolios+portfolio_snapshots schema and insert series."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS portfolios (
            symbol TEXT PRIMARY KEY,
            current_balance REAL,
            initial_balance REAL,
            total_pnl REAL
        );
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            balance REAL NOT NULL,
            total_pnl REAL NOT NULL DEFAULT 0,
            open_positions INTEGER NOT NULL DEFAULT 0,
            drawdown_pct REAL NOT NULL DEFAULT 0.0,
            snapshot_time TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.execute("INSERT OR REPLACE INTO portfolios VALUES ('__MASTER__', ?, 10000, 0)",
                 (equity_series[-1] if equity_series else 10000.0,))
    for v in equity_series:
        conn.execute(
            "INSERT INTO portfolio_snapshots (symbol, balance, snapshot_time) "
            "VALUES ('__MASTER__', ?, datetime('now'))",
            (v,),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def tournament_dir(tmp_path):
    """A self-contained instances/ tree with state.json + db for two players."""
    inst = tmp_path / "instances"
    inst.mkdir()

    def _make(name, equity):
        d = inst / name
        d.mkdir()
        (d / "profile.py").write_text(
            "from instances.profile import Profile\n"
            f"PROFILE = Profile(name='{name}', db_path='paper_trades_{name}.db')\n"
        )
        db = tmp_path / f"paper_trades_{name}.db"
        _seed_db(str(db), equity)
        (d / "state.json").write_text(json.dumps({
            "name": name,
            "pid": 1,
            "started_at": "2026-04-19",
            "db_path": str(db),
            "equity": equity[-1] if equity else 10000.0,
            "realized_pnl": (equity[-1] - 10000.0) if equity else 0,
            "total_trades": 5,
        }))
        return d

    _make("alpha", [10000, 10100, 10200, 10300, 10400, 10550])  # steady winner
    _make("bravo", [10000, 9900, 10100, 9800, 10000, 9700])     # noisy / declining
    return inst


# ──────────────────────────────────────────────────────────────────────
# Sharpe math
# ──────────────────────────────────────────────────────────────────────


def test_compute_sharpe_handles_empty():
    assert tournament_judge.compute_sharpe([]) == 0.0


def test_compute_sharpe_handles_too_short():
    assert tournament_judge.compute_sharpe([10000.0]) == 0.0


def test_compute_sharpe_constant_series_is_zero():
    assert tournament_judge.compute_sharpe([10000.0] * 10) == 0.0


def test_compute_sharpe_positive_for_uptrend():
    s = tournament_judge.compute_sharpe([10000, 10100, 10200, 10300, 10400, 10500])
    assert s > 0.0


def test_compute_sharpe_negative_for_downtrend():
    s = tournament_judge.compute_sharpe([10000, 9900, 9800, 9700, 9600, 9500])
    assert s < 0.0


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────


def test_evaluate_instance_returns_metrics(tournament_dir):
    m = tournament_judge.evaluate_instance("alpha", tournament_dir)
    assert m["name"] == "alpha"
    assert m["snapshots"] == 6
    assert "sharpe_7d" in m


def test_evaluate_instance_missing_state_returns_error(tmp_path):
    inst = tmp_path / "instances"
    inst.mkdir()
    (inst / "lonely").mkdir()
    m = tournament_judge.evaluate_instance("lonely", inst)
    assert "error" in m


# ──────────────────────────────────────────────────────────────────────
# Selection
# ──────────────────────────────────────────────────────────────────────


def test_select_winner_picks_highest_sharpe(tournament_dir):
    metrics = [
        tournament_judge.evaluate_instance("alpha", tournament_dir),
        tournament_judge.evaluate_instance("bravo", tournament_dir),
    ]
    winner = tournament_judge.select_winner(metrics)
    assert winner is not None
    assert winner["name"] == "alpha"


def test_select_winner_excludes_short_series():
    metrics = [
        {"name": "x", "sharpe_7d": 5.0, "snapshots": 1},
        {"name": "y", "sharpe_7d": 0.1, "snapshots": 50},
    ]
    winner = tournament_judge.select_winner(metrics)
    assert winner["name"] == "y"


def test_select_winner_returns_none_when_all_invalid():
    assert tournament_judge.select_winner([{"name": "x", "error": "nope"}]) is None


def test_select_winner_returns_none_for_empty():
    assert tournament_judge.select_winner([]) is None


# ──────────────────────────────────────────────────────────────────────
# Promotion
# ──────────────────────────────────────────────────────────────────────


def test_promote_creates_new_profile(tournament_dir):
    new_name = "champion-20260425"
    path = tournament_judge.promote("alpha", new_name, tournament_dir, dry_run=False)
    assert path.exists()
    body = path.read_text()
    assert new_name in body
    assert "alpha" in body
    assert (tournament_dir / new_name / "__init__.py").exists()


def test_promote_dry_run_does_not_write(tournament_dir):
    new_name = "champion-20260101"
    path = tournament_judge.promote("alpha", new_name, tournament_dir, dry_run=True)
    assert not path.exists()


def test_promote_missing_winner_raises(tournament_dir):
    with pytest.raises(FileNotFoundError):
        tournament_judge.promote("ghost", "champion-x", tournament_dir)


def test_champion_name_format():
    n = tournament_judge.champion_name(datetime(2026, 4, 25, tzinfo=timezone.utc))
    assert n == "champion-20260425"


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────


def test_append_log_creates_and_appends(tmp_path):
    log_path = tmp_path / "log.json"
    tournament_judge.append_log({"a": 1}, log_path)
    tournament_judge.append_log({"b": 2}, log_path)
    rows = json.loads(log_path.read_text())
    assert len(rows) == 2
    assert rows[0]["a"] == 1


def test_append_log_recovers_from_corrupt_file(tmp_path):
    log_path = tmp_path / "log.json"
    log_path.write_text("not json {")
    tournament_judge.append_log({"a": 1}, log_path)
    rows = json.loads(log_path.read_text())
    assert rows == [{"a": 1}]


def test_write_brain_record_creates_dir(tmp_path):
    brain = tmp_path / "brain"
    out = tournament_judge.write_brain_record({"x": 1}, brain)
    assert out is not None
    assert out.exists()
    assert json.loads(out.read_text())["x"] == 1


# ──────────────────────────────────────────────────────────────────────
# End-to-end
# ──────────────────────────────────────────────────────────────────────


def test_run_full_flow_promotes_winner(tournament_dir, tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    brain_path = tmp_path / "brain"
    monkeypatch.setattr(tournament_judge, "TOURNAMENT_LOG", log_path)
    monkeypatch.setattr(tournament_judge, "BRAIN_DIR", brain_path)
    rec = tournament_judge.run(
        instances_dir=tournament_dir,
        today=datetime(2026, 4, 26, tzinfo=timezone.utc),
    )
    assert rec["winner"]["name"] == "alpha"
    assert rec["promoted_to"] == "champion-20260426"
    assert (tournament_dir / "champion-20260426" / "profile.py").exists()
    assert log_path.exists()


def test_run_dry_run_does_not_create_directory(tournament_dir, tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    brain_path = tmp_path / "brain"
    monkeypatch.setattr(tournament_judge, "TOURNAMENT_LOG", log_path)
    monkeypatch.setattr(tournament_judge, "BRAIN_DIR", brain_path)
    rec = tournament_judge.run(
        instances_dir=tournament_dir,
        dry_run=True,
        today=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )
    assert rec["dry_run"] is True
    assert not (tournament_dir / "champion-20260427").exists()


def test_run_skips_existing_champion_dirs(tournament_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(tournament_judge, "TOURNAMENT_LOG", tmp_path / "log.json")
    monkeypatch.setattr(tournament_judge, "BRAIN_DIR", tmp_path / "brain")
    # Pre-existing champion shouldn't enter the candidate pool.
    champ = tournament_dir / "champion-20260101"
    champ.mkdir()
    (champ / "profile.py").write_text(
        "from instances.profile import Profile\n"
        "PROFILE = Profile(name='champion-20260101', db_path='x.db')\n"
    )
    rec = tournament_judge.run(
        instances_dir=tournament_dir,
        today=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    names = [m["name"] for m in rec["evaluated"]]
    assert "champion-20260101" not in names
