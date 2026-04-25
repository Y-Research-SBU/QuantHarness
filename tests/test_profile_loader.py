"""Tests for ProfileLoader."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from instances.profile import Profile
from profile_loader import ProfileLoader


@pytest.fixture
def tmp_loader(tmp_path):
    """Build a loader that points at a fresh instances dir.

    We do NOT add tmp_path to sys.path — each profile's body imports
    ``from instances.profile import Profile`` which we want to resolve
    to the real, repo-level Profile class.
    """
    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()

    def _make_profile(name: str, body: str = "") -> Path:
        d = instances_dir / name
        d.mkdir()
        body = body or f"""
from instances.profile import Profile
PROFILE = Profile(name='{name}', db_path='paper_trades_{name}.db')
"""
        (d / "profile.py").write_text(body)
        return d

    yield instances_dir, _make_profile


def test_list_profiles_empty(tmp_path):
    loader = ProfileLoader(instances_dir=tmp_path / "nope")
    assert loader.list_profiles() == []


def test_list_profiles_finds_directories(tmp_loader):
    instances_dir, make = tmp_loader
    make("alpha")
    make("bravo")
    loader = ProfileLoader(instances_dir=instances_dir)
    assert loader.list_profiles() == ["alpha", "bravo"]


def test_list_profiles_skips_dirs_without_profile_py(tmp_loader):
    instances_dir, make = tmp_loader
    make("alpha")
    (instances_dir / "no_profile").mkdir()
    loader = ProfileLoader(instances_dir=instances_dir)
    assert "no_profile" not in loader.list_profiles()


def test_load_returns_profile(tmp_loader):
    instances_dir, make = tmp_loader
    make("charlie")
    loader = ProfileLoader(instances_dir=instances_dir)
    p = loader.load("charlie")
    assert isinstance(p, Profile)
    assert p.name == "charlie"
    assert p.db_path == "paper_trades_charlie.db"


def test_load_missing_raises(tmp_loader):
    instances_dir, _ = tmp_loader
    loader = ProfileLoader(instances_dir=instances_dir)
    with pytest.raises(FileNotFoundError):
        loader.load("ghost")


def test_overrides_merge_simple_fields(tmp_loader):
    instances_dir, make = tmp_loader
    d = make("delta")
    overrides = {"position_size_multiplier": 3.0, "regime_filter_enabled": False}
    (d / "overrides.json").write_text(json.dumps(overrides))
    loader = ProfileLoader(instances_dir=instances_dir)
    p = loader.load("delta")
    assert p.position_size_multiplier == 3.0
    assert p.regime_filter_enabled is False
    # untouched defaults stay
    assert p.entry_threshold_multiplier == 1.0


def test_overrides_ignore_unknown_keys(tmp_loader):
    instances_dir, make = tmp_loader
    d = make("echo")
    (d / "overrides.json").write_text(json.dumps({
        "position_size_multiplier": 5.0,
        "totally_made_up_field": 99,
    }))
    loader = ProfileLoader(instances_dir=instances_dir)
    p = loader.load("echo")
    assert p.position_size_multiplier == 5.0
    assert not hasattr(p, "totally_made_up_field")


def test_overrides_cell_overrides_coerced_to_tuples(tmp_loader):
    instances_dir, make = tmp_loader
    d = make("foxtrot")
    (d / "overrides.json").write_text(json.dumps({
        "cell_overrides": [["AAPL", "breakout"], ["BTC-USD", "ema_crossover"]],
    }))
    loader = ProfileLoader(instances_dir=instances_dir)
    p = loader.load("foxtrot")
    assert p.cell_overrides == [("AAPL", "breakout"), ("BTC-USD", "ema_crossover")]
    assert p.is_cell_allowed("AAPL", "breakout") is True


def test_invalid_overrides_json_is_ignored(tmp_loader):
    instances_dir, make = tmp_loader
    d = make("golf")
    (d / "overrides.json").write_text("not json {{{")
    loader = ProfileLoader(instances_dir=instances_dir)
    p = loader.load("golf")
    # Falls back to base profile defaults.
    assert p.position_size_multiplier == 1.0


def test_read_state_returns_none_when_missing(tmp_loader):
    instances_dir, make = tmp_loader
    make("hotel")
    loader = ProfileLoader(instances_dir=instances_dir)
    assert loader.read_state("hotel") is None


def test_read_state_parses_json(tmp_loader):
    instances_dir, make = tmp_loader
    d = make("india")
    (d / "state.json").write_text(json.dumps({"pid": 42, "equity": 12000}))
    loader = ProfileLoader(instances_dir=instances_dir)
    state = loader.read_state("india")
    assert state["pid"] == 42
    assert state["equity"] == 12000


def test_is_alive_for_self_pid_is_true(tmp_loader):
    instances_dir, _ = tmp_loader
    loader = ProfileLoader(instances_dir=instances_dir)
    assert loader.is_alive(os.getpid()) is True


def test_is_alive_for_invalid_pid_is_false(tmp_loader):
    instances_dir, _ = tmp_loader
    loader = ProfileLoader(instances_dir=instances_dir)
    assert loader.is_alive(None) is False
    assert loader.is_alive(0) is False
    assert loader.is_alive(99999999) is False


def test_describe_returns_status_running_when_pid_alive(tmp_loader):
    instances_dir, make = tmp_loader
    d = make("juliet")
    (d / "state.json").write_text(json.dumps({
        "pid": os.getpid(),
        "started_at": "2026-04-25",
        "equity": 9000,
        "open_positions": 1,
        "realized_pnl": 50.0,
    }))
    loader = ProfileLoader(instances_dir=instances_dir)
    desc = loader.describe("juliet")
    assert desc["name"] == "juliet"
    assert desc["status"] == "running"
    assert desc["equity"] == 9000


def test_describe_returns_stopped_when_no_state(tmp_loader):
    instances_dir, make = tmp_loader
    make("kilo")
    loader = ProfileLoader(instances_dir=instances_dir)
    desc = loader.describe("kilo")
    assert desc["status"] == "stopped"
    assert desc["pid"] is None


def test_describe_all_returns_one_per_profile(tmp_loader):
    instances_dir, make = tmp_loader
    make("alpha")
    make("bravo")
    loader = ProfileLoader(instances_dir=instances_dir)
    out = loader.describe_all()
    assert len(out) == 2
    assert {row["name"] for row in out} == {"alpha", "bravo"}


def test_describe_all_handles_broken_profile(tmp_loader):
    instances_dir, make = tmp_loader
    make("alpha")
    # Profile with a syntax error
    bad = instances_dir / "broken"
    bad.mkdir()
    (bad / "__init__.py").write_text("")
    (bad / "profile.py").write_text("this is not python(((")
    loader = ProfileLoader(instances_dir=instances_dir)
    out = loader.describe_all()
    names = {row["name"] for row in out}
    assert "alpha" in names
    assert "broken" in names
    broken = next(r for r in out if r["name"] == "broken")
    assert broken["status"] == "error"


def test_describe_includes_profile_settings(tmp_loader):
    instances_dir, make = tmp_loader
    body = """
from instances.profile import Profile
PROFILE = Profile(
    name='lima',
    db_path='paper_trades_lima.db',
    position_size_multiplier=1.5,
    asset_categories=['crypto'],
    regime_filter_enabled=False,
)
"""
    make("lima", body=body)
    loader = ProfileLoader(instances_dir=instances_dir)
    desc = loader.describe("lima")
    assert desc["profile"]["position_size_multiplier"] == 1.5
    assert desc["profile"]["asset_categories"] == ["crypto"]
    assert desc["profile"]["regime_filter_enabled"] is False
