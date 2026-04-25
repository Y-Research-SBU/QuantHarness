"""Tests for run_instance.py — CLI parsing, profile loading, DB isolation."""

from __future__ import annotations

import importlib
import sys

import pytest

import run_instance
from instances.profile import Profile


def test_load_profile_imports_module():
    p = run_instance.load_profile("baseline")
    assert isinstance(p, Profile)
    assert p.name == "baseline"


def test_load_profile_each_shipped_profile():
    for name in ["baseline", "crypto_aggro", "forex_focus", "top25_only"]:
        p = run_instance.load_profile(name)
        assert p.name == name


def test_load_profile_missing_raises():
    with pytest.raises(FileNotFoundError):
        run_instance.load_profile("does_not_exist_zzz")


def test_arg_parser_requires_profile():
    parser = run_instance.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing --profile


def test_arg_parser_parses_profile_and_max_cycles():
    parser = run_instance.build_arg_parser()
    args = parser.parse_args(["--profile", "baseline", "--max-cycles", "3"])
    assert args.profile == "baseline"
    assert args.max_cycles == 3
    assert args.no_kronos is False
    assert args.once is False


def test_arg_parser_db_override():
    parser = run_instance.build_arg_parser()
    args = parser.parse_args(["--profile", "baseline", "--db", "/tmp/foo.db"])
    assert args.db == "/tmp/foo.db"


def test_resolve_symbols_respects_cli_override():
    p = Profile(name="x", universe=["BTC-USD"])
    # Even though profile says only BTC, an explicit override wins.
    out = run_instance.resolve_symbols(p, override=["AAPL"])
    assert out == ["AAPL"]


def test_resolve_symbols_uses_profile_universe():
    p = Profile(name="x", universe=["BTC-USD", "ETH-USD"])
    out = run_instance.resolve_symbols(p)
    # Result is a subset of the actual MARKETS keys, intersected with universe.
    from market_config import MARKETS
    expected = [s for s in MARKETS.keys() if s in {"BTC-USD", "ETH-USD"}]
    assert out == expected


def test_resolve_symbols_empty_universe_returns_all_markets():
    p = Profile(name="x")
    out = run_instance.resolve_symbols(p)
    from market_config import MARKETS
    assert set(out) == set(MARKETS.keys())


def test_resolve_symbols_with_cell_overrides():
    p = Profile(
        name="x",
        cell_overrides=[("AAPL", "breakout"), ("BTC-USD", "momentum")],
    )
    out = run_instance.resolve_symbols(p)
    assert sorted(out) == ["AAPL", "BTC-USD"]


def test_db_paths_are_unique_across_shipped_profiles():
    paths = set()
    for name in ["baseline", "crypto_aggro", "forex_focus", "top25_only"]:
        p = run_instance.load_profile(name)
        assert p.db_path not in paths, f"duplicate db_path: {p.db_path}"
        paths.add(p.db_path)


def test_db_paths_do_not_collide_with_live_paper_trades_db():
    # The live runner uses paper_trades.db. Our profiles must not target it.
    for name in ["baseline", "crypto_aggro", "forex_focus", "top25_only"]:
        p = run_instance.load_profile(name)
        assert p.db_path != "paper_trades.db"


def test_write_state_file_creates_json(tmp_path, monkeypatch):
    monkeypatch.setattr(run_instance, "INSTANCES_DIR", tmp_path)
    state = {"name": "test", "pid": 123, "started_at": "now"}
    run_instance.write_state_file("test", state)
    out = tmp_path / "test" / "state.json"
    assert out.exists()
    import json
    loaded = json.loads(out.read_text())
    assert loaded["pid"] == 123
