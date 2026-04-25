"""Tests for instances.profile.Profile and the four shipped profiles."""

from __future__ import annotations

import importlib

import pytest

from instances.profile import Profile


# ---------------------------------------------------------------------------
# Profile dataclass behavior
# ---------------------------------------------------------------------------


def test_profile_defaults_allow_everything():
    p = Profile(name="x")
    assert p.is_cell_allowed("BTC-USD", "momentum", "crypto") is True
    assert p.is_cell_allowed("AAPL", "breakout", "stocks") is True


def test_profile_universe_filter_excludes_non_listed():
    p = Profile(name="x", universe=["BTC-USD"])
    assert p.is_cell_allowed("BTC-USD", "momentum", "crypto") is True
    assert p.is_cell_allowed("ETH-USD", "momentum", "crypto") is False


def test_profile_asset_category_filter():
    p = Profile(name="x", asset_categories=["crypto"])
    assert p.is_cell_allowed("BTC-USD", "momentum", "crypto") is True
    assert p.is_cell_allowed("AAPL", "momentum", "stocks") is False


def test_profile_strategy_filter():
    p = Profile(name="x", strategies=["bb_squeeze"])
    assert p.is_cell_allowed("AAPL", "bb_squeeze", "stocks") is True
    assert p.is_cell_allowed("AAPL", "momentum", "stocks") is False


def test_profile_cell_overrides_take_precedence():
    p = Profile(
        name="x",
        universe=["AAPL"],                       # would filter
        asset_categories=["crypto"],             # would filter
        strategies=["momentum"],                 # would filter
        cell_overrides=[("BTC-USD", "ema_crossover")],
    )
    # Only the explicit override trades, regardless of other filters.
    assert p.is_cell_allowed("BTC-USD", "ema_crossover", "crypto") is True
    # AAPL/momentum is filtered OUT because cell_overrides is restrictive.
    assert p.is_cell_allowed("AAPL", "momentum", "stocks") is False


def test_profile_filter_symbols_universe():
    p = Profile(name="x", universe=["BTC-USD", "ETH-USD"])
    assert p.filter_symbols(["BTC-USD", "AAPL", "ETH-USD"]) == ["BTC-USD", "ETH-USD"]


def test_profile_filter_symbols_empty_universe_returns_all():
    p = Profile(name="x")
    assert p.filter_symbols(["BTC-USD", "AAPL"]) == ["BTC-USD", "AAPL"]


def test_profile_filter_symbols_with_cell_overrides():
    p = Profile(
        name="x",
        cell_overrides=[("AAPL", "breakout"), ("BTC-USD", "momentum")],
    )
    out = p.filter_symbols(["AAPL", "BTC-USD", "TSLA"])
    assert sorted(out) == ["AAPL", "BTC-USD"]


def test_profile_multipliers_default_to_one():
    p = Profile(name="x")
    assert p.position_size_multiplier == 1.0
    assert p.entry_threshold_multiplier == 1.0


def test_profile_db_path_default():
    p = Profile(name="x")
    assert p.db_path == "paper_trades.db"


# ---------------------------------------------------------------------------
# Shipped profiles
# ---------------------------------------------------------------------------


@pytest.fixture(params=["baseline", "crypto_aggro", "forex_focus", "top25_only"])
def shipped_profile(request):
    mod = importlib.import_module(f"instances.{request.param}.profile")
    return mod.PROFILE


def test_shipped_profile_has_unique_db_path(shipped_profile):
    # No two profiles should share the same db_path (else writes collide).
    assert shipped_profile.db_path.endswith(".db")
    assert shipped_profile.name in shipped_profile.db_path


def test_baseline_profile():
    from instances.baseline.profile import PROFILE
    assert PROFILE.name == "baseline"
    assert PROFILE.use_l0_whitelist is True
    assert PROFILE.regime_filter_enabled is True
    assert PROFILE.position_size_multiplier == 1.0
    assert PROFILE.universe == []
    assert PROFILE.cell_overrides == []


def test_crypto_aggro_profile():
    from instances.crypto_aggro.profile import PROFILE
    assert PROFILE.name == "crypto_aggro"
    assert PROFILE.asset_categories == ["crypto"]
    assert PROFILE.position_size_multiplier == 1.5
    assert PROFILE.entry_threshold_multiplier == 0.8
    assert PROFILE.regime_filter_enabled is False
    # Crypto trades, stock does not.
    assert PROFILE.is_cell_allowed("BTC-USD", "ema_crossover", "crypto") is True
    assert PROFILE.is_cell_allowed("AAPL", "ema_crossover", "stocks") is False


def test_forex_focus_profile():
    from instances.forex_focus.profile import PROFILE
    assert PROFILE.name == "forex_focus"
    assert set(PROFILE.universe) == {"EURUSD=X", "GBPUSD=X"}
    assert set(PROFILE.strategies) == {
        "bb_squeeze", "momentum", "ema_crossover", "multi_factor",
    }
    assert PROFILE.position_size_multiplier == 2.0
    assert PROFILE.is_cell_allowed("EURUSD=X", "bb_squeeze", "forex") is True
    assert PROFILE.is_cell_allowed("EURUSD=X", "mean_reversion", "forex") is False
    assert PROFILE.is_cell_allowed("BTC-USD", "bb_squeeze", "crypto") is False


def test_top25_profile_has_25_cells():
    from instances.top25_only.profile import PROFILE, TOP25_CELLS
    assert PROFILE.name == "top25_only"
    assert len(PROFILE.cell_overrides) == 25
    assert len(TOP25_CELLS) == 25
    # Each entry is a (symbol, strategy) pair.
    for cell in PROFILE.cell_overrides:
        assert len(cell) == 2
        assert isinstance(cell[0], str) and isinstance(cell[1], str)
    # Top-1 cell from the report:
    assert ("TSLA", "bb_squeeze") in PROFILE.cell_overrides


def test_top25_profile_only_lets_those_cells_trade():
    from instances.top25_only.profile import PROFILE
    # Approved cell:
    assert PROFILE.is_cell_allowed("TSLA", "bb_squeeze", "stocks") is True
    # TSLA momentum is NOT in the top-25 list, must be rejected.
    assert PROFILE.is_cell_allowed("TSLA", "momentum", "stocks") is False
    # Random non-listed cell:
    assert PROFILE.is_cell_allowed("BTC-USD", "mean_reversion", "crypto") is False


def test_top25_profile_2x_size_no_regime():
    from instances.top25_only.profile import PROFILE
    assert PROFILE.position_size_multiplier == 2.0
    assert PROFILE.regime_filter_enabled is False


def test_all_shipped_profiles_use_whitelist_by_default():
    for name in ["baseline", "crypto_aggro", "forex_focus", "top25_only"]:
        mod = importlib.import_module(f"instances.{name}.profile")
        assert mod.PROFILE.use_l0_whitelist is True
