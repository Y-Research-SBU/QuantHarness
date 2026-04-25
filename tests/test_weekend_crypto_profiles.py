"""Tests for the weekend-focused crypto profiles."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make the repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(name: str):
    return importlib.import_module(f"instances.{name}.profile").PROFILE


# ════════════════════════════════════════════════════════════════
# crypto_kronos_only
# ════════════════════════════════════════════════════════════════


def test_crypto_kronos_only_loads():
    p = _load("crypto_kronos_only")
    assert p.name == "crypto_kronos_only"


def test_crypto_kronos_only_is_crypto_only():
    p = _load("crypto_kronos_only")
    assert p.asset_categories == ["crypto"]


def test_crypto_kronos_only_has_only_kronos_strategies():
    p = _load("crypto_kronos_only")
    assert set(p.strategies) == {
        "kronos_momentum_confirm",
        "kronos_divergence",
        "multi_timeframe_kronos",
    }


def test_crypto_kronos_only_bypasses_whitelist():
    """Kronos strategies are not in the L0 whitelist, so the profile
    must bypass it or it would never trade."""
    p = _load("crypto_kronos_only")
    assert p.use_l0_whitelist is False


def test_crypto_kronos_only_keeps_l1_evolution():
    """We still want auto-disable on per-symbol Kronos failures."""
    p = _load("crypto_kronos_only")
    assert p.use_l1_evolution is True


def test_crypto_kronos_only_db_is_isolated():
    p = _load("crypto_kronos_only")
    assert p.db_path == "paper_trades_crypto_kronos_only.db"


def test_crypto_kronos_only_standard_sizing():
    p = _load("crypto_kronos_only")
    assert p.position_size_multiplier == 1.0
    assert p.entry_threshold_multiplier == 1.0


def test_crypto_kronos_only_regime_filter_on():
    p = _load("crypto_kronos_only")
    assert p.regime_filter_enabled is True


# ════════════════════════════════════════════════════════════════
# crypto_breakout_vol
# ════════════════════════════════════════════════════════════════


def test_crypto_breakout_vol_loads():
    p = _load("crypto_breakout_vol")
    assert p.name == "crypto_breakout_vol"


def test_crypto_breakout_vol_is_crypto_only():
    p = _load("crypto_breakout_vol")
    assert p.asset_categories == ["crypto"]


def test_crypto_breakout_vol_strategies_are_volatility_based():
    p = _load("crypto_breakout_vol")
    assert set(p.strategies) == {"bb_squeeze", "breakout", "ema_crossover"}


def test_crypto_breakout_vol_uses_whitelist():
    p = _load("crypto_breakout_vol")
    assert p.use_l0_whitelist is True


def test_crypto_breakout_vol_oversize_positions():
    p = _load("crypto_breakout_vol")
    assert p.position_size_multiplier == 1.5


def test_crypto_breakout_vol_standard_entry_threshold():
    """We pump size, not entry looseness — we want trend confirmation."""
    p = _load("crypto_breakout_vol")
    assert p.entry_threshold_multiplier == 1.0


def test_crypto_breakout_vol_db_is_isolated():
    p = _load("crypto_breakout_vol")
    assert p.db_path == "paper_trades_crypto_breakout_vol.db"


# ════════════════════════════════════════════════════════════════
# Cross-profile invariants
# ════════════════════════════════════════════════════════════════


def test_all_db_paths_are_unique():
    """Each instance must write to its own DB to keep results isolated."""
    names = [
        "baseline",
        "crypto_aggro",
        "forex_focus",
        "top25_only",
        "crypto_kronos_only",
        "crypto_breakout_vol",
    ]
    paths = [_load(n).db_path for n in names]
    assert len(paths) == len(set(paths)), f"duplicate db paths: {paths}"


def test_all_profile_names_match_their_directory():
    names = [
        "baseline",
        "crypto_aggro",
        "forex_focus",
        "top25_only",
        "crypto_kronos_only",
        "crypto_breakout_vol",
    ]
    for n in names:
        p = _load(n)
        assert p.name == n


def test_crypto_profiles_only_request_crypto_category():
    """Sanity: anything named crypto_* should be crypto-only."""
    crypto_profiles = ["crypto_aggro", "crypto_kronos_only", "crypto_breakout_vol"]
    for n in crypto_profiles:
        p = _load(n)
        assert p.asset_categories == ["crypto"], f"{n} asset_categories={p.asset_categories}"
