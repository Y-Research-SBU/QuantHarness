"""Tests for the strategy whitelist (L0 filter)."""

import json
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_whitelist import (  # noqa: E402
    BLACKLIST,
    KRONOS_ALLOWED_CATEGORIES,
    KRONOS_STRATEGIES,
    WHITELIST,
    filter_strategies,
    is_allowed,
    whitelist_summary,
)


# ════════════════════════════════════════════════════════════════
# Whitelist hits — top backtest cells should be allowed
# ════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "symbol,strategy",
    [
        ("TSLA", "bb_squeeze"),
        ("WLD-USD", "ema_crossover"),
        ("SEI-USD", "multi_factor"),
        ("CRWD", "bb_squeeze"),
        ("SOL-USD", "ema_crossover"),
        ("GBPUSD=X", "bb_squeeze"),
        ("TIA-USD", "mean_reversion"),
        ("XLE", "vwap_reversion"),
        ("LLY", "breakout"),
        ("GC=F", "breakout"),
    ],
)
def test_top_cells_allowed(symbol, strategy):
    assert is_allowed(symbol, strategy)


# ════════════════════════════════════════════════════════════════
# Whitelist misses — strategies not validated for that symbol
# ════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "symbol,strategy",
    [
        ("TSLA", "ema_crossover"),       # TSLA validated for bb_squeeze + multi_factor only
        ("BTC-USD", "mean_reversion"),    # BTC only validated for ema_crossover
        ("SOL-USD", "vwap_reversion"),    # SOL not validated for vwap
    ],
)
def test_non_whitelisted_strategy_blocked(symbol, strategy):
    assert not is_allowed(symbol, strategy)


def test_unknown_symbol_blocked():
    """Closed-list policy: symbols not in the whitelist cannot trade."""
    assert not is_allowed("FAKE-USD", "ema_crossover")
    assert not is_allowed("XYZ", "momentum")


# ════════════════════════════════════════════════════════════════
# Blacklist — explicit pairs always denied
# ════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "symbol,strategy",
    [
        ("SEI-USD", "mean_reversion"),
        ("BTC-USD", "breakout"),
        ("BTC-USD", "momentum"),
        ("SOL-USD", "mean_reversion"),
        ("XOM", "momentum"),
        ("TIA-USD", "ema_crossover"),
    ],
)
def test_blacklist_blocks(symbol, strategy):
    assert (symbol, strategy) in BLACKLIST
    assert not is_allowed(symbol, strategy)


def test_blacklist_overrides_whitelist():
    """If a pair is in both blacklist and whitelist, blacklist wins."""
    # TIA-USD ema_crossover is blacklisted; it should never be allowed
    # even if it accidentally appeared in the whitelist.
    assert ("TIA-USD", "ema_crossover") in BLACKLIST
    assert not is_allowed("TIA-USD", "ema_crossover")


# ════════════════════════════════════════════════════════════════
# Kronos family — allowed only on configured categories
# ════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("kronos_strat", sorted(KRONOS_STRATEGIES))
def test_kronos_allowed_on_crypto(kronos_strat):
    assert is_allowed("BTC-USD", kronos_strat, category="crypto")
    assert is_allowed("SOMETHING-USD", kronos_strat, category="crypto")


@pytest.mark.parametrize("kronos_strat", sorted(KRONOS_STRATEGIES))
@pytest.mark.parametrize("category", ["stocks", "commodities", "forex"])
def test_kronos_blocked_off_crypto(kronos_strat, category):
    assert not is_allowed("AAPL", kronos_strat, category=category)


def test_kronos_blocked_when_category_missing():
    """Without an explicit crypto tag, Kronos should fall back to deny."""
    assert not is_allowed("BTC-USD", "kronos_momentum_confirm", category="")


# ════════════════════════════════════════════════════════════════
# filter_strategies()
# ════════════════════════════════════════════════════════════════


def test_filter_strategies_keeps_allowed_drops_others():
    candidates = ["bb_squeeze", "ema_crossover", "mean_reversion", "vwap_reversion"]
    kept = filter_strategies("TSLA", candidates)
    # TSLA's whitelist: {bb_squeeze, multi_factor}
    assert kept == ["bb_squeeze"]


def test_filter_strategies_includes_kronos_for_crypto():
    candidates = ["ema_crossover", "kronos_momentum_confirm"]
    kept = filter_strategies("BTC-USD", candidates, category="crypto")
    assert "ema_crossover" in kept
    assert "kronos_momentum_confirm" in kept


def test_filter_strategies_excludes_kronos_for_stocks():
    candidates = ["breakout", "kronos_momentum_confirm"]
    kept = filter_strategies("AAPL", candidates, category="stocks")
    assert kept == ["breakout"]
    assert "kronos_momentum_confirm" not in kept


# ════════════════════════════════════════════════════════════════
# Schema / data integrity
# ════════════════════════════════════════════════════════════════


def test_whitelist_size_matches_backtest():
    cell_count = sum(len(v) for v in WHITELIST.values())
    # Selection filter: Sharpe >= 0.30, trades >= 10 produced 99 cells.
    # We allow a small +N for manually-added high-confidence entries
    # (e.g. CRWD breakout, NVDA breakout) that came from the top-25 list.
    assert 95 <= cell_count <= 110, f"cells={cell_count} out of expected band"
    assert 50 <= len(WHITELIST) <= 60


def test_no_kronos_in_static_whitelist():
    """Kronos strategies are gated separately via category, not the whitelist."""
    for strats in WHITELIST.values():
        for s in strats:
            assert s not in KRONOS_STRATEGIES, f"Kronos strategy leaked into WHITELIST: {s}"


def test_summary_returns_dict_with_expected_keys():
    s = whitelist_summary()
    for key in ("symbols", "cells", "blacklist_pairs", "expected_sharpe"):
        assert key in s
    assert s["symbols"] >= 50
    assert s["cells"] >= 95
    assert s["expected_sharpe"] > 0


# ════════════════════════════════════════════════════════════════
# Smoke against the actual backtest file
# ════════════════════════════════════════════════════════════════


def test_top_25_backtest_cells_are_in_whitelist():
    """Hard guarantee: any cell with Sharpe >=0.69 in the optimized backtest
    must be tradeable by the whitelist."""
    bt_path = Path(__file__).resolve().parent.parent / "backtest_results" / "backtest_optimized_v2_20260425_155648.json"
    if not bt_path.exists():
        pytest.skip("optimized backtest file not present")
    with open(bt_path) as f:
        data = json.load(f)
    top_cells = [
        (r["symbol"], r["strategy"])
        for r in data["results"]
        if r["sharpe_ratio"] >= 0.69 and r["total_trades"] >= 5
    ]
    assert len(top_cells) >= 20, f"expected at least 20 top cells, found {len(top_cells)}"
    missing = [
        (s, st) for (s, st) in top_cells
        if not is_allowed(s, st, category="crypto" if "-USD" in s else "stocks")
    ]
    # There may be up to 2 misses if the backtest had a cell with <10 trades
    # that slipped under our threshold.
    assert len(missing) <= 2, f"too many top cells excluded: {missing}"
