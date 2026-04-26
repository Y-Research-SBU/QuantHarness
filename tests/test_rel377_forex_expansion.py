"""
Tests for REL-377: Forex universe expansion.

Covers:
  - All 6 new forex symbols are present in MARKETS with FOREX category
  - Existing forex symbols (EURUSD=X, GBPUSD=X) are not regressed
  - Whitelist additions are valid cells with correct schema
  - Loaders/parsers (get_markets_by_category, get_correlation_groups,
    get_all_symbols) handle the new entries
  - Backtest result schema is well-formed for the saved REL-377 results
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

from market_config import (
    MARKETS,
    MarketCategory,
    MarketConfig,
    StrategyType,
    get_all_symbols,
    get_correlation_groups,
    get_markets_by_category,
)
from strategy_whitelist import (
    BLACKLIST,
    WHITELIST,
    filter_strategies,
    is_allowed,
    whitelist_summary,
)


REL377_NEW_SYMBOLS = (
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "EURJPY=X",
    "GBPJPY=X",
)

# Cells that should have been added to the whitelist by REL-377.
# (symbol, strategy, sharpe, trades) — sharpe/trades from the saved backtest.
REL377_NEW_WHITELIST_CELLS = (
    ("AUDUSD=X", "ema_crossover", 0.58, 29),
    ("USDCAD=X", "vwap_reversion", 0.72, 15),
    ("USDCHF=X", "mean_reversion", 0.66, 44),
    ("EURJPY=X", "mean_reversion", 0.55, 50),
    ("GBPJPY=X", "mean_reversion", 0.52, 46),
    ("GBPJPY=X", "vwap_reversion", 0.44, 24),
    ("GBPJPY=X", "ema_crossover", 0.37, 30),
)


# ════════════════════════════════════════════════════════════════
# 1. All 6 new symbols are present in MARKETS with FOREX category
# ════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("symbol", REL377_NEW_SYMBOLS)
def test_new_forex_symbol_in_markets(symbol):
    assert symbol in MARKETS, f"{symbol} missing from MARKETS"


@pytest.mark.parametrize("symbol", REL377_NEW_SYMBOLS)
def test_new_forex_symbol_has_forex_category(symbol):
    assert MARKETS[symbol].category == MarketCategory.FOREX


@pytest.mark.parametrize("symbol", REL377_NEW_SYMBOLS)
def test_new_forex_symbol_metadata_matches_existing(symbol):
    """New symbols should mirror EURUSD=X / GBPUSD=X metadata exactly."""
    cfg = MARKETS[symbol]
    eur = MARKETS["EURUSD=X"]
    assert cfg.timeframes == eur.timeframes
    assert cfg.scan_interval_hours == eur.scan_interval_hours
    assert cfg.correlation_group == eur.correlation_group == "forex"
    assert cfg.symbol == symbol
    # Display name should be a non-empty human-readable string.
    assert cfg.display_name and "/" in cfg.display_name


def test_forex_universe_now_has_eight_symbols():
    forex = get_markets_by_category(MarketCategory.FOREX)
    assert len(forex) == 8, f"Expected 8 forex symbols, got {len(forex)}: {sorted(forex)}"
    expected = {"EURUSD=X", "GBPUSD=X"} | set(REL377_NEW_SYMBOLS)
    assert set(forex.keys()) == expected


# ════════════════════════════════════════════════════════════════
# 2. No regressions to existing forex symbols
# ════════════════════════════════════════════════════════════════


def test_eurusd_still_present_and_unchanged():
    cfg = MARKETS["EURUSD=X"]
    assert cfg.category == MarketCategory.FOREX
    assert cfg.display_name == "EUR/USD"
    assert cfg.correlation_group == "forex"
    assert cfg.timeframes == ["4h"]


def test_gbpusd_still_present_and_unchanged():
    cfg = MARKETS["GBPUSD=X"]
    assert cfg.category == MarketCategory.FOREX
    assert cfg.display_name == "GBP/USD"
    assert cfg.correlation_group == "forex"
    assert cfg.timeframes == ["4h"]


def test_existing_eurusd_whitelist_preserved():
    assert "EURUSD=X" in WHITELIST
    assert WHITELIST["EURUSD=X"] == frozenset({"mean_reversion", "ema_crossover"})


def test_existing_gbpusd_whitelist_preserved():
    assert "GBPUSD=X" in WHITELIST
    assert WHITELIST["GBPUSD=X"] == frozenset({"bb_squeeze", "momentum", "multi_factor"})


# ════════════════════════════════════════════════════════════════
# 3. Whitelist additions: schema + correctness
# ════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("symbol,strategy,sharpe,trades", REL377_NEW_WHITELIST_CELLS)
def test_new_whitelist_cell_present(symbol, strategy, sharpe, trades):
    assert symbol in WHITELIST, f"{symbol} not in WHITELIST"
    assert strategy in WHITELIST[symbol], (
        f"{symbol}/{strategy} (sharpe {sharpe:+.2f}, {trades} trades) "
        f"missing from WHITELIST. Got: {WHITELIST[symbol]}"
    )


@pytest.mark.parametrize("symbol,strategy,sharpe,trades", REL377_NEW_WHITELIST_CELLS)
def test_new_whitelist_cell_passes_filter_threshold(symbol, strategy, sharpe, trades):
    """Every admitted cell must clear Sharpe>=0.30 AND trades>=10."""
    assert sharpe >= 0.30, f"{symbol}/{strategy} sharpe {sharpe} below 0.30"
    assert trades >= 10, f"{symbol}/{strategy} trades {trades} below 10"


@pytest.mark.parametrize("symbol,strategy,_sharpe,_trades", REL377_NEW_WHITELIST_CELLS)
def test_new_whitelist_cell_is_allowed_via_api(symbol, strategy, _sharpe, _trades):
    """is_allowed() is the real public surface; verify it agrees with WHITELIST."""
    assert is_allowed(symbol, strategy)


def test_new_whitelist_strategies_are_real_strategy_types():
    """Every strategy on a REL-377 cell must correspond to a real StrategyType."""
    valid = {s.value for s in StrategyType}
    for symbol, strategy, _s, _t in REL377_NEW_WHITELIST_CELLS:
        assert strategy in valid, f"{strategy} not a valid StrategyType"


def test_whitelist_values_are_frozensets_of_strings():
    """Schema invariant: WHITELIST[symbol] is a frozenset of strategy names."""
    for symbol, strategies in WHITELIST.items():
        assert isinstance(strategies, frozenset), f"{symbol} value not frozenset"
        for strat in strategies:
            assert isinstance(strat, str), f"{symbol} has non-str strategy {strat!r}"


def test_usdjpy_correctly_omitted_from_whitelist():
    """USDJPY=X best cell was multi_factor +0.29 (78 trades) — must NOT be admitted."""
    assert "USDJPY=X" not in WHITELIST, (
        "USDJPY=X did not clear the Sharpe>=0.30 threshold and should not be in the whitelist."
    )


def test_no_new_blacklist_pairs_for_forex_expansion():
    """REL-377 added cells, not blacklist entries. Existing blacklist is untouched."""
    for symbol in REL377_NEW_SYMBOLS:
        for strat in ("momentum", "mean_reversion", "breakout", "multi_factor",
                      "vwap_reversion", "bb_squeeze", "ema_crossover"):
            # We didn't add any forex blacklist entries in this change.
            # (We add cells via WHITELIST; failures stay implicit via closed-list policy.)
            assert (symbol, strat) not in BLACKLIST


def test_filter_strategies_works_for_new_forex_symbol():
    """Public API: filter_strategies() should let high-Sharpe cells through."""
    candidates = ["momentum", "mean_reversion", "ema_crossover", "vwap_reversion",
                  "bb_squeeze", "multi_factor", "breakout"]
    allowed = filter_strategies("GBPJPY=X", candidates, category="forex")
    assert set(allowed) == {"mean_reversion", "vwap_reversion", "ema_crossover"}


def test_filter_strategies_blocks_unwhitelisted_for_new_forex_symbol():
    """USDJPY=X has no cells; closed-list policy means everything is filtered out."""
    candidates = ["momentum", "mean_reversion", "ema_crossover", "vwap_reversion",
                  "bb_squeeze", "multi_factor", "breakout"]
    allowed = filter_strategies("USDJPY=X", candidates, category="forex")
    assert allowed == []


def test_whitelist_summary_reflects_growth():
    summary = whitelist_summary()
    # We grew from 52→58 symbols and 99→109 cells (per docstring).
    assert summary["symbols"] >= 58
    assert summary["cells"] >= 109


# ════════════════════════════════════════════════════════════════
# 4. Loaders / parsers handle new entries
# ════════════════════════════════════════════════════════════════


def test_get_all_symbols_includes_all_new_forex():
    all_syms = set(get_all_symbols())
    for s in REL377_NEW_SYMBOLS:
        assert s in all_syms


def test_get_markets_by_category_returns_new_forex_pairs():
    forex = get_markets_by_category(MarketCategory.FOREX)
    for s in REL377_NEW_SYMBOLS:
        assert s in forex
        assert isinstance(forex[s], MarketConfig)


def test_correlation_group_forex_contains_new_pairs():
    groups = get_correlation_groups()
    assert "forex" in groups
    forex_group = set(groups["forex"])
    for s in REL377_NEW_SYMBOLS:
        assert s in forex_group
    # And the originals are still there.
    assert {"EURUSD=X", "GBPUSD=X"} <= forex_group


def test_each_new_forex_symbol_is_valid_market_config():
    for s in REL377_NEW_SYMBOLS:
        cfg = MARKETS[s]
        assert isinstance(cfg, MarketConfig)
        assert cfg.symbol == s
        assert cfg.display_name
        assert isinstance(cfg.category, MarketCategory)
        assert cfg.timeframes
        assert cfg.initial_balance > 0
        assert 0 < cfg.max_position_pct <= 1


# ════════════════════════════════════════════════════════════════
# 5. Backtest result schema validation (the saved REL-377 run)
# ════════════════════════════════════════════════════════════════


REPO_ROOT = Path(__file__).resolve().parent.parent
REL377_BACKTEST_GLOB = str(REPO_ROOT / "backtest_results" / "backtest_REL377-forex_*.json")


def _latest_rel377_backtest_path():
    matches = sorted(glob.glob(REL377_BACKTEST_GLOB))
    return matches[-1] if matches else None


@pytest.fixture(scope="module")
def rel377_backtest_payload():
    path = _latest_rel377_backtest_path()
    if path is None:
        pytest.skip("REL-377 backtest results JSON not present")
    with open(path) as fh:
        return json.load(fh)


def test_rel377_backtest_payload_top_level_schema(rel377_backtest_payload):
    payload = rel377_backtest_payload
    assert "generated_at" in payload
    assert "args" in payload
    assert "results" in payload
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) > 0


def test_rel377_backtest_each_result_has_required_fields(rel377_backtest_payload):
    required = {
        "symbol", "strategy", "timeframe", "starting_capital", "ending_capital",
        "total_return_pct", "total_trades", "winning_trades", "losing_trades",
        "win_rate", "sharpe_ratio", "max_drawdown_pct", "profit_factor",
    }
    for r in rel377_backtest_payload["results"]:
        missing = required - r.keys()
        assert not missing, f"Backtest result missing fields {missing}: {r}"
        # Type sanity
        assert isinstance(r["total_trades"], int)
        assert isinstance(r["sharpe_ratio"], (int, float))
        assert 0.0 <= r["win_rate"] <= 1.0


def test_rel377_backtest_covers_all_six_new_symbols(rel377_backtest_payload):
    symbols_in_payload = {r["symbol"] for r in rel377_backtest_payload["results"]}
    for s in REL377_NEW_SYMBOLS:
        assert s in symbols_in_payload, (
            f"{s} not present in REL-377 backtest results — "
            f"did the harness skip it? Got {sorted(symbols_in_payload)}"
        )


def test_rel377_backtest_admitted_cells_match_recorded_metrics(rel377_backtest_payload):
    """Every cell we added to WHITELIST must exist in the saved backtest with
    a Sharpe ≥ 0.30 AND trades ≥ 10. This guards against drift between
    the docs/whitelist and the JSON of record."""
    by_pair = {(r["symbol"], r["strategy"]): r for r in rel377_backtest_payload["results"]}
    for symbol, strategy, expected_sharpe, expected_trades in REL377_NEW_WHITELIST_CELLS:
        rec = by_pair.get((symbol, strategy))
        assert rec is not None, f"{symbol}/{strategy} missing from backtest payload"
        assert rec["sharpe_ratio"] >= 0.30, (
            f"{symbol}/{strategy} sharpe {rec['sharpe_ratio']} < 0.30"
        )
        assert rec["total_trades"] >= 10
        # Tolerate small floating-point rounding (we round to 2dp in docs).
        assert abs(rec["sharpe_ratio"] - expected_sharpe) < 0.05, (
            f"{symbol}/{strategy} sharpe drift: doc says {expected_sharpe}, "
            f"backtest says {rec['sharpe_ratio']}"
        )
        assert rec["total_trades"] == expected_trades, (
            f"{symbol}/{strategy} trade count drift: doc says {expected_trades}, "
            f"backtest says {rec['total_trades']}"
        )


def test_rel377_backtest_doc_exists():
    doc = REPO_ROOT / "docs" / "backtests" / "REL-377-forex-expansion.md"
    assert doc.exists(), f"Expected REL-377 backtest doc at {doc}"
    contents = doc.read_text()
    # Spot-check some expected substrings.
    for s in REL377_NEW_SYMBOLS:
        assert s in contents, f"{s} missing from REL-377 doc"
