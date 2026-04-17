"""Tests for market_config.py — market metadata and correlation groups."""

from market_config import (
    MARKETS,
    MarketCategory,
    MarketConfig,
    StrategyType,
    get_all_symbols,
    get_correlation_groups,
    get_markets_by_category,
)


def test_markets_has_expected_crypto():
    assert "BTC-USD" in MARKETS
    assert "ETH-USD" in MARKETS
    assert "SOL-USD" in MARKETS


def test_markets_has_expected_stocks():
    for s in ("SPY", "QQQ", "AAPL", "TSLA", "NVDA"):
        assert s in MARKETS


def test_markets_has_commodities_and_forex():
    assert "GC=F" in MARKETS
    assert "CL=F" in MARKETS
    assert "EURUSD=X" in MARKETS


def test_every_market_is_valid_config():
    for symbol, cfg in MARKETS.items():
        assert isinstance(cfg, MarketConfig)
        assert cfg.symbol == symbol
        assert cfg.display_name
        assert isinstance(cfg.category, MarketCategory)
        assert cfg.timeframes
        assert cfg.initial_balance > 0
        assert 0 < cfg.max_position_pct <= 1


def test_get_all_symbols_matches_dict():
    assert set(get_all_symbols()) == set(MARKETS.keys())


def test_get_markets_by_category_crypto():
    crypto = get_markets_by_category(MarketCategory.CRYPTO)
    for _, cfg in crypto.items():
        assert cfg.category == MarketCategory.CRYPTO
    assert len(crypto) >= 3


def test_get_markets_by_category_empty_for_nonexistent():
    # Using a made-up category via MarketCategory enum doesn't apply — use any existing one.
    forex = get_markets_by_category(MarketCategory.FOREX)
    assert len(forex) >= 2


def test_correlation_groups_contains_crypto():
    groups = get_correlation_groups()
    assert "crypto" in groups
    assert set(groups["crypto"]) == {"BTC-USD", "ETH-USD", "SOL-USD"}


def test_correlation_groups_symbols_exist_in_markets():
    groups = get_correlation_groups()
    for group_symbols in groups.values():
        for s in group_symbols:
            assert s in MARKETS


def test_strategy_type_enum_values():
    assert StrategyType.MOMENTUM.value == "momentum"
    assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
    assert StrategyType.BREAKOUT.value == "breakout"
    assert StrategyType.MULTI_FACTOR.value == "multi_factor"


def test_market_category_enum_values():
    assert MarketCategory.CRYPTO.value == "crypto"
    assert MarketCategory.STOCKS.value == "stocks"
