"""Crypto breakout + volatility profile.

Crypto-only, but only volatility-driven strategies: bb_squeeze, breakout,
ema_crossover. These were the backtest's strongest performers on crypto
(ema_crossover +0.33, bb_squeeze +0.13). Position size +50%; regime
filter on (we want trend confirmation for breakout/EMA strategies).

Designed to perform best on weekends and after-hours when crypto sees
elevated volatility relative to equities (which are closed).
"""

from instances.profile import Profile

PROFILE = Profile(
    name="crypto_breakout_vol",
    universe=[],
    asset_categories=["crypto"],
    strategies=[
        "bb_squeeze",
        "breakout",
        "ema_crossover",
    ],
    cell_overrides=[],
    position_size_multiplier=1.5,
    entry_threshold_multiplier=1.0,    # standard entry; not loosened
    regime_filter_enabled=True,
    use_l0_whitelist=True,             # whitelist still gates per-symbol
    use_l1_evolution=True,
    db_path="paper_trades_crypto_breakout_vol.db",
    initial_balance_per_market=10000.0,
)
