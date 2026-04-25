"""Crypto-aggressive profile.

Crypto-only universe, +50% position size, looser entry thresholds (-20%),
no regime filter. Whitelist still applies. Tests whether crypto markets
reward more aggressive sizing & looser entries.
"""

from instances.profile import Profile

PROFILE = Profile(
    name="crypto_aggro",
    universe=[],
    asset_categories=["crypto"],
    strategies=[],
    cell_overrides=[],
    position_size_multiplier=1.5,
    entry_threshold_multiplier=0.8,
    regime_filter_enabled=False,
    use_l0_whitelist=True,
    use_l1_evolution=True,
    db_path="paper_trades_crypto_aggro.db",
    initial_balance_per_market=10000.0,
)
