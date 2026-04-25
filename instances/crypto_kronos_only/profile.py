"""Crypto Kronos-only profile.

Pure crypto. Only the 3 Kronos strategies (kronos_momentum_confirm,
kronos_divergence, multi_timeframe_kronos) — bypasses L0 whitelist
since the whitelist deliberately excludes Kronos strategies (untested
on the full universe). This is our forward-test of Kronos in production
in isolation, so we can compare its real-time performance vs the other
strategies head-to-head.

Standard position size, regime filter ON (we trust the regime detector
to keep Kronos out of bad setups), L1 evolution ON so failing Kronos
strategies get auto-disabled per-symbol.
"""

from instances.profile import Profile

PROFILE = Profile(
    name="crypto_kronos_only",
    universe=[],                       # all crypto in MARKETS
    asset_categories=["crypto"],
    strategies=[
        "kronos_momentum_confirm",
        "kronos_divergence",
        "multi_timeframe_kronos",
    ],
    cell_overrides=[],
    position_size_multiplier=1.0,
    entry_threshold_multiplier=1.0,
    regime_filter_enabled=True,
    use_l0_whitelist=False,            # bypass whitelist (Kronos not in it)
    use_l1_evolution=True,
    db_path="paper_trades_crypto_kronos_only.db",
    initial_balance_per_market=10000.0,
)
