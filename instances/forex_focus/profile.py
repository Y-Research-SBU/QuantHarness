"""Forex-focus profile.

Just EURUSD=X and GBPUSD=X with 4 strategies. Tests whether a tight
forex-only book outperforms the diversified baseline at 2× sizing.
The asset class heatmap shows forex has the highest mean Sharpe across
several strategies.
"""

from instances.profile import Profile

PROFILE = Profile(
    name="forex_focus",
    universe=["EURUSD=X", "GBPUSD=X"],
    asset_categories=["forex"],
    strategies=[
        "bb_squeeze",
        "momentum",
        "ema_crossover",
        "multi_factor",
    ],
    cell_overrides=[],
    position_size_multiplier=2.0,
    entry_threshold_multiplier=1.0,
    regime_filter_enabled=True,
    use_l0_whitelist=True,
    use_l1_evolution=True,
    db_path="paper_trades_forex_focus.db",
    initial_balance_per_market=10000.0,
)
