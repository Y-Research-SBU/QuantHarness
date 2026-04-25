"""Baseline profile — same behavior as the original whitelist runner.

All 85 markets, L0 whitelist enforced, regime gate on, no multipliers.
This is the control instance against which all other tournament entrants
are measured.
"""

from instances.profile import Profile

PROFILE = Profile(
    name="baseline",
    universe=[],                       # all markets
    asset_categories=[],               # all categories
    strategies=[],                     # all strategies (whitelist still gates)
    cell_overrides=[],
    position_size_multiplier=1.0,
    entry_threshold_multiplier=1.0,
    regime_filter_enabled=True,
    use_l0_whitelist=True,
    use_l1_evolution=True,
    db_path="paper_trades_baseline.db",
    initial_balance_per_market=10000.0,
)
