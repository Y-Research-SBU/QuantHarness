"""Top-25 profile.

Trades only the 25 (symbol, strategy) cells with Sharpe >= 0.69 from the
2026-04-25 backtest (see backtest_results/optimization_report.md). 2x
position sizing, no regime gate. This is the "concentrate firepower on
proven cells" hypothesis.
"""

from instances.profile import Profile

# Sourced from optimization_report.md "Top 25 (Symbol, Strategy) Cells"
# table, sorted by Sharpe (>= 0.69) with >=5 trades in the optimized run.
TOP25_CELLS = [
    ("TSLA", "bb_squeeze"),
    ("WLD-USD", "ema_crossover"),
    ("SEI-USD", "multi_factor"),
    ("CRWD", "bb_squeeze"),
    ("EIGEN-USD", "ema_crossover"),
    ("WLD-USD", "bb_squeeze"),
    ("WIF-USD", "ema_crossover"),
    ("XLE", "vwap_reversion"),
    ("MKR-USD", "ema_crossover"),
    ("SOL-USD", "ema_crossover"),
    ("TIA-USD", "mean_reversion"),
    ("GBPUSD=X", "bb_squeeze"),
    ("AVAX-USD", "vwap_reversion"),
    ("ENA-USD", "multi_factor"),
    ("TSLA", "multi_factor"),
    ("DIA", "ema_crossover"),
    ("NET", "vwap_reversion"),
    ("FET-USD", "bb_squeeze"),
    ("GC=F", "breakout"),
    ("CRV-USD", "ema_crossover"),
    ("AVGO", "breakout"),
    ("CRWD", "breakout"),
    ("GBPUSD=X", "momentum"),
    ("LLY", "breakout"),
    ("NVDA", "breakout"),
]

PROFILE = Profile(
    name="top25_only",
    universe=[],
    asset_categories=[],
    strategies=[],
    cell_overrides=list(TOP25_CELLS),
    position_size_multiplier=2.0,
    entry_threshold_multiplier=1.0,
    regime_filter_enabled=False,
    use_l0_whitelist=True,
    use_l1_evolution=True,
    db_path="paper_trades_top25_only.db",
    initial_balance_per_market=10000.0,
)
