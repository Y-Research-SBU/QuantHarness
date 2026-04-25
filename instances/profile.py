"""
Tournament profile schema.

A :class:`Profile` describes one paper-trading instance: which symbols and
strategies it can trade, sizing/threshold multipliers, whether the L0
backtest whitelist and L1 evolution filter apply, and where to write its
SQLite DB.

Profiles are simple dataclasses imported from each ``instances/<name>/profile.py``
module under the variable name ``PROFILE``. ``run_instance.py`` resolves
``--profile <name>`` by importing that module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Profile:
    """One tournament instance configuration.

    Filters are applied as a chain — an empty list means "don't filter on
    this dimension". ``cell_overrides``, when non-empty, restricts trading
    to exactly those (symbol, strategy) pairs and overrides everything
    else (universe, asset_categories, strategies).
    """

    name: str
    universe: List[str] = field(default_factory=list)              # symbol filter; [] = all
    asset_categories: List[str] = field(default_factory=list)      # e.g. ["crypto"]; [] = all
    strategies: List[str] = field(default_factory=list)            # strategy filter; [] = all
    cell_overrides: List[Tuple[str, str]] = field(default_factory=list)
    position_size_multiplier: float = 1.0
    entry_threshold_multiplier: float = 1.0  # <1.0 = looser, >1.0 = stricter
    regime_filter_enabled: bool = True
    use_l0_whitelist: bool = True
    use_l1_evolution: bool = True
    db_path: str = "paper_trades.db"
    initial_balance_per_market: float = 10000.0

    def is_cell_allowed(self, symbol: str, strategy: str, category: str = "") -> bool:
        """Return True if a (symbol, strategy) cell may trade under this profile.

        Order:
          1. If ``cell_overrides`` is set, only those exact pairs trade.
          2. Otherwise apply universe/asset_categories/strategies filters.
        """
        if self.cell_overrides:
            return (symbol, strategy) in set(self.cell_overrides)

        if self.universe and symbol not in self.universe:
            return False

        if self.asset_categories and category and category not in self.asset_categories:
            return False

        if self.strategies and strategy not in self.strategies:
            return False

        return True

    def filter_symbols(self, all_symbols: List[str]) -> List[str]:
        """Return the subset of ``all_symbols`` this profile can trade."""
        if self.cell_overrides:
            return sorted({sym for sym, _ in self.cell_overrides})
        if not self.universe:
            return list(all_symbols)
        return [s for s in all_symbols if s in self.universe]


__all__ = ["Profile"]
