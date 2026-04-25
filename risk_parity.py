"""
Risk parity / volatility-scaled position sizing.

Implements Markowitz-inspired portfolio weighting where each asset's weight is
determined by its volatility (and optionally its correlations with other assets)
rather than treating all assets equally. High-volatility assets like FLOKI
receive smaller dollar allocations than low-volatility assets like MSFT so that
each asset contributes a roughly equal amount of risk to the portfolio.

References:
    151 Trading Strategies §3.18 (Markowitz / Risk Parity).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Minimum number of assets required for a meaningful covariance computation.
MIN_ASSETS_FOR_COVARIANCE = 5

# Regularization added to covariance diagonal to avoid singular matrices.
COVARIANCE_REGULARIZATION = 1e-6


def compute_rolling_covariance(
    returns_df: pd.DataFrame,
    window: int = 60,
) -> np.ndarray:
    """Compute the covariance matrix from the trailing `window` daily returns.

    Args:
        returns_df: DataFrame of returns, columns = symbols, rows = time.
        window: Number of trailing rows to use for the covariance estimate.

    Returns:
        (N, N) covariance matrix as a numpy array, or an empty (0, 0) array if
        there is insufficient data.
    """
    if returns_df is None or returns_df.empty:
        return np.zeros((0, 0))

    n_assets = returns_df.shape[1]
    if n_assets == 0:
        return np.zeros((0, 0))

    trailing = returns_df.tail(window)
    # Need at least 2 rows to compute a covariance, and meaningfully more for
    # the estimate to be usable. We require at least min(window, half-window)
    # observations after dropping NaNs.
    cleaned = trailing.dropna(how="any")
    if len(cleaned) < 2:
        return np.zeros((0, 0))

    cov = cleaned.cov().to_numpy()
    # Add a tiny ridge to ensure positive-definiteness.
    cov = cov + COVARIANCE_REGULARIZATION * np.eye(cov.shape[0])
    return cov


def compute_inverse_vol_weights(volatilities: np.ndarray) -> np.ndarray:
    """Simple inverse-volatility weights, normalized to sum to 1.

    Args:
        volatilities: 1-D array of per-asset volatilities (e.g. stddev of
            returns). Must all be > 0.

    Returns:
        1-D array of weights summing to 1. Falls back to equal-weight if all
        inputs are zero or non-finite.
    """
    vols = np.asarray(volatilities, dtype=float).flatten()
    if vols.size == 0:
        return np.zeros(0)

    # Replace NaN/inf and non-positive entries with a sentinel so they don't
    # blow up the reciprocal.
    safe = np.where(np.isfinite(vols) & (vols > 0), vols, np.nan)

    if np.all(np.isnan(safe)):
        # Equal-weight fallback.
        return np.full(vols.size, 1.0 / vols.size)

    inv = np.where(np.isnan(safe), 0.0, 1.0 / safe)
    total = inv.sum()
    if total <= 0:
        return np.full(vols.size, 1.0 / vols.size)
    return inv / total


def compute_risk_parity_weights(
    cov_matrix: np.ndarray,
    expected_returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute risk-parity (or Markowitz-optimal) portfolio weights.

    If `expected_returns` is provided, uses Markowitz optimization:
        w_i = gamma * sum_j (C^-1_ij * E_j)
    The output is normalized so that sum(|w_i|) = 1.

    If `expected_returns` is None, uses pure inverse-volatility weighting,
    where each asset's weight is proportional to 1 / sqrt(C_ii).

    Args:
        cov_matrix: (N, N) covariance matrix.
        expected_returns: Optional (N,) vector of expected returns.

    Returns:
        (N,) numpy array of weights, normalized.
    """
    if cov_matrix is None or cov_matrix.size == 0:
        return np.zeros(0)

    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix must be square; got shape {cov.shape}")

    n = cov.shape[0]

    if expected_returns is None:
        # Pure inverse-volatility weighting from the diagonal.
        diag = np.diag(cov)
        # Variance can be tiny but not negative for a valid covariance; guard
        # anyway to keep sqrt finite.
        diag_safe = np.where(diag > 0, diag, COVARIANCE_REGULARIZATION)
        vols = np.sqrt(diag_safe)
        return compute_inverse_vol_weights(vols)

    er = np.asarray(expected_returns, dtype=float).flatten()
    if er.size != n:
        raise ValueError(
            f"expected_returns length {er.size} does not match cov dim {n}"
        )

    # Add ridge regularization so we can always invert.
    regularized = cov + COVARIANCE_REGULARIZATION * np.eye(n)
    try:
        inv_cov = np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        # Singular; fall back to pseudo-inverse.
        logger.warning("Covariance matrix singular; using pseudo-inverse")
        inv_cov = np.linalg.pinv(regularized)

    raw_weights = inv_cov @ er
    abs_sum = np.sum(np.abs(raw_weights))
    if abs_sum <= 0:
        # Degenerate case; fall back to inverse-vol on the diagonal.
        return compute_risk_parity_weights(cov, expected_returns=None)
    return raw_weights / abs_sum


def get_position_size_for_symbol(
    symbol: str,
    signal_strength: float,
    portfolio_capital: float,
    weights: Dict[str, float],
    max_position_pct: float = 0.10,
) -> float:
    """Translate a risk-parity weight into a dollar position size for a symbol.

    Args:
        symbol: Symbol to size.
        signal_strength: Strategy signal strength in [0, 1].
        portfolio_capital: Total portfolio capital in USD.
        weights: Mapping of symbol -> risk-parity weight (already normalized).
        max_position_pct: Hard cap on a single position as a fraction of
            capital. Always enforced regardless of weight.

    Returns:
        Dollar size for the position. Zero if the symbol has no weight.
    """
    if portfolio_capital <= 0:
        return 0.0
    if symbol not in weights:
        return 0.0

    weight = abs(float(weights[symbol]))
    strength = max(0.0, min(1.0, float(signal_strength)))

    raw_size = portfolio_capital * weight * strength
    cap = portfolio_capital * max_position_pct
    return float(min(raw_size, cap))


# ─────────────────────────── Manager ───────────────────────────


@dataclass
class _SymbolPriceHistory:
    """Per-symbol bounded price history used to derive returns."""
    prices: Deque[Tuple[datetime, float]] = field(default_factory=lambda: deque(maxlen=512))

    def add(self, ts: datetime, price: float) -> None:
        self.prices.append((ts, price))

    def to_series(self) -> pd.Series:
        if not self.prices:
            return pd.Series(dtype=float)
        idx = [t for t, _ in self.prices]
        vals = [p for _, p in self.prices]
        return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


class RiskParityManager:
    """Maintains rolling price history and computes risk-parity weights.

    Usage:
        manager = RiskParityManager(lookback_days=60)
        manager.update_prices("BTC-USD", 65000.0, datetime.utcnow())
        ...
        size = manager.get_position_size("BTC-USD", 0.8, capital=100_000)
    """

    def __init__(
        self,
        lookback_days: int = 60,
        rebalance_hours: int = 24,
        max_position_pct: float = 0.10,
        min_assets: int = MIN_ASSETS_FOR_COVARIANCE,
        dollar_neutral: bool = False,
    ):
        """Initialize the manager.

        Args:
            lookback_days: Trailing window of daily returns used for covariance.
            rebalance_hours: Minimum hours between weight recomputations.
            max_position_pct: Hard cap on any single position.
            min_assets: Minimum tracked symbols required before computing
                covariance-based weights. Below this, equal-weight is used.
            dollar_neutral: If True, recenter weights so they sum to 0
                (long/short balanced) while keeping sum(|w|) = 1.
        """
        self.lookback_days = lookback_days
        self.rebalance_hours = rebalance_hours
        self.max_position_pct = max_position_pct
        self.min_assets = min_assets
        self.dollar_neutral = dollar_neutral

        self._lock = threading.RLock()
        self._history: Dict[str, _SymbolPriceHistory] = {}
        self._weights: Dict[str, float] = {}
        self._last_rebalance: Optional[datetime] = None

    def update_prices(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Add a price observation for a symbol.

        Args:
            symbol: The symbol whose price is being recorded.
            price: The latest price.
            timestamp: When the observation occurred.
        """
        if price is None or not np.isfinite(price) or price <= 0:
            return
        with self._lock:
            hist = self._history.setdefault(symbol, _SymbolPriceHistory())
            hist.add(timestamp, float(price))

    def _build_returns_frame(self) -> pd.DataFrame:
        """Build a daily-returns DataFrame from accumulated price history."""
        if not self._history:
            return pd.DataFrame()

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days * 2)
        series_by_symbol: Dict[str, pd.Series] = {}
        for symbol, hist in self._history.items():
            s = hist.to_series()
            if s.empty:
                continue
            # Tz-normalize index to UTC for consistent resampling.
            if s.index.tz is None:
                s.index = s.index.tz_localize("UTC")
            else:
                s.index = s.index.tz_convert("UTC")
            s = s[s.index >= cutoff]
            if s.empty:
                continue
            # Resample to daily close (last observation of the day).
            daily = s.resample("1D").last().dropna()
            if len(daily) < 2:
                continue
            series_by_symbol[symbol] = daily

        if not series_by_symbol:
            return pd.DataFrame()

        prices_df = pd.DataFrame(series_by_symbol).sort_index()
        returns = prices_df.pct_change(fill_method=None).dropna(how="all")
        return returns

    def _compute_weights(self) -> Dict[str, float]:
        """Compute risk-parity weights from current price history."""
        returns = self._build_returns_frame()
        if returns.empty or returns.shape[1] < self.min_assets:
            # Equal-weight fallback across whatever symbols we do have.
            symbols = list(self._history.keys())
            if not symbols:
                return {}
            equal_w = 1.0 / len(symbols)
            return {s: equal_w for s in symbols}

        cov = compute_rolling_covariance(returns, window=self.lookback_days)
        if cov.size == 0:
            symbols = list(returns.columns)
            equal_w = 1.0 / len(symbols)
            return {s: equal_w for s in symbols}

        weights_arr = compute_risk_parity_weights(cov)
        symbols = list(returns.columns)
        weights = {s: float(w) for s, w in zip(symbols, weights_arr)}

        if self.dollar_neutral and weights:
            mean_w = np.mean(list(weights.values()))
            shifted = {s: w - mean_w for s, w in weights.items()}
            abs_sum = sum(abs(w) for w in shifted.values())
            if abs_sum > 0:
                weights = {s: w / abs_sum for s, w in shifted.items()}

        return weights

    def _needs_rebalance(self) -> bool:
        """Return True if enough time has elapsed to recompute weights."""
        if self._last_rebalance is None:
            return True
        elapsed = datetime.now(timezone.utc) - self._last_rebalance
        return elapsed >= timedelta(hours=self.rebalance_hours)

    def get_weights(self, force_recompute: bool = False) -> Dict[str, float]:
        """Return the current weight map, recomputing if rebalance is due.

        Args:
            force_recompute: Bypass the rebalance interval and recompute now.
        """
        with self._lock:
            if force_recompute or self._needs_rebalance() or not self._weights:
                self._weights = self._compute_weights()
                self._last_rebalance = datetime.now(timezone.utc)
            return dict(self._weights)

    def get_position_size(
        self,
        symbol: str,
        signal_strength: float,
        capital: float,
    ) -> float:
        """Get the dollar position size for `symbol` under current weights.

        Args:
            symbol: Symbol to size.
            signal_strength: Strategy signal strength in [0, 1].
            capital: Total portfolio capital in USD.

        Returns:
            Dollar position size, capped at max_position_pct of capital.
        """
        weights = self.get_weights()
        return get_position_size_for_symbol(
            symbol=symbol,
            signal_strength=signal_strength,
            portfolio_capital=capital,
            weights=weights,
            max_position_pct=self.max_position_pct,
        )
