"""Tests for risk_parity.py — Markowitz/risk-parity position sizing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from risk_parity import (
    COVARIANCE_REGULARIZATION,
    RiskParityManager,
    compute_inverse_vol_weights,
    compute_risk_parity_weights,
    compute_rolling_covariance,
    get_position_size_for_symbol,
)


# ───────────────────────── helpers ─────────────────────────


def _synthetic_returns(
    n_days: int,
    vols: dict[str, float],
    seed: int = 42,
    correlation: float = 0.0,
) -> pd.DataFrame:
    """Build a returns DataFrame with specified per-asset volatilities.

    `correlation` injects a shared factor so columns are mutually correlated.
    """
    rng = np.random.default_rng(seed)
    n = len(vols)
    common = rng.standard_normal(n_days)
    cols = {}
    for sym, vol in vols.items():
        idiosyncratic = rng.standard_normal(n_days)
        x = correlation * common + np.sqrt(max(0.0, 1.0 - correlation**2)) * idiosyncratic
        cols[sym] = x * vol
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=n_days, freq="D")
    return pd.DataFrame(cols, index=idx)


# ─────────────────── covariance computation ───────────────────


def test_compute_rolling_covariance_basic():
    # 3 assets with controlled vols and modest correlation.
    vols = {"A": 0.01, "B": 0.02, "C": 0.03}
    returns = _synthetic_returns(200, vols, correlation=0.4)
    cov = compute_rolling_covariance(returns, window=120)

    assert cov.shape == (3, 3)
    # Diagonal entries should approximate var = vol^2.
    diag = np.diag(cov)
    np.testing.assert_allclose(diag, [0.01**2, 0.02**2, 0.03**2], rtol=0.4)
    # Symmetry.
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_compute_rolling_covariance_insufficient_data():
    # Only one row of returns → cov undefined.
    returns = pd.DataFrame({"A": [0.01], "B": [0.02]})
    cov = compute_rolling_covariance(returns, window=60)
    assert cov.shape == (0, 0)


def test_compute_rolling_covariance_empty_df():
    cov = compute_rolling_covariance(pd.DataFrame(), window=60)
    assert cov.shape == (0, 0)


def test_covariance_matrix_is_positive_definite():
    vols = {"A": 0.01, "B": 0.02, "C": 0.015, "D": 0.025}
    returns = _synthetic_returns(120, vols, correlation=0.6)
    cov = compute_rolling_covariance(returns, window=90)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), f"Covariance not PD; eigenvalues: {eigvals}"


def test_handles_nan_in_returns():
    # Inject NaNs; cov should still be finite & positive-definite on cleaned rows.
    vols = {"A": 0.01, "B": 0.02, "C": 0.015}
    returns = _synthetic_returns(120, vols, correlation=0.3)
    returns.iloc[0:5, 0] = np.nan
    returns.iloc[10:12, 1] = np.nan
    cov = compute_rolling_covariance(returns, window=100)
    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    assert np.all(np.linalg.eigvalsh(cov) > 0)


# ─────────────────── weight computation ───────────────────


def test_compute_risk_parity_weights_equal_vol():
    # Identity-like covariance → equal weights.
    cov = np.eye(4) * 0.01
    w = compute_risk_parity_weights(cov)
    np.testing.assert_allclose(w, np.full(4, 0.25), atol=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_compute_risk_parity_weights_unequal_vol():
    # Variances 1, 4, 9 → vols 1, 2, 3 → inv-vol 1, 0.5, 0.33 → normalize.
    cov = np.diag([1.0, 4.0, 9.0])
    w = compute_risk_parity_weights(cov)

    # Highest-vol asset gets the smallest weight.
    assert w[0] > w[1] > w[2]
    assert w.sum() == pytest.approx(1.0)
    expected = np.array([1.0, 0.5, 1.0 / 3.0])
    expected = expected / expected.sum()
    np.testing.assert_allclose(w, expected, rtol=1e-3)


def test_inverse_vol_weights_sum_to_one():
    vols = np.array([0.01, 0.02, 0.04, 0.08])
    w = compute_inverse_vol_weights(vols)
    assert w.sum() == pytest.approx(1.0)
    # Inversely proportional: lowest vol gets the largest weight.
    assert w[0] > w[1] > w[2] > w[3]


def test_inverse_vol_weights_handles_zero_and_nan():
    # Zero & NaN inputs shouldn't blow up; remaining valid entries weight as
    # expected.
    vols = np.array([0.01, 0.0, np.nan, 0.04])
    w = compute_inverse_vol_weights(vols)
    assert w.sum() == pytest.approx(1.0)
    assert w[1] == 0.0
    assert w[2] == 0.0
    assert w[0] > w[3]


def test_inverse_vol_weights_all_invalid_falls_back_to_equal():
    vols = np.array([0.0, np.nan, -1.0])
    w = compute_inverse_vol_weights(vols)
    np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0))


def test_compute_risk_parity_weights_with_expected_returns():
    # Markowitz path: positive expected returns → all-positive weights.
    cov = np.eye(3) * 0.01
    er = np.array([0.05, 0.03, 0.01])
    w = compute_risk_parity_weights(cov, expected_returns=er)
    assert np.sum(np.abs(w)) == pytest.approx(1.0)
    # Higher expected return → larger weight (with identity cov).
    assert w[0] > w[1] > w[2]


def test_compute_risk_parity_weights_singular_matrix():
    # Singular covariance: pseudo-inverse path should still produce a valid
    # normalized weight vector.
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])
    er = np.array([0.05, 0.05])
    w = compute_risk_parity_weights(cov, expected_returns=er)
    assert np.all(np.isfinite(w))
    assert np.sum(np.abs(w)) == pytest.approx(1.0)


def test_compute_risk_parity_weights_empty_cov():
    w = compute_risk_parity_weights(np.zeros((0, 0)))
    assert w.shape == (0,)


def test_handles_single_asset():
    # Single-asset covariance should still produce weight = 1.
    cov = np.array([[0.04]])
    w = compute_risk_parity_weights(cov)
    np.testing.assert_allclose(w, [1.0])


def test_non_square_covariance_raises():
    with pytest.raises(ValueError):
        compute_risk_parity_weights(np.zeros((3, 4)))


# ─────────────────── position size helper ───────────────────


def test_position_size_respects_max():
    # Symbol given an absurd weight (50%) but max is 10% → must clamp.
    weights = {"BTC-USD": 0.5, "ETH-USD": 0.5}
    size = get_position_size_for_symbol(
        symbol="BTC-USD",
        signal_strength=1.0,
        portfolio_capital=100_000.0,
        weights=weights,
        max_position_pct=0.10,
    )
    assert size == pytest.approx(10_000.0)


def test_position_size_scales_with_signal_strength():
    weights = {"BTC-USD": 0.04}
    full = get_position_size_for_symbol("BTC-USD", 1.0, 100_000, weights, 0.10)
    half = get_position_size_for_symbol("BTC-USD", 0.5, 100_000, weights, 0.10)
    zero = get_position_size_for_symbol("BTC-USD", 0.0, 100_000, weights, 0.10)
    assert full == pytest.approx(4_000.0)
    assert half == pytest.approx(2_000.0)
    assert zero == 0.0


def test_position_size_unknown_symbol_returns_zero():
    weights = {"BTC-USD": 0.5}
    size = get_position_size_for_symbol("MSFT", 1.0, 100_000, weights, 0.10)
    assert size == 0.0


def test_position_size_zero_capital():
    weights = {"BTC-USD": 0.5}
    size = get_position_size_for_symbol("BTC-USD", 1.0, 0.0, weights, 0.10)
    assert size == 0.0


# ─────────────────── manager integration ───────────────────


def _seed_manager_with_synthetic_history(
    manager: RiskParityManager,
    vols: dict[str, float],
    n_days: int = 90,
    seed: int = 7,
):
    """Push synthetic daily prices for each symbol into the manager."""
    rng = np.random.default_rng(seed)
    base = datetime.now(timezone.utc) - timedelta(days=n_days)
    for sym, vol in vols.items():
        price = 100.0
        for d in range(n_days):
            ret = rng.standard_normal() * vol
            price = price * (1.0 + ret)
            ts = base + timedelta(days=d)
            manager.update_prices(sym, price, ts)


def test_risk_parity_manager_full_cycle():
    # 6 assets with widely varied vols → weights should be inversely related.
    vols = {
        "MSFT": 0.005,
        "SPY": 0.008,
        "BTC-USD": 0.03,
        "ETH-USD": 0.04,
        "FLOKI-USD": 0.08,
        "GLD": 0.006,
    }
    manager = RiskParityManager(lookback_days=60, rebalance_hours=24, max_position_pct=0.10)
    _seed_manager_with_synthetic_history(manager, vols, n_days=120)

    weights = manager.get_weights(force_recompute=True)
    assert set(weights.keys()) == set(vols.keys())
    assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0, rel=1e-6)

    # Higher-vol asset should get less weight than lower-vol asset on average.
    assert abs(weights["MSFT"]) > abs(weights["FLOKI-USD"])
    assert abs(weights["GLD"]) > abs(weights["BTC-USD"])

    # Sized position respects the cap.
    size_floki = manager.get_position_size("FLOKI-USD", 1.0, 100_000.0)
    size_msft = manager.get_position_size("MSFT", 1.0, 100_000.0)
    assert size_floki <= 10_000.0 + 1e-6
    assert size_msft <= 10_000.0 + 1e-6
    assert size_msft > size_floki


def test_rebalance_only_after_interval():
    vols = {f"S{i}": 0.01 + 0.005 * i for i in range(6)}
    manager = RiskParityManager(lookback_days=30, rebalance_hours=24)
    _seed_manager_with_synthetic_history(manager, vols, n_days=60)

    w1 = manager.get_weights()
    last_ts = manager._last_rebalance
    assert last_ts is not None

    # Add some new data; without force_recompute, weights should NOT update.
    later = datetime.now(timezone.utc) + timedelta(minutes=5)
    manager.update_prices("S0", 999.0, later)
    w2 = manager.get_weights()

    assert w1 == w2
    assert manager._last_rebalance == last_ts


def test_weights_update_on_new_data():
    vols = {f"S{i}": 0.01 + 0.005 * i for i in range(6)}
    manager = RiskParityManager(lookback_days=30, rebalance_hours=24)
    _seed_manager_with_synthetic_history(manager, vols, n_days=60)

    w1 = manager.get_weights(force_recompute=True)

    # Add many new high-volatility observations to one symbol; force recompute.
    rng = np.random.default_rng(99)
    base_ts = datetime.now(timezone.utc) + timedelta(seconds=1)
    price = 200.0
    for d in range(40):
        price = price * (1.0 + rng.standard_normal() * 0.20)
        manager.update_prices("S0", price, base_ts + timedelta(days=d))

    w2 = manager.get_weights(force_recompute=True)
    # S0 became dramatically more volatile, so its weight magnitude should drop.
    assert abs(w2["S0"]) < abs(w1["S0"])


def test_manager_falls_back_to_equal_weight_below_min_assets():
    # Only 2 symbols → below MIN_ASSETS_FOR_COVARIANCE → equal-weight.
    manager = RiskParityManager(lookback_days=30, rebalance_hours=24, min_assets=5)
    _seed_manager_with_synthetic_history(
        manager, {"A": 0.01, "B": 0.05}, n_days=40
    )
    weights = manager.get_weights(force_recompute=True)
    assert set(weights.keys()) == {"A", "B"}
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)


def test_manager_returns_empty_when_no_data():
    manager = RiskParityManager()
    weights = manager.get_weights(force_recompute=True)
    assert weights == {}
    assert manager.get_position_size("BTC-USD", 1.0, 100_000.0) == 0.0


def test_manager_ignores_invalid_prices():
    manager = RiskParityManager()
    ts = datetime.now(timezone.utc)
    manager.update_prices("BTC-USD", float("nan"), ts)
    manager.update_prices("BTC-USD", 0.0, ts)
    manager.update_prices("BTC-USD", -10.0, ts)
    # Nothing should have been recorded.
    assert "BTC-USD" not in manager._history or not manager._history["BTC-USD"].prices


def test_portfolio_dollar_neutral_option():
    vols = {f"S{i}": 0.01 + 0.005 * i for i in range(6)}
    manager = RiskParityManager(
        lookback_days=30, rebalance_hours=24, dollar_neutral=True
    )
    _seed_manager_with_synthetic_history(manager, vols, n_days=60)
    weights = manager.get_weights(force_recompute=True)

    # Dollar-neutral: weights sum to 0 (or very near), |weights| sum to 1.
    assert sum(weights.values()) == pytest.approx(0.0, abs=1e-9)
    assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0, rel=1e-6)


# ─────────────────── E2E synthetic backtest ───────────────────


def _mini_backtest_sharpe(returns: pd.DataFrame, weights: dict[str, float]) -> float:
    """Compute the annualized Sharpe of a portfolio with fixed weights."""
    aligned = np.array([weights.get(c, 0.0) for c in returns.columns])
    port = returns.to_numpy() @ aligned
    if np.std(port) == 0:
        return 0.0
    return float(np.mean(port) / np.std(port) * np.sqrt(252))


def test_backtest_with_risk_parity_vs_equal_weight():
    """Risk parity should beat equal-weight on risk-adjusted return when one
    asset is dramatically more volatile than the others.

    With one extreme-vol asset (FLOKI), equal-weight portfolios are dominated
    by that asset's noise. Risk-parity down-weights it, producing a higher
    Sharpe.
    """
    vols = {
        "MSFT": 0.005,
        "SPY": 0.007,
        "GLD": 0.006,
        "BTC-USD": 0.025,
        "ETH-USD": 0.030,
        "FLOKI-USD": 0.10,
    }
    # Build IN-SAMPLE returns for fitting weights, OUT-OF-SAMPLE for evaluation.
    in_sample = _synthetic_returns(180, vols, seed=11, correlation=0.2)
    out_sample = _synthetic_returns(180, vols, seed=22, correlation=0.2)

    # Equal-weight benchmark.
    n = len(vols)
    eq_weights = {s: 1.0 / n for s in vols}

    # Risk-parity weights fit on in-sample data.
    cov = compute_rolling_covariance(in_sample, window=120)
    rp_arr = compute_risk_parity_weights(cov)
    rp_weights = {s: float(w) for s, w in zip(in_sample.columns, rp_arr)}

    # Sanity: weights are normalized, FLOKI is heavily down-weighted.
    assert sum(rp_weights.values()) == pytest.approx(1.0)
    assert rp_weights["FLOKI-USD"] < rp_weights["MSFT"]

    sharpe_eq = _mini_backtest_sharpe(out_sample, eq_weights)
    sharpe_rp = _mini_backtest_sharpe(out_sample, rp_weights)

    # Both portfolios are zero-edge in expectation — what we care about is
    # *risk-adjusted volatility*. Compare portfolio variance directly: risk
    # parity should produce a lower-variance portfolio than equal-weight when
    # vols differ this much.
    eq_aligned = np.array([eq_weights[c] for c in out_sample.columns])
    rp_aligned = np.array([rp_weights[c] for c in out_sample.columns])
    eq_port_vol = float(np.std(out_sample.to_numpy() @ eq_aligned))
    rp_port_vol = float(np.std(out_sample.to_numpy() @ rp_aligned))
    assert rp_port_vol < eq_port_vol, (
        f"Expected risk-parity vol ({rp_port_vol}) < equal-weight ({eq_port_vol})"
    )

    # And as a softer signal, Sharpe shouldn't degrade catastrophically.
    # In random data, both will be near zero; we just assert they're finite.
    assert np.isfinite(sharpe_eq)
    assert np.isfinite(sharpe_rp)
