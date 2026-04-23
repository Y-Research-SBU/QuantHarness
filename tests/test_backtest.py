"""Tests for backtest.py — metric helpers and walk-forward backtester."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from backtest import (
    Backtester,
    BacktestResult,
    BacktestTrade,
    _resolve_strategies,
    _resolve_suite,
    compute_max_drawdown_pct,
    compute_sharpe,
    main,
    run_backtest_suite,
    save_results,
    summarise_results,
)
from market_config import StrategyType
from strategies import BaseStrategy, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 200, drift: float = 0.0008, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * vol + drift
    prices = 100.0 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "Datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="h"),
        "Open": prices,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": rng.integers(1000, 10000, size=n),
    })
    df.attrs["symbol"] = "TST-USD"
    df.attrs["timeframe"] = "1h"
    return df


class _AlwaysLongStrategy(BaseStrategy):
    """A strategy that emits a tight LONG signal whenever it is flat."""

    def __init__(self, take_profit_pct: float = 1.0, stop_loss_pct: float = 1.0):
        super().__init__(StrategyType.MOMENTUM)
        self.tp = take_profit_pct / 100.0
        self.sl = stop_loss_pct / 100.0

    def generate_signal(self, df, indicator_data=None, agent_reports=None) -> Optional[Signal]:
        price = float(df["Close"].iloc[-1])
        return Signal(
            direction="LONG",
            strength=0.8,
            strategy=self.strategy_type,
            symbol=df.attrs.get("symbol", "TST"),
            timeframe=df.attrs.get("timeframe", "1h"),
            entry_price=price,
            stop_loss=price * (1 - self.sl),
            take_profit=price * (1 + self.tp),
            risk_reward_ratio=self.tp / self.sl,
            reasoning="always-long stub",
        )


class _NoSignalStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(StrategyType.MOMENTUM)

    def generate_signal(self, df, indicator_data=None, agent_reports=None) -> Optional[Signal]:
        return None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def test_compute_sharpe_zero_vol():
    assert compute_sharpe(np.array([100.0, 100.0, 100.0])) == 0.0


def test_compute_sharpe_positive():
    eq = np.array([100.0, 101.0, 102.5, 104.0, 105.0])
    sharpe = compute_sharpe(eq)
    assert sharpe > 0


def test_compute_sharpe_handles_short_curve():
    assert compute_sharpe(np.array([100.0])) == 0.0
    assert compute_sharpe(np.array([])) == 0.0


def test_compute_max_drawdown_no_drawdown():
    eq = np.array([100.0, 101.0, 102.0, 103.0])
    assert compute_max_drawdown_pct(eq) == 0.0


def test_compute_max_drawdown_positive_value():
    eq = np.array([100.0, 110.0, 90.0, 95.0])
    mdd = compute_max_drawdown_pct(eq)
    assert mdd == pytest.approx(((110.0 - 90.0) / 110.0) * 100.0, rel=1e-4)


def test_compute_max_drawdown_empty():
    assert compute_max_drawdown_pct(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Backtester behaviour
# ---------------------------------------------------------------------------


def test_backtester_returns_empty_result_for_short_df():
    bt = Backtester(warmup_bars=50, starting_capital=1000.0)
    df = _make_df(n=20)
    result = bt.run(df, _AlwaysLongStrategy(), symbol="TEST")
    assert result.total_trades == 0
    assert result.starting_capital == 1000.0
    assert result.ending_capital == 1000.0


def test_backtester_no_trades_when_strategy_silent():
    bt = Backtester(warmup_bars=30, starting_capital=1000.0)
    result = bt.run(_make_df(n=120), _NoSignalStrategy(), symbol="TEST")
    assert result.total_trades == 0
    assert result.win_rate == 0.0
    assert result.sharpe_ratio == 0.0


def test_backtester_executes_trades_with_always_long_strategy():
    bt = Backtester(
        warmup_bars=20,
        starting_capital=10_000.0,
        max_holding_bars=40,
        commission_pct=0.0,
    )
    result = bt.run(_make_df(n=200, seed=1), _AlwaysLongStrategy(), symbol="TEST")
    assert result.total_trades >= 1
    assert result.winning_trades + result.losing_trades == result.total_trades
    # exit reasons should fall into the supported set
    for trade in result.trades:
        assert trade.exit_reason in {"take_profit", "stop_loss", "end_of_data", "time_exit"}


def test_backtester_records_take_profit_and_stop_loss():
    bt = Backtester(
        warmup_bars=20,
        starting_capital=10_000.0,
        max_holding_bars=200,
        commission_pct=0.0,
    )
    result = bt.run(_make_df(n=400, vol=0.02, seed=2), _AlwaysLongStrategy(), symbol="TEST")
    reasons = {t.exit_reason for t in result.trades}
    assert "take_profit" in reasons or "stop_loss" in reasons


def test_backtester_capital_changes_with_pnl():
    bt = Backtester(warmup_bars=20, starting_capital=10_000.0, max_holding_bars=60, commission_pct=0.0)
    result = bt.run(_make_df(n=300, seed=3), _AlwaysLongStrategy(), symbol="TEST")
    expected = 10_000.0 + sum(t.pnl for t in result.trades)
    assert result.ending_capital == pytest.approx(expected, rel=1e-6)


def test_backtester_open_trade_closes_at_end_of_data():
    bt = Backtester(
        warmup_bars=10,
        starting_capital=10_000.0,
        max_holding_bars=10_000,
        commission_pct=0.0,
    )
    # Use a strategy with extremely wide SL/TP so trades stay open until EOD.
    result = bt.run(
        _make_df(n=100, vol=0.001, seed=4),
        _AlwaysLongStrategy(take_profit_pct=50.0, stop_loss_pct=50.0),
        symbol="TEST",
    )
    assert result.total_trades >= 1
    assert result.trades[-1].exit_reason in {"end_of_data", "time_exit"}


def test_backtester_handles_strategy_exception():
    class BoomStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(StrategyType.MOMENTUM)

        def generate_signal(self, df, indicator_data=None, agent_reports=None):
            raise RuntimeError("kapow")

    bt = Backtester(warmup_bars=20, starting_capital=1000.0)
    result = bt.run(_make_df(n=120), BoomStrategy(), symbol="TEST")
    # Should not raise — and should produce zero trades.
    assert result.total_trades == 0


def test_backtester_metadata_carries_through():
    bt = Backtester(warmup_bars=20, starting_capital=1000.0, max_holding_bars=40, commission_pct=0.0)
    result = bt.run(_make_df(n=200, seed=5), _AlwaysLongStrategy(), symbol="ABCD", timeframe="4h")
    assert result.symbol == "ABCD"
    assert result.timeframe == "4h"
    for trade in result.trades:
        assert trade.symbol == "ABCD"


def test_backtester_summary_line_includes_metrics():
    result = BacktestResult(
        symbol="TST", strategy="momentum", timeframe="1h",
        starting_capital=1000.0, ending_capital=1100.0, total_return_pct=10.0,
        total_trades=5, winning_trades=3, losing_trades=2, win_rate=0.6,
        sharpe_ratio=1.2, max_drawdown_pct=5.0, profit_factor=1.5,
        avg_win=50.0, avg_loss=-20.0, best_trade=80.0, worst_trade=-30.0,
        trades=[],
    )
    line = result.summary_line()
    assert "TST" in line and "momentum" in line and "+10.00%" in line


def test_summarise_results_with_no_results():
    assert "no backtest results" in summarise_results([])


def test_summarise_results_with_results():
    res = BacktestResult(
        symbol="X", strategy="momentum", timeframe="1h",
        starting_capital=1000.0, ending_capital=1050.0, total_return_pct=5.0,
        total_trades=2, winning_trades=2, losing_trades=0, win_rate=1.0,
        sharpe_ratio=2.0, max_drawdown_pct=1.0, profit_factor=10.0,
        avg_win=25.0, avg_loss=0.0, best_trade=30.0, worst_trade=20.0, trades=[],
    )
    text = summarise_results([res])
    assert "Symbol" in text and "X" in text and "momentum" in text


def test_save_results_writes_json(tmp_path):
    res = BacktestResult(
        symbol="X", strategy="momentum", timeframe="1h",
        starting_capital=1000.0, ending_capital=1050.0, total_return_pct=5.0,
        total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
        sharpe_ratio=0.0, max_drawdown_pct=0.0, profit_factor=0.0,
        avg_win=0.0, avg_loss=0.0, best_trade=0.0, worst_trade=0.0, trades=[],
    )
    path = save_results([res], output_dir=str(tmp_path))
    assert os.path.exists(path)
    payload = json.loads(open(path).read())
    assert payload["results"][0]["strategy"] == "momentum"


# ---------------------------------------------------------------------------
# Suite driver
# ---------------------------------------------------------------------------


def test_run_backtest_suite_with_data_loader():
    df = _make_df(n=200, seed=10)

    def loader(symbol, interval, years):
        d = df.copy()
        d.attrs["symbol"] = symbol
        d.attrs["timeframe"] = interval
        return d

    results = run_backtest_suite(
        symbols_intervals=[("TEST", "1h")],
        strategies=[StrategyType.MOMENTUM],
        years=0.5,
        data_loader=loader,
        backtester=Backtester(warmup_bars=20, starting_capital=1000.0, max_holding_bars=30, commission_pct=0.0),
    )
    assert len(results) == 1
    assert results[0].symbol == "TEST"
    assert results[0].strategy == StrategyType.MOMENTUM.value


def test_run_backtest_suite_skips_missing_data(caplog):
    def loader(symbol, interval, years):
        return pd.DataFrame()
    results = run_backtest_suite(
        symbols_intervals=[("EMPTY", "1d")],
        strategies=[StrategyType.MOMENTUM],
        data_loader=loader,
    )
    assert results == []


def test_resolve_strategies_default():
    out = _resolve_strategies(None)
    assert StrategyType.MOMENTUM in out
    assert StrategyType.KRONOS_MOMENTUM_CONFIRM in out


def test_resolve_strategies_filters_unknown():
    out = _resolve_strategies(["momentum", "not_a_strategy"])
    assert out == [StrategyType.MOMENTUM]


def test_resolve_suite_default():
    suite = _resolve_suite(None, None)
    assert ("BTC-USD", "1h") in suite


def test_resolve_suite_with_interval_override():
    suite = _resolve_suite(["AAPL", "BTC-USD"], "4h")
    assert suite == [("AAPL", "4h"), ("BTC-USD", "4h")]


def test_resolve_suite_picks_default_interval():
    suite = _resolve_suite(["AAPL"], None)
    assert suite == [("AAPL", "1d")]
    suite2 = _resolve_suite(["BTC-USD"], None)
    assert suite2 == [("BTC-USD", "1h")]


def test_main_skips_save_when_flag_set(tmp_path, monkeypatch):
    df = _make_df(n=120, seed=11)

    def fake_fetch(symbol, interval, years):
        d = df.copy()
        d.attrs["symbol"] = symbol
        d.attrs["timeframe"] = interval
        return d

    monkeypatch.setattr("backtest.fetch_history", fake_fetch)
    rc = main([
        "--symbols", "TEST",
        "--interval", "1h",
        "--years", "0.1",
        "--strategies", "momentum",
        "--output-dir", str(tmp_path),
        "--no-save",
    ])
    assert rc == 0
    assert os.listdir(tmp_path) == []
