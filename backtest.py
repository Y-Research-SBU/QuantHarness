"""
Backtesting framework for QuantAgent strategies (including the Kronos suite).

Walks historical OHLCV data candle-by-candle, asks each registered strategy
for a signal at every step, and simulates a single open position per
strategy per symbol. Tracks the resulting trade ledger and reports a
standard set of performance metrics (Sharpe, max drawdown, win rate,
profit factor, total return).

Usage:
    python backtest.py                  # default: BTC-USD 1h, SPY 1d, 1y
    python backtest.py --symbols BTC-USD --years 2 --interval 1h
    python backtest.py --strategies kronos_divergence multi_factor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_fetcher import fetch_market_data
from market_config import StrategyType
from strategies import STRATEGIES, BaseStrategy, Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BacktestTrade:
    """A single simulated trade."""

    symbol: str
    strategy: str
    direction: str
    entry_time: str
    entry_price: float
    exit_time: Optional[str]
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    quantity: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str  # "take_profit", "stop_loss", "end_of_data"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """Aggregate result for one (symbol, strategy) backtest."""

    symbol: str
    strategy: str
    timeframe: str
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    trades: List[BacktestTrade] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trades"] = [t for t in d["trades"]]
        return d

    def summary_line(self) -> str:
        return (
            f"{self.symbol:<10} {self.strategy:<28} "
            f"trades={self.total_trades:>4} wr={self.win_rate * 100:5.1f}% "
            f"return={self.total_return_pct:+7.2f}% sharpe={self.sharpe_ratio:+5.2f} "
            f"mdd={self.max_drawdown_pct:5.2f}% pf={self.profit_factor:5.2f}"
        )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """Bar-by-bar single-position backtester."""

    def __init__(
        self,
        starting_capital: float = 10_000.0,
        risk_per_trade_pct: float = 0.02,
        warmup_bars: int = 60,
        max_holding_bars: int = 500,
        commission_pct: float = 0.0005,
        seed: Optional[int] = 42,
        signal_lookback: int = 200,
    ):
        self.starting_capital = starting_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.warmup_bars = warmup_bars
        self.max_holding_bars = max_holding_bars
        self.commission_pct = commission_pct
        # ``signal_lookback`` caps the number of trailing bars passed to
        # ``strategy.generate_signal``. Strategies in this codebase look back
        # at most ~50 bars, so 200 is generous while keeping the per-bar cost
        # O(1) instead of O(N) on a growing history.
        self.signal_lookback = signal_lookback
        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Core walk-forward loop
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> BacktestResult:
        """Run a single (strategy, symbol) backtest over ``df``."""
        symbol = symbol or df.attrs.get("symbol", "UNKNOWN")
        timeframe = timeframe or df.attrs.get("timeframe", "1d")

        if len(df) <= self.warmup_bars + 5:
            return self._empty_result(symbol, strategy.strategy_type.value, timeframe)

        df = df.reset_index(drop=True).copy()
        df.attrs["symbol"] = symbol
        df.attrs["timeframe"] = timeframe

        capital = self.starting_capital
        equity_curve: List[float] = [capital]
        trades: List[BacktestTrade] = []
        open_trade: Optional[BacktestTrade] = None
        bars_in_trade = 0

        n = len(df)
        for i in range(self.warmup_bars, n):
            start = max(0, i + 1 - self.signal_lookback)
            window = df.iloc[start : i + 1]
            window.attrs["symbol"] = symbol
            window.attrs["timeframe"] = timeframe
            current_bar = df.iloc[i]
            current_close = float(current_bar["Close"])
            current_high = float(current_bar["High"])
            current_low = float(current_bar["Low"])
            current_time = self._format_time(current_bar.get("Datetime", i))

            # Check open trade for exit
            if open_trade is not None:
                bars_in_trade += 1
                exit_price, exit_reason = self._check_exit(
                    open_trade, current_high, current_low, bars_in_trade
                )
                if exit_price is not None:
                    capital = self._close_trade(open_trade, exit_price, exit_reason, current_time, capital, bars_in_trade)
                    trades.append(open_trade)
                    open_trade = None
                    bars_in_trade = 0

            equity_curve.append(self._mark_to_market(capital, open_trade, current_close))

            # Look for a new entry only when flat
            if open_trade is None:
                signal = self._safe_generate(strategy, window)
                if signal is not None and signal.direction in ("LONG", "SHORT"):
                    open_trade = self._open_trade(
                        signal, capital, current_time
                    )
                    bars_in_trade = 0

        # Close any open trade at the final bar
        if open_trade is not None:
            final_bar = df.iloc[-1]
            final_price = float(final_bar["Close"])
            final_time = self._format_time(final_bar.get("Datetime", n - 1))
            capital = self._close_trade(open_trade, final_price, "end_of_data", final_time, capital, bars_in_trade)
            trades.append(open_trade)
            equity_curve[-1] = capital

        return self._build_result(
            symbol=symbol,
            strategy_name=strategy.strategy_type.value,
            timeframe=timeframe,
            trades=trades,
            equity_curve=np.array(equity_curve, dtype=float),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_generate(self, strategy: BaseStrategy, window: pd.DataFrame) -> Optional[Signal]:
        try:
            return strategy.generate_signal(window)
        except Exception as exc:
            logger.debug("Strategy %s threw on window len=%d: %s", strategy.strategy_type.value, len(window), exc)
            return None

    def _check_exit(
        self,
        trade: BacktestTrade,
        high: float,
        low: float,
        bars_held: int,
    ) -> Tuple[Optional[float], str]:
        """Return (exit_price, reason) if this bar would trigger an exit."""
        if bars_held >= self.max_holding_bars:
            # Time exit at the close of the bar
            return (high + low) / 2.0, "time_exit"

        if trade.direction == "LONG":
            if low <= trade.stop_loss:
                return trade.stop_loss, "stop_loss"
            if high >= trade.take_profit:
                return trade.take_profit, "take_profit"
        else:  # SHORT
            if high >= trade.stop_loss:
                return trade.stop_loss, "stop_loss"
            if low <= trade.take_profit:
                return trade.take_profit, "take_profit"
        return None, ""

    def _open_trade(self, signal: Signal, capital: float, time: str) -> BacktestTrade:
        risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        if risk_per_unit <= 0:
            qty = 0.0
        else:
            qty = (capital * self.risk_per_trade_pct) / risk_per_unit
            # Cap at 100% of capital notional
            max_qty = capital / max(signal.entry_price, 1e-9)
            qty = min(qty, max_qty)
        return BacktestTrade(
            symbol=signal.symbol,
            strategy=signal.strategy.value,
            direction=signal.direction,
            entry_time=time,
            entry_price=float(signal.entry_price),
            exit_time=None,
            exit_price=None,
            stop_loss=float(signal.stop_loss),
            take_profit=float(signal.take_profit),
            quantity=float(qty),
            pnl=0.0,
            pnl_pct=0.0,
            bars_held=0,
            exit_reason="",
        )

    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_price: float,
        reason: str,
        time: str,
        capital: float,
        bars_held: int,
    ) -> float:
        if trade.direction == "LONG":
            gross = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross = (trade.entry_price - exit_price) * trade.quantity
        commission = (abs(trade.entry_price) + abs(exit_price)) * trade.quantity * self.commission_pct
        net = gross - commission

        trade.exit_price = float(exit_price)
        trade.exit_time = time
        trade.exit_reason = reason
        trade.pnl = float(net)
        trade.bars_held = bars_held
        if trade.entry_price > 0 and trade.quantity > 0:
            trade.pnl_pct = float(net / (trade.entry_price * trade.quantity) * 100.0)
        return capital + net

    def _mark_to_market(
        self, capital: float, open_trade: Optional[BacktestTrade], current_price: float
    ) -> float:
        if open_trade is None:
            return capital
        if open_trade.direction == "LONG":
            unreal = (current_price - open_trade.entry_price) * open_trade.quantity
        else:
            unreal = (open_trade.entry_price - current_price) * open_trade.quantity
        return capital + unreal

    @staticmethod
    def _format_time(value: Any) -> str:
        if isinstance(value, (pd.Timestamp, datetime)):
            return pd.Timestamp(value).isoformat()
        try:
            return pd.Timestamp(value).isoformat()
        except (ValueError, TypeError):
            return str(value)

    # ------------------------------------------------------------------
    # Result building & metrics
    # ------------------------------------------------------------------

    def _empty_result(self, symbol: str, strategy: str, timeframe: str) -> BacktestResult:
        return BacktestResult(
            symbol=symbol,
            strategy=strategy,
            timeframe=timeframe,
            starting_capital=self.starting_capital,
            ending_capital=self.starting_capital,
            total_return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            trades=[],
        )

    def _build_result(
        self,
        symbol: str,
        strategy_name: str,
        timeframe: str,
        trades: List[BacktestTrade],
        equity_curve: np.ndarray,
    ) -> BacktestResult:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = len(wins) / len(trades) if trades else 0.0

        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

        starting = float(equity_curve[0])
        ending = float(equity_curve[-1])
        total_return_pct = (ending - starting) / starting * 100.0 if starting > 0 else 0.0

        sharpe = compute_sharpe(equity_curve)
        mdd = compute_max_drawdown_pct(equity_curve)

        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
        best = float(max((t.pnl for t in trades), default=0.0))
        worst = float(min((t.pnl for t in trades), default=0.0))

        return BacktestResult(
            symbol=symbol,
            strategy=strategy_name,
            timeframe=timeframe,
            starting_capital=starting,
            ending_capital=ending,
            total_return_pct=total_return_pct,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown_pct=mdd,
            profit_factor=profit_factor if np.isfinite(profit_factor) else 0.0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best,
            worst_trade=worst,
            trades=trades,
        )


# ---------------------------------------------------------------------------
# Metric helpers (exported for tests)
# ---------------------------------------------------------------------------


def compute_sharpe(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    """Sharpe ratio computed from per-bar returns of the equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / np.where(equity_curve[:-1] == 0, 1.0, equity_curve[:-1])
    if returns.size == 0:
        return 0.0
    sigma = float(np.std(returns))
    if sigma == 0:
        return 0.0
    mu = float(np.mean(returns))
    return float(mu / sigma * np.sqrt(periods_per_year))


def compute_max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """Max drawdown of the equity curve, expressed as a positive percentage."""
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / np.where(running_max == 0, 1.0, running_max)
    return float(abs(np.min(drawdowns)) * 100.0)


# ---------------------------------------------------------------------------
# Multi-symbol / multi-strategy driver
# ---------------------------------------------------------------------------


def fetch_history(symbol: str, interval: str, years: float) -> pd.DataFrame:
    """Fetch an extended history for a symbol via yfinance."""
    end = datetime.utcnow()
    start = end - timedelta(days=int(years * 365))
    df = fetch_market_data(symbol, interval=interval, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index(drop=True)
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = interval
    return df


def run_backtest_suite(
    symbols_intervals: Sequence[Tuple[str, str]],
    strategies: Optional[Sequence[StrategyType]] = None,
    years: float = 1.0,
    starting_capital: float = 10_000.0,
    data_loader: Optional[Callable[[str, str, float], pd.DataFrame]] = None,
    backtester: Optional[Backtester] = None,
) -> List[BacktestResult]:
    """Run all (symbol, interval, strategy) combinations and return results."""
    if strategies is None:
        strategies = list(StrategyType)
    if data_loader is None:
        data_loader = fetch_history
    if backtester is None:
        backtester = Backtester(starting_capital=starting_capital)

    results: List[BacktestResult] = []

    for symbol, interval in symbols_intervals:
        df = data_loader(symbol, interval, years)
        if df is None or df.empty:
            logger.warning("No data for %s (%s) — skipping", symbol, interval)
            continue
        for st in strategies:
            strategy = STRATEGIES.get(st)
            if strategy is None:
                continue
            logger.info("Backtesting %s on %s (%s)", st.value, symbol, interval)
            result = backtester.run(df.copy(), strategy, symbol=symbol, timeframe=interval)
            results.append(result)

    return results


def summarise_results(results: Sequence[BacktestResult]) -> str:
    if not results:
        return "(no backtest results)"
    lines = [
        "Symbol     Strategy                     Trades   WinRate  Return    Sharpe   MDD     PF",
        "-" * 92,
    ]
    for r in results:
        lines.append(r.summary_line())
    return "\n".join(lines)


def save_results(results: Sequence[BacktestResult], output_dir: str) -> str:
    """Persist results as a JSON file under ``output_dir`` and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"backtest_{timestamp}.json")
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DEFAULT_SUITE: Tuple[Tuple[str, str], ...] = (
    ("BTC-USD", "1h"),
    ("SPY", "1d"),
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantAgent backtester")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to backtest (defaults to BTC-USD and SPY).",
    )
    parser.add_argument(
        "--interval",
        default=None,
        help="Single override interval for all symbols. If not set: BTC-USD=1h, others=1d.",
    )
    parser.add_argument("--years", type=float, default=1.0, help="History length in years.")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Strategy names to run (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_results",
        help="Directory to write the JSON result file into.",
    )
    parser.add_argument(
        "--starting-capital", type=float, default=10_000.0, help="Starting capital per backtest."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip writing the JSON output file."
    )
    return parser


def _resolve_strategies(names: Optional[Sequence[str]]) -> List[StrategyType]:
    if not names:
        return list(StrategyType)
    out: List[StrategyType] = []
    for n in names:
        try:
            out.append(StrategyType(n))
        except ValueError:
            logger.warning("Unknown strategy '%s' — skipping", n)
    return out


def _resolve_suite(symbols: Optional[Sequence[str]], interval: Optional[str]) -> List[Tuple[str, str]]:
    if not symbols:
        return list(DEFAULT_SUITE)
    out: List[Tuple[str, str]] = []
    for s in symbols:
        if interval:
            out.append((s, interval))
        elif "-USD" in s:
            out.append((s, "1h"))
        else:
            out.append((s, "1d"))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    suite = _resolve_suite(args.symbols, args.interval)
    strategies = _resolve_strategies(args.strategies)

    print(f"Running backtest suite: {suite} | strategies={[s.value for s in strategies]} | years={args.years}")
    results = run_backtest_suite(
        symbols_intervals=suite,
        strategies=strategies,
        years=args.years,
        starting_capital=args.starting_capital,
    )

    print()
    print(summarise_results(results))
    print()

    if not args.no_save:
        path = save_results(results, args.output_dir)
        print(f"Saved {len(results)} results to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
