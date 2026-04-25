"""
Pluggable strategy framework for QuantAgent.

Strategies:
1. Momentum — Follow trends on 4hr/daily. Enter on pullbacks in trend direction.
2. Mean Reversion — Fade RSI extremes (>70 short, <30 long) on 1hr timeframe.
3. Breakout — Pattern agent detects formation → enter on breakout with volume confirmation.
4. Multi-Factor — Weighted scoring from all 5 agents. Only trade when 4/5 agree.
5. Kronos Momentum Confirm — Take Kronos directional bets when indicators/patterns agree.
6. Kronos Divergence — Contrarian entry when Kronos disagrees with current trend.
7. Multi-Timeframe Kronos — Trade only when Kronos agrees across multiple horizons.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from adaptive_params import get_timeframe_defaults
from market_config import StrategyType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared lazy KronosForecastAgent (used by Kronos-powered strategies)
# ---------------------------------------------------------------------------

_SHARED_KRONOS_AGENT = None


def _get_shared_kronos_agent():
    """Return a process-wide :class:`KronosForecastAgent` singleton."""
    global _SHARED_KRONOS_AGENT
    if _SHARED_KRONOS_AGENT is None:
        from kronos_agent import KronosForecastAgent  # local import to avoid heavy import at module load
        _SHARED_KRONOS_AGENT = KronosForecastAgent()
    return _SHARED_KRONOS_AGENT


def _resolve_params(
    adaptive_params: Optional[Dict[str, float]],
    timeframe: Optional[str],
) -> Dict[str, float]:
    """Merge timeframe-aware defaults with caller-supplied adaptive params.

    Adaptive (per-strategy/per-symbol tuned) values win over timeframe defaults
    so the self-improver's L2 output always takes precedence once it has data.
    """
    merged = get_timeframe_defaults(timeframe)
    if adaptive_params:
        for k, v in adaptive_params.items():
            if v is not None:
                merged[k] = v
    return merged


def _extract_kronos_forecast(agent_reports: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pull the structured Kronos forecast dict out of the agent_reports payload."""
    if not agent_reports:
        return None
    for key in ("kronos_forecast_data", "kronos", "kronos_data"):
        value = agent_reports.get(key)
        if isinstance(value, dict) and "direction" in value:
            return value
    return None


@dataclass
class Signal:
    """Trading signal produced by a strategy."""
    direction: str          # "LONG", "SHORT", or "NEUTRAL"
    strength: float         # 0.0 to 1.0
    strategy: StrategyType
    symbol: str
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _compute_vwap(df: pd.DataFrame) -> float:
    """Volume-weighted average price across the supplied frame."""
    if "Volume" not in df.columns or len(df) == 0:
        return float(df["Close"].iloc[-1]) if len(df) else 0.0
    typical = (df["High"].values + df["Low"].values + df["Close"].values) / 3.0
    vol = df["Volume"].values.astype(float)
    vol_sum = float(np.sum(vol))
    if vol_sum <= 0:
        return float(np.mean(typical))
    return float(np.sum(typical * vol) / vol_sum)


def _compute_bollinger_bands(
    close: np.ndarray, period: int = 20, num_std: float = 2.0
) -> Tuple[float, float, float, float]:
    """Return (upper, middle, lower, width) for the most recent bar."""
    if len(close) < period:
        price = float(close[-1]) if len(close) else 0.0
        return price, price, price, 0.0
    window = close[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window))
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    return upper, middle, lower, width


def _rolling_bb_widths(
    close: np.ndarray, lookback: int, period: int = 20, num_std: float = 2.0
) -> np.ndarray:
    """Compute BB width for each of the last ``lookback`` bars."""
    widths: List[float] = []
    if len(close) < period + 1:
        return np.array(widths, dtype=float)
    start = max(0, len(close) - lookback)
    for i in range(start, len(close)):
        window = close[max(0, i - period + 1): i + 1]
        if len(window) < period:
            continue
        std = float(np.std(window))
        widths.append(2.0 * num_std * std)
    return np.array(widths, dtype=float)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, strategy_type: StrategyType):
        self.strategy_type = strategy_type
        self.enabled = True

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        """
        Generate a trading signal from market data and optional agent reports.

        Args:
            df: OHLCV DataFrame
            indicator_data: Pre-computed indicators (RSI, MACD, etc.)
            agent_reports: Reports from indicator/pattern/trend/decision agents
            adaptive_params: Optional tuned params from the self-improver
                (e.g. rsi_overbought/oversold, sl_atr_mult, tp_atr_mult).
                Strategies fall back to their own defaults when absent.

        Returns:
            Signal if trade opportunity found, None otherwise
        """
        pass
    
    def _compute_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic indicators from price data without TA-Lib dependency."""
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        
        result = {}
        
        # RSI
        result["rsi"] = self._compute_rsi(close)
        
        # MACD
        macd, signal, hist = self._compute_macd(close)
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = hist
        
        # Stochastic
        stoch_k, stoch_d = self._compute_stochastic(high, low, close)
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d
        
        # ATR
        result["atr"] = self._compute_atr(high, low, close)
        
        # SMA
        result["sma_20"] = self._compute_sma(close, 20)
        result["sma_50"] = self._compute_sma(close, 50)
        
        # EMA
        result["ema_12"] = self._compute_ema(close, 12)
        result["ema_26"] = self._compute_ema(close, 26)
        
        # Volume (if available)
        if "Volume" in df.columns:
            vol = df["Volume"].values
            result["volume_sma"] = self._compute_sma(vol.astype(float), 20)
            result["volume_ratio"] = vol[-1] / result["volume_sma"] if result["volume_sma"] > 0 else 1.0
        
        return result
    
    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        """Compute RSI from close prices. Returns latest RSI value."""
        if len(close) < period + 1:
            return 50.0  # Neutral if insufficient data
        
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)
    
    @staticmethod
    def _compute_macd(
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> Tuple[float, float, float]:
        """Compute MACD. Returns (macd_line, signal_line, histogram)."""
        if len(close) < slow + signal_period:
            return 0.0, 0.0, 0.0
        
        def ema(data, period):
            alpha = 2.0 / (period + 1)
            result = np.zeros_like(data, dtype=float)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result
        
        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])
    
    @staticmethod
    def _compute_stochastic(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[float, float]:
        """Compute Stochastic Oscillator. Returns (%K, %D)."""
        if len(close) < k_period:
            return 50.0, 50.0
        
        lowest_low = np.min(low[-k_period:])
        highest_high = np.max(high[-k_period:])
        
        if highest_high == lowest_low:
            return 50.0, 50.0
        
        stoch_k = 100.0 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        
        # Simple %D as average of last d_period %K values
        k_values = []
        for i in range(d_period):
            idx = -(i + 1)
            if abs(idx) > len(close) - k_period:
                break
            ll = np.min(low[idx - k_period + 1:len(low) + idx + 1] if idx != -1 else low[-k_period:])
            hh = np.max(high[idx - k_period + 1:len(high) + idx + 1] if idx != -1 else high[-k_period:])
            if hh != ll:
                k_values.append(100.0 * (close[idx] - ll) / (hh - ll))
        
        stoch_d = np.mean(k_values) if k_values else stoch_k
        
        return float(stoch_k), float(stoch_d)
    
    @staticmethod
    def _compute_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """Compute Average True Range."""
        if len(close) < period + 1:
            return 0.0
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr = np.mean(tr[-period:])
        return float(atr)
    
    @staticmethod
    def _compute_sma(data: np.ndarray, period: int) -> float:
        """Compute Simple Moving Average. Returns latest value."""
        if len(data) < period:
            return float(np.mean(data)) if len(data) > 0 else 0.0
        return float(np.mean(data[-period:]))
    
    @staticmethod
    def _compute_ema(data: np.ndarray, period: int) -> float:
        """Compute Exponential Moving Average. Returns latest value."""
        if len(data) == 0:
            return 0.0
        alpha = 2.0 / (period + 1)
        ema_val = data[0]
        for val in data[1:]:
            ema_val = alpha * val + (1 - alpha) * ema_val
        return float(ema_val)


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy: Follow trends on 4hr/daily timeframes.
    Enter on pullbacks in trend direction.
    """

    def __init__(self):
        super().__init__(StrategyType.MOMENTUM)

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < 50:
            return None

        indicators = indicator_data or self._compute_indicators(df)
        close = df["Close"].values
        current_price = float(close[-1])

        rsi = indicators.get("rsi", 50.0)
        macd_hist = indicators.get("macd_hist", 0.0)
        sma_20 = indicators.get("sma_20", current_price)
        sma_50 = indicators.get("sma_50", current_price)
        atr = indicators.get("atr", current_price * 0.02)

        # L2: timeframe-aware defaults overlaid with self-improver params.
        params = _resolve_params(adaptive_params, df.attrs.get("timeframe"))
        sl_mult = float(params["sl_atr_mult"])
        tp_mult = float(params["tp_atr_mult"])

        # Trend direction: price above both SMAs = uptrend
        uptrend = current_price > sma_20 and sma_20 > sma_50
        downtrend = current_price < sma_20 and sma_20 < sma_50

        # Pullback detection: RSI in moderate zone
        bullish_pullback = uptrend and 40 <= rsi <= 60 and macd_hist > 0
        bearish_pullback = downtrend and 40 <= rsi <= 60 and macd_hist < 0

        if bullish_pullback:
            stop_loss = current_price - sl_mult * atr
            take_profit = current_price + tp_mult * atr
            rr = (take_profit - current_price) / (current_price - stop_loss) if current_price > stop_loss else 1.5
            
            strength = min(1.0, (rsi - 30) / 40 * 0.5 + (0.5 if macd_hist > 0 else 0))
            
            return Signal(
                direction="LONG",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "4h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=f"Momentum LONG: Uptrend pullback. RSI={rsi:.1f}, MACD hist={macd_hist:.2f}, "
                          f"Price above SMA20({sma_20:.2f}) and SMA50({sma_50:.2f})",
                metadata=indicators,
            )
        
        elif bearish_pullback:
            stop_loss = current_price + sl_mult * atr
            take_profit = current_price - tp_mult * atr
            rr = (current_price - take_profit) / (stop_loss - current_price) if stop_loss > current_price else 1.5

            strength = min(1.0, (70 - rsi) / 40 * 0.5 + (0.5 if macd_hist < 0 else 0))

            return Signal(
                direction="SHORT",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "4h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=f"Momentum SHORT: Downtrend pullback. RSI={rsi:.1f}, MACD hist={macd_hist:.2f}, "
                          f"Price below SMA20({sma_20:.2f}) and SMA50({sma_50:.2f})",
                metadata=indicators,
            )

        return None


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion strategy: Fade RSI extremes.
    RSI > 70 → short, RSI < 30 → long.
    Best on 1hr timeframe.
    """

    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    def __init__(self):
        super().__init__(StrategyType.MEAN_REVERSION)

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < 20:
            return None

        indicators = indicator_data or self._compute_indicators(df)
        close = df["Close"].values
        current_price = float(close[-1])

        rsi = indicators.get("rsi", 50.0)
        stoch_k = indicators.get("stoch_k", 50.0)
        atr = indicators.get("atr", current_price * 0.02)
        sma_20 = indicators.get("sma_20", current_price)

        # L2: timeframe-aware defaults overlaid with self-improver params.
        params = _resolve_params(adaptive_params, df.attrs.get("timeframe"))
        overbought = float(params.get("rsi_overbought", self.RSI_OVERBOUGHT))
        oversold = float(params.get("rsi_oversold", self.RSI_OVERSOLD))
        sl_mult = float(params["sl_atr_mult"])

        # Overbought → SHORT
        if rsi > overbought and stoch_k > 80:
            stop_loss = current_price + sl_mult * atr
            take_profit = sma_20  # Revert to mean (SMA20)
            rr = abs(current_price - take_profit) / abs(stop_loss - current_price) if stop_loss != current_price else 1.0
            
            strength = min(1.0, (rsi - 70) / 30)
            
            return Signal(
                direction="SHORT",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=f"Mean Reversion SHORT: Overbought. RSI={rsi:.1f}, Stoch %K={stoch_k:.1f}. "
                          f"Target reversion to SMA20={sma_20:.2f}",
                metadata=indicators,
            )
        
        # Oversold → LONG
        elif rsi < oversold and stoch_k < 20:
            stop_loss = current_price - sl_mult * atr
            take_profit = sma_20  # Revert to mean
            rr = abs(take_profit - current_price) / abs(current_price - stop_loss) if current_price != stop_loss else 1.0
            
            strength = min(1.0, (30 - rsi) / 30)
            
            return Signal(
                direction="LONG",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=f"Mean Reversion LONG: Oversold. RSI={rsi:.1f}, Stoch %K={stoch_k:.1f}. "
                          f"Target reversion to SMA20={sma_20:.2f}",
                metadata=indicators,
            )
        
        return None


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy: Enter on breakout from consolidation patterns.
    Uses price range compression + volume confirmation.
    """
    
    def __init__(self):
        super().__init__(StrategyType.BREAKOUT)

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < 30:
            return None

        indicators = indicator_data or self._compute_indicators(df)
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        current_price = float(close[-1])

        atr = indicators.get("atr", current_price * 0.02)

        # L2: timeframe-aware SL distance for the breakout buffer.
        params = _resolve_params(adaptive_params, df.attrs.get("timeframe"))
        sl_mult = float(params["sl_atr_mult"])

        # Detect consolidation: recent range is narrow
        lookback = 20
        recent_high = np.max(high[-lookback:])
        recent_low = np.min(low[-lookback:])
        range_pct = (recent_high - recent_low) / recent_low if recent_low > 0 else 0
        
        # Short-term range (last 5 bars)
        short_high = np.max(high[-5:])
        short_low = np.min(low[-5:])
        
        # Volume confirmation
        volume_ratio = indicators.get("volume_ratio", 1.0)
        volume_confirmed = volume_ratio > 1.2  # 20% above average
        
        # Check for breakout
        breakout_up = current_price > recent_high and range_pct < 0.15
        breakout_down = current_price < recent_low and range_pct < 0.15
        
        # Also check using pattern agent report if available
        pattern_bullish = False
        pattern_bearish = False
        if agent_reports and "pattern_report" in agent_reports:
            report = agent_reports["pattern_report"].lower()
            bullish_patterns = ["ascending triangle", "bull flag", "inverse head and shoulders",
                              "double bottom", "rounded bottom", "falling wedge"]
            bearish_patterns = ["descending triangle", "bear flag", "head and shoulders",
                              "double top", "rising wedge"]
            pattern_bullish = any(p in report for p in bullish_patterns)
            pattern_bearish = any(p in report for p in bearish_patterns)
        
        if (breakout_up or pattern_bullish) and (volume_confirmed or pattern_bullish):
            stop_loss = recent_low - 0.5 * sl_mult * atr
            take_profit = current_price + (current_price - stop_loss) * 1.5
            rr = (take_profit - current_price) / (current_price - stop_loss) if current_price > stop_loss else 1.5
            
            strength = 0.5
            if volume_confirmed:
                strength += 0.25
            if pattern_bullish:
                strength += 0.25
            
            return Signal(
                direction="LONG",
                strength=min(1.0, strength),
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "4h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=f"Breakout LONG: Price broke above {recent_high:.2f} (range {range_pct:.1%}). "
                          f"Volume ratio: {volume_ratio:.1f}x. Pattern: {'bullish' if pattern_bullish else 'none'}",
                metadata=indicators,
            )
        
        elif (breakout_down or pattern_bearish) and (volume_confirmed or pattern_bearish):
            stop_loss = recent_high + 0.5 * sl_mult * atr
            take_profit = current_price - (stop_loss - current_price) * 1.5
            rr = (current_price - take_profit) / (stop_loss - current_price) if stop_loss > current_price else 1.5
            
            strength = 0.5
            if volume_confirmed:
                strength += 0.25
            if pattern_bearish:
                strength += 0.25
            
            return Signal(
                direction="SHORT",
                strength=min(1.0, strength),
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "4h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=f"Breakout SHORT: Price broke below {recent_low:.2f} (range {range_pct:.1%}). "
                          f"Volume ratio: {volume_ratio:.1f}x. Pattern: {'bearish' if pattern_bearish else 'none'}",
                metadata=indicators,
            )
        
        return None


class MultiFactorStrategy(BaseStrategy):
    """
    Multi-Factor strategy: Weighted scoring from all agents.
    Only trade when 4/5 signals agree on direction.
    """
    
    AGREEMENT_THRESHOLD = 4  # Need 4 out of 5 signals
    
    def __init__(self):
        super().__init__(StrategyType.MULTI_FACTOR)

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < 30:
            return None

        indicators = indicator_data or self._compute_indicators(df)
        close = df["Close"].values
        current_price = float(close[-1])
        atr = indicators.get("atr", current_price * 0.02)

        # L2: timeframe-aware SL/TP multipliers for the breakout-style stop.
        params = _resolve_params(adaptive_params, df.attrs.get("timeframe"))
        sl_mult = float(params["sl_atr_mult"])
        tp_mult = float(params["tp_atr_mult"])

        # Score from 5 factors
        scores = []
        reasons = []
        
        # 1. RSI signal
        rsi = indicators.get("rsi", 50.0)
        if rsi < 40:
            scores.append(1)  # Bullish
            reasons.append(f"RSI bullish ({rsi:.1f})")
        elif rsi > 60:
            scores.append(-1)  # Bearish
            reasons.append(f"RSI bearish ({rsi:.1f})")
        else:
            scores.append(0)
            reasons.append(f"RSI neutral ({rsi:.1f})")
        
        # 2. MACD signal
        macd_hist = indicators.get("macd_hist", 0.0)
        if macd_hist > 0:
            scores.append(1)
            reasons.append(f"MACD bullish (hist={macd_hist:.2f})")
        elif macd_hist < 0:
            scores.append(-1)
            reasons.append(f"MACD bearish (hist={macd_hist:.2f})")
        else:
            scores.append(0)
            reasons.append("MACD neutral")
        
        # 3. Trend (SMA crossover)
        sma_20 = indicators.get("sma_20", current_price)
        sma_50 = indicators.get("sma_50", current_price)
        if sma_20 > sma_50 and current_price > sma_20:
            scores.append(1)
            reasons.append("Trend bullish (SMA20 > SMA50)")
        elif sma_20 < sma_50 and current_price < sma_20:
            scores.append(-1)
            reasons.append("Trend bearish (SMA20 < SMA50)")
        else:
            scores.append(0)
            reasons.append("Trend neutral")
        
        # 4. Stochastic
        stoch_k = indicators.get("stoch_k", 50.0)
        stoch_d = indicators.get("stoch_d", 50.0)
        if stoch_k < 30 and stoch_k > stoch_d:
            scores.append(1)
            reasons.append(f"Stoch bullish (%K={stoch_k:.1f})")
        elif stoch_k > 70 and stoch_k < stoch_d:
            scores.append(-1)
            reasons.append(f"Stoch bearish (%K={stoch_k:.1f})")
        else:
            scores.append(0)
            reasons.append(f"Stoch neutral (%K={stoch_k:.1f})")
        
        # 5. Agent decision (if available)
        if agent_reports and "decision" in agent_reports:
            decision = agent_reports["decision"]
            if isinstance(decision, str):
                decision_upper = decision.upper()
            else:
                decision_upper = str(decision).upper()
            if "LONG" in decision_upper:
                scores.append(1)
                reasons.append("Agent decision: LONG")
            elif "SHORT" in decision_upper:
                scores.append(-1)
                reasons.append("Agent decision: SHORT")
            else:
                scores.append(0)
                reasons.append("Agent decision: neutral")
        else:
            # Without agent report, use price action
            if current_price > float(close[-2]) > float(close[-3]):
                scores.append(1)
                reasons.append("Price action: bullish (higher closes)")
            elif current_price < float(close[-2]) < float(close[-3]):
                scores.append(-1)
                reasons.append("Price action: bearish (lower closes)")
            else:
                scores.append(0)
                reasons.append("Price action: neutral")
        
        # Count agreements
        bullish_count = sum(1 for s in scores if s > 0)
        bearish_count = sum(1 for s in scores if s < 0)
        
        total_signals = len(scores)
        
        if bullish_count >= self.AGREEMENT_THRESHOLD:
            stop_loss = current_price - sl_mult * atr
            take_profit = current_price + tp_mult * atr
            rr = (take_profit - current_price) / (current_price - stop_loss) if current_price > stop_loss else 1.5
            
            return Signal(
                direction="LONG",
                strength=bullish_count / total_signals,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "4h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=f"Multi-Factor LONG: {bullish_count}/{total_signals} agree. " + "; ".join(reasons),
                metadata={**indicators, "scores": scores},
            )
        
        elif bearish_count >= self.AGREEMENT_THRESHOLD:
            stop_loss = current_price + sl_mult * atr
            take_profit = current_price - tp_mult * atr
            rr = (current_price - take_profit) / (stop_loss - current_price) if stop_loss > current_price else 1.5
            
            return Signal(
                direction="SHORT",
                strength=bearish_count / total_signals,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "4h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=f"Multi-Factor SHORT: {bearish_count}/{total_signals} agree. " + "; ".join(reasons),
                metadata={**indicators, "scores": scores},
            )
        
        return None


class KronosMomentumConfirmStrategy(BaseStrategy):
    """
    Take Kronos directional forecasts when classical indicators *and* the
    pattern agent agree.

    Long entry: Kronos predicts up ≥ ``min_pct`` over the forecast horizon
    AND RSI is in a non-overbought zone (< ``rsi_max``)
    AND the pattern report (if present) is bullish — or no pattern report
        rejects a bullish read.

    Short entry: mirror image (down ≥ ``min_pct``, RSI > ``rsi_min``,
    pattern report not strongly bullish).
    """

    BULLISH_PATTERNS = (
        "ascending triangle",
        "bull flag",
        "bullish",
        "inverse head and shoulders",
        "double bottom",
        "rounded bottom",
        "falling wedge",
        "morning star",
        "hammer",
    )
    BEARISH_PATTERNS = (
        "descending triangle",
        "bear flag",
        "bearish",
        "head and shoulders",
        "double top",
        "rising wedge",
        "evening star",
        "shooting star",
    )

    # Minimum risk:reward ratio required to take a Kronos-confirmed trade.
    # Raised from implicit 1.0 to 1.5 — cuts low-conviction setups that were
    # tanking win rate even though magnitude/confidence gates passed.
    MIN_RR = 1.5

    def __init__(
        self,
        min_pct: float = 3.0,
        rsi_max: float = 70.0,
        rsi_min: float = 30.0,
        min_confidence: float = 0.5,
        kronos_runner: Optional[Callable] = None,
    ):
        super().__init__(StrategyType.KRONOS_MOMENTUM_CONFIRM)
        self.min_pct = min_pct
        self.rsi_max = rsi_max
        self.rsi_min = rsi_min
        self.min_confidence = min_confidence
        self._kronos_runner = kronos_runner  # for tests / dependency injection

    # ------------------------------------------------------------------

    def _forecast(
        self, df: pd.DataFrame, agent_reports: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        cached = _extract_kronos_forecast(agent_reports)
        if cached is not None:
            return cached
        if self._kronos_runner is not None:
            try:
                out = self._kronos_runner(df)
                return out.to_dict() if hasattr(out, "to_dict") else out
            except Exception as exc:
                logger.warning("Kronos runner failed in %s: %s", self.strategy_type.value, exc)
                return None
        try:
            forecast = _get_shared_kronos_agent().predict(
                df, timeframe=df.attrs.get("timeframe")
            )
            return forecast.to_dict()
        except Exception as exc:
            logger.warning("Kronos prediction failed in %s: %s", self.strategy_type.value, exc)
            return None

    @staticmethod
    def _pattern_sentiment(agent_reports: Optional[Dict[str, Any]]) -> str:
        if not agent_reports:
            return "unknown"
        report = agent_reports.get("pattern_report") or ""
        if not isinstance(report, str):
            report = str(report)
        report_l = report.lower()
        bullish = any(p in report_l for p in KronosMomentumConfirmStrategy.BULLISH_PATTERNS)
        bearish = any(p in report_l for p in KronosMomentumConfirmStrategy.BEARISH_PATTERNS)
        if bullish and not bearish:
            return "bullish"
        if bearish and not bullish:
            return "bearish"
        if not report_l.strip():
            return "unknown"
        return "neutral"

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < 30:
            return None

        forecast = self._forecast(df, agent_reports)
        if not forecast:
            return None

        magnitude = float(forecast.get("magnitude_pct", 0.0))
        confidence = float(forecast.get("confidence", 0.0))
        # L2: timeframe-aware defaults overlaid with self-improver params.
        params = _resolve_params(adaptive_params, df.attrs.get("timeframe"))
        min_conf = float(params.get("kronos_min_confidence", self.min_confidence))
        sl_mult = float(params["sl_atr_mult"])
        tp_mult = float(params["tp_atr_mult"])
        if confidence < min_conf:
            return None

        indicators = indicator_data or self._compute_indicators(df)
        rsi = float(indicators.get("rsi", 50.0))
        atr = float(indicators.get("atr", df["Close"].iloc[-1] * 0.02))
        current_price = float(df["Close"].iloc[-1])
        sentiment = self._pattern_sentiment(agent_reports)

        # LONG conditions
        if magnitude >= self.min_pct and rsi < self.rsi_max and sentiment != "bearish":
            stop_loss = current_price - sl_mult * atr
            take_profit = current_price + max(tp_mult * atr, current_price * magnitude / 100.0)
            rr = (take_profit - current_price) / max(current_price - stop_loss, 1e-9)
            if rr < self.MIN_RR:
                return None
            strength = float(np.clip(0.5 * confidence + 0.5 * min(1.0, magnitude / 5.0), 0.0, 1.0))
            return Signal(
                direction="LONG",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"Kronos predicts {magnitude:+.2f}% (conf {confidence:.2f}); "
                    f"RSI={rsi:.1f} (< {self.rsi_max}); pattern sentiment={sentiment}."
                ),
                metadata={**indicators, "kronos": forecast},
            )

        # SHORT conditions
        if magnitude <= -self.min_pct and rsi > self.rsi_min and sentiment != "bullish":
            stop_loss = current_price + sl_mult * atr
            take_profit = current_price - max(tp_mult * atr, current_price * abs(magnitude) / 100.0)
            rr = (current_price - take_profit) / max(stop_loss - current_price, 1e-9)
            if rr < self.MIN_RR:
                return None
            strength = float(
                np.clip(0.5 * confidence + 0.5 * min(1.0, abs(magnitude) / 5.0), 0.0, 1.0)
            )
            return Signal(
                direction="SHORT",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"Kronos predicts {magnitude:+.2f}% (conf {confidence:.2f}); "
                    f"RSI={rsi:.1f} (> {self.rsi_min}); pattern sentiment={sentiment}."
                ),
                metadata={**indicators, "kronos": forecast},
            )

        return None


class KronosDivergenceStrategy(BaseStrategy):
    """
    Contrarian strategy: trade *against* the prevailing trend when Kronos
    disagrees with strong conviction.

    Trend bullish (price > SMA20 > SMA50) but Kronos predicts DOWN with
    high confidence → SHORT (mean-reversion / exhaustion play).
    Trend bearish but Kronos predicts UP with high confidence → LONG.
    """

    def __init__(
        self,
        min_pct: float = 1.0,
        min_confidence: float = 0.4,
        kronos_runner: Optional[Callable] = None,
    ):
        super().__init__(StrategyType.KRONOS_DIVERGENCE)
        self.min_pct = min_pct
        self.min_confidence = min_confidence
        self._kronos_runner = kronos_runner

    def _forecast(self, df: pd.DataFrame, agent_reports: Optional[Dict[str, Any]]):
        cached = _extract_kronos_forecast(agent_reports)
        if cached is not None:
            return cached
        if self._kronos_runner is not None:
            try:
                out = self._kronos_runner(df)
                return out.to_dict() if hasattr(out, "to_dict") else out
            except Exception as exc:
                logger.warning("Kronos runner failed in %s: %s", self.strategy_type.value, exc)
                return None
        try:
            forecast = _get_shared_kronos_agent().predict(
                df, timeframe=df.attrs.get("timeframe")
            )
            return forecast.to_dict()
        except Exception as exc:
            logger.warning("Kronos prediction failed in %s: %s", self.strategy_type.value, exc)
            return None

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < 50:
            return None

        forecast = self._forecast(df, agent_reports)
        if not forecast:
            return None

        confidence = float(forecast.get("confidence", 0.0))
        magnitude = float(forecast.get("magnitude_pct", 0.0))
        params = _resolve_params(adaptive_params, df.attrs.get("timeframe"))
        min_conf = float(params.get("kronos_min_confidence", self.min_confidence))
        sl_mult = float(params["sl_atr_mult"])
        tp_mult = float(params["tp_atr_mult"])
        if confidence < min_conf:
            return None

        indicators = indicator_data or self._compute_indicators(df)
        current_price = float(df["Close"].iloc[-1])
        sma_20 = float(indicators.get("sma_20", current_price))
        sma_50 = float(indicators.get("sma_50", current_price))
        atr = float(indicators.get("atr", current_price * 0.02))

        uptrend = current_price > sma_20 > sma_50
        downtrend = current_price < sma_20 < sma_50

        # Bullish trend but Kronos says DOWN → contrarian SHORT
        if uptrend and magnitude <= -self.min_pct:
            stop_loss = current_price + sl_mult * atr
            take_profit = max(sma_20, current_price - max(tp_mult * atr, abs(magnitude) / 100.0 * current_price))
            if take_profit >= current_price:
                take_profit = current_price - tp_mult * atr
            rr = (current_price - take_profit) / max(stop_loss - current_price, 1e-9)
            strength = float(np.clip(0.4 + 0.6 * confidence, 0.0, 1.0))
            return Signal(
                direction="SHORT",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"Divergence SHORT: uptrend (price {current_price:.2f} > SMA20 {sma_20:.2f} > SMA50 {sma_50:.2f})"
                    f" but Kronos predicts {magnitude:+.2f}% (conf {confidence:.2f})."
                ),
                metadata={**indicators, "kronos": forecast},
            )

        # Bearish trend but Kronos says UP → contrarian LONG
        if downtrend and magnitude >= self.min_pct:
            stop_loss = current_price - sl_mult * atr
            take_profit = min(sma_20, current_price + max(tp_mult * atr, magnitude / 100.0 * current_price))
            if take_profit <= current_price:
                take_profit = current_price + tp_mult * atr
            rr = (take_profit - current_price) / max(current_price - stop_loss, 1e-9)
            strength = float(np.clip(0.4 + 0.6 * confidence, 0.0, 1.0))
            return Signal(
                direction="LONG",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"Divergence LONG: downtrend (price {current_price:.2f} < SMA20 {sma_20:.2f} < SMA50 {sma_50:.2f})"
                    f" but Kronos predicts {magnitude:+.2f}% (conf {confidence:.2f})."
                ),
                metadata={**indicators, "kronos": forecast},
            )

        return None


class MultiTimeframeKronosStrategy(BaseStrategy):
    """
    Run Kronos at several forecast horizons on the same OHLCV series and only
    trade when *all* horizons agree on direction with sufficient confidence.

    This is a pragmatic stand-in for true multi-timeframe data (which would
    require fetching 1h/4h/1d frames in lockstep and is awkward in
    backtesting). Different forecast horizons surface short-term and
    medium-term views from the same model and provide a useful agreement
    filter.
    """

    def __init__(
        self,
        horizons: Tuple[int, ...] = (6, 12, 24),
        min_pct: float = 0.5,
        min_confidence: float = 0.25,
        kronos_runner: Optional[Callable] = None,
    ):
        super().__init__(StrategyType.MULTI_TIMEFRAME_KRONOS)
        self.horizons = horizons
        self.min_pct = min_pct
        self.min_confidence = min_confidence
        self._kronos_runner = kronos_runner

    def _forecast(self, df: pd.DataFrame, horizon: int) -> Optional[Dict[str, Any]]:
        if self._kronos_runner is not None:
            try:
                out = self._kronos_runner(df, horizon)
                return out.to_dict() if hasattr(out, "to_dict") else out
            except Exception as exc:
                logger.warning("Kronos runner failed in MTF: %s", exc)
                return None
        try:
            forecast = _get_shared_kronos_agent().predict(
                df, horizon=horizon, timeframe=df.attrs.get("timeframe")
            )
            return forecast.to_dict()
        except Exception as exc:
            logger.warning("Kronos prediction failed in MTF: %s", exc)
            return None

    @staticmethod
    def _direction_from_pct(magnitude_pct: float, threshold: float) -> str:
        if magnitude_pct > threshold:
            return "UP"
        if magnitude_pct < -threshold:
            return "DOWN"
        return "NEUTRAL"

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < 50:
            return None

        forecasts: List[Dict[str, Any]] = []
        for h in self.horizons:
            f = self._forecast(df, h)
            if f is None:
                return None
            forecasts.append(f)

        directions = [self._direction_from_pct(float(f.get("magnitude_pct", 0.0)), self.min_pct) for f in forecasts]
        confidences = [float(f.get("confidence", 0.0)) for f in forecasts]

        params = _resolve_params(adaptive_params, df.attrs.get("timeframe"))
        min_conf = float(params.get("kronos_min_confidence", self.min_confidence))
        sl_mult = float(params["sl_atr_mult"])
        tp_mult = float(params["tp_atr_mult"])
        if any(c < min_conf for c in confidences):
            return None

        if all(d == "UP" for d in directions):
            agreed_direction = "LONG"
        elif all(d == "DOWN" for d in directions):
            agreed_direction = "SHORT"
        else:
            return None

        indicators = indicator_data or self._compute_indicators(df)
        current_price = float(df["Close"].iloc[-1])
        atr = float(indicators.get("atr", current_price * 0.02))

        avg_magnitude = float(np.mean([abs(float(f.get("magnitude_pct", 0.0))) for f in forecasts]))
        avg_confidence = float(np.mean(confidences))
        strength = float(np.clip(0.5 * avg_confidence + 0.5 * min(1.0, avg_magnitude / 5.0), 0.0, 1.0))

        if agreed_direction == "LONG":
            stop_loss = current_price - sl_mult * atr
            take_profit = current_price + max(tp_mult * atr, current_price * avg_magnitude / 100.0)
            rr = (take_profit - current_price) / max(current_price - stop_loss, 1e-9)
        else:
            stop_loss = current_price + sl_mult * atr
            take_profit = current_price - max(tp_mult * atr, current_price * avg_magnitude / 100.0)
            rr = (current_price - take_profit) / max(stop_loss - current_price, 1e-9)

        return Signal(
            direction=agreed_direction,
            strength=strength,
            strategy=self.strategy_type,
            symbol=df.attrs.get("symbol", "UNKNOWN"),
            timeframe=df.attrs.get("timeframe", "1h"),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr,
            reasoning=(
                f"Multi-Timeframe Kronos {agreed_direction}: horizons {list(self.horizons)} all "
                f"agree (avg mag {avg_magnitude:.2f}%, avg conf {avg_confidence:.2f})."
            ),
            metadata={
                **indicators,
                "kronos_horizons": list(self.horizons),
                "kronos_forecasts": forecasts,
            },
        )


class VWAPReversionStrategy(BaseStrategy):
    """
    VWAP mean-reversion strategy.

    LONG when price sits ≥ ``vwap_band_pct`` below VWAP AND RSI < ``rsi_oversold``
    (oversold at a discount to volume-weighted fair value).
    SHORT when price sits ≥ ``vwap_band_pct`` above VWAP AND RSI > ``rsi_overbought``
    (overbought at a premium).

    Stop: 1.5x ATR from entry. Target: VWAP (mean reversion goal).
    Works best on intraday frames (5m/15m/1h/4h) where VWAP carries information.
    """

    def __init__(
        self,
        vwap_band_pct: float = 0.02,   # 2% deviation from VWAP
        rsi_overbought: float = 60.0,
        rsi_oversold: float = 40.0,
        sl_atr_mult: float = 1.5,
    ):
        super().__init__(StrategyType.VWAP_REVERSION)
        self.vwap_band_pct = vwap_band_pct
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.sl_atr_mult = sl_atr_mult

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < 20 or "Volume" not in df.columns:
            return None

        indicators = indicator_data or self._compute_indicators(df)
        current_price = float(df["Close"].iloc[-1])
        rsi = float(indicators.get("rsi", 50.0))
        atr = float(indicators.get("atr", current_price * 0.02))
        vwap = _compute_vwap(df)
        if vwap <= 0:
            return None

        deviation = (current_price - vwap) / vwap  # positive = above VWAP
        sl_mult = self.sl_atr_mult

        # LONG: price is well below VWAP and RSI confirms oversold.
        if deviation <= -self.vwap_band_pct and rsi < self.rsi_oversold:
            stop_loss = current_price - sl_mult * atr
            take_profit = vwap
            if take_profit <= current_price:
                return None
            rr = (take_profit - current_price) / max(current_price - stop_loss, 1e-9)
            strength = float(
                np.clip(0.4 + 0.3 * min(1.0, abs(deviation) / 0.05)
                        + 0.3 * min(1.0, (self.rsi_oversold - rsi) / 20.0), 0.0, 1.0)
            )
            return Signal(
                direction="LONG",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"VWAP Reversion LONG: price {current_price:.4f} is "
                    f"{deviation * 100:.2f}% below VWAP {vwap:.4f}, RSI={rsi:.1f} < {self.rsi_oversold}."
                ),
                metadata={**indicators, "vwap": vwap, "vwap_deviation": deviation},
            )

        # SHORT: price well above VWAP and RSI confirms overbought.
        if deviation >= self.vwap_band_pct and rsi > self.rsi_overbought:
            stop_loss = current_price + sl_mult * atr
            take_profit = vwap
            if take_profit >= current_price:
                return None
            rr = (current_price - take_profit) / max(stop_loss - current_price, 1e-9)
            strength = float(
                np.clip(0.4 + 0.3 * min(1.0, deviation / 0.05)
                        + 0.3 * min(1.0, (rsi - self.rsi_overbought) / 20.0), 0.0, 1.0)
            )
            return Signal(
                direction="SHORT",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"VWAP Reversion SHORT: price {current_price:.4f} is "
                    f"{deviation * 100:.2f}% above VWAP {vwap:.4f}, RSI={rsi:.1f} > {self.rsi_overbought}."
                ),
                metadata={**indicators, "vwap": vwap, "vwap_deviation": deviation},
            )

        return None


class BollingerBandSqueezeStrategy(BaseStrategy):
    """
    Bollinger-Band squeeze breakout.

    Detect a low-vol regime (band width in the bottom 20th percentile over the
    last 50 bars), then wait for the squeeze to release and trade the breakout
    direction through the upper or lower band.

    Stop: opposite band at entry. Target: 2x the current band width from entry.
    """

    LOOKBACK = 50
    PERCENTILE = 20.0  # width must sit at or below this pct of the recent distribution

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(StrategyType.BB_SQUEEZE)
        self.period = period
        self.num_std = num_std

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < self.LOOKBACK + self.period:
            return None

        close = df["Close"].values
        indicators = indicator_data or self._compute_indicators(df)

        widths = _rolling_bb_widths(close, self.LOOKBACK, self.period, self.num_std)
        if len(widths) < 10:
            return None

        prev_width = float(widths[-2])
        curr_width = float(widths[-1])
        threshold = float(np.percentile(widths[:-1], self.PERCENTILE))

        # Squeeze: the *previous* bar had width in the bottom 20th percentile,
        # and the current bar shows expansion (width increasing).
        if prev_width > threshold or curr_width <= prev_width:
            return None

        upper, middle, lower, _ = _compute_bollinger_bands(
            close, self.period, self.num_std
        )
        current_price = float(close[-1])

        # LONG breakout: price punches above the upper band.
        if current_price > upper:
            stop_loss = lower
            if stop_loss >= current_price:
                return None
            band_width = upper - lower
            take_profit = current_price + 2.0 * band_width
            rr = (take_profit - current_price) / max(current_price - stop_loss, 1e-9)
            expansion_ratio = curr_width / prev_width if prev_width > 0 else 1.0
            strength = float(np.clip(0.5 + 0.25 * min(1.0, expansion_ratio - 1.0), 0.0, 1.0))
            return Signal(
                direction="LONG",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"BB Squeeze LONG breakout: prev width {prev_width:.4f} ≤ "
                    f"p{self.PERCENTILE:.0f}={threshold:.4f}, curr {curr_width:.4f} > prev. "
                    f"Price {current_price:.4f} > upper {upper:.4f}."
                ),
                metadata={**indicators, "bb_upper": upper, "bb_middle": middle,
                          "bb_lower": lower, "bb_width": curr_width,
                          "bb_prev_width": prev_width},
            )

        # SHORT breakout: price breaks below the lower band.
        if current_price < lower:
            stop_loss = upper
            if stop_loss <= current_price:
                return None
            band_width = upper - lower
            take_profit = current_price - 2.0 * band_width
            if take_profit <= 0:
                return None
            rr = (current_price - take_profit) / max(stop_loss - current_price, 1e-9)
            expansion_ratio = curr_width / prev_width if prev_width > 0 else 1.0
            strength = float(np.clip(0.5 + 0.25 * min(1.0, expansion_ratio - 1.0), 0.0, 1.0))
            return Signal(
                direction="SHORT",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"BB Squeeze SHORT breakout: prev width {prev_width:.4f} ≤ "
                    f"p{self.PERCENTILE:.0f}={threshold:.4f}, curr {curr_width:.4f} > prev. "
                    f"Price {current_price:.4f} < lower {lower:.4f}."
                ),
                metadata={**indicators, "bb_upper": upper, "bb_middle": middle,
                          "bb_lower": lower, "bb_width": curr_width,
                          "bb_prev_width": prev_width},
            )

        return None


class EMACrossoverStrategy(BaseStrategy):
    """
    Classical EMA(9)/EMA(21) crossover trend follower.

    LONG when fast EMA crosses above slow EMA, confirmed by a positive MACD
    histogram and volume above its 20-bar average. SHORT on the mirror image.

    Stop: 2x ATR; target: 3x ATR (so R:R = 1.5 by construction).
    """

    FAST_PERIOD = 9
    SLOW_PERIOD = 21

    def __init__(
        self,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0,
        volume_mult: float = 1.0,
    ):
        super().__init__(StrategyType.EMA_CROSSOVER)
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.volume_mult = volume_mult

    @staticmethod
    def _ema_series(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        out = np.zeros_like(data, dtype=float)
        if len(data) == 0:
            return out
        out[0] = float(data[0])
        for i in range(1, len(data)):
            out[i] = alpha * float(data[i]) + (1 - alpha) * out[i - 1]
        return out

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicator_data: Optional[Dict] = None,
        agent_reports: Optional[Dict[str, str]] = None,
        adaptive_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        if len(df) < self.SLOW_PERIOD + 5:
            return None

        close = df["Close"].values.astype(float)
        indicators = indicator_data or self._compute_indicators(df)

        fast = self._ema_series(close, self.FAST_PERIOD)
        slow = self._ema_series(close, self.SLOW_PERIOD)

        prev_diff = fast[-2] - slow[-2]
        curr_diff = fast[-1] - slow[-1]
        bullish_cross = prev_diff <= 0 and curr_diff > 0
        bearish_cross = prev_diff >= 0 and curr_diff < 0

        if not (bullish_cross or bearish_cross):
            return None

        macd_hist = float(indicators.get("macd_hist", 0.0))
        volume_ratio = float(indicators.get("volume_ratio", 1.0))
        atr = float(indicators.get("atr", float(close[-1]) * 0.02))
        current_price = float(close[-1])

        # Confirm with MACD direction AND volume > average.
        if volume_ratio < self.volume_mult:
            return None

        params = _resolve_params(adaptive_params, df.attrs.get("timeframe"))
        sl_mult = float(params.get("sl_atr_mult", self.sl_atr_mult))
        tp_mult = float(params.get("tp_atr_mult", self.tp_atr_mult))
        # Keep built-in 1.5 R:R floor even on tight-SL timeframes (5m/15m).
        if tp_mult < sl_mult * 1.5:
            tp_mult = sl_mult * 1.5

        if bullish_cross and macd_hist > 0:
            stop_loss = current_price - sl_mult * atr
            take_profit = current_price + tp_mult * atr
            rr = (take_profit - current_price) / max(current_price - stop_loss, 1e-9)
            strength = float(np.clip(0.5 + 0.25 * min(1.0, volume_ratio - 1.0)
                                     + 0.25 * min(1.0, abs(curr_diff) / atr if atr > 0 else 0.0),
                                     0.0, 1.0))
            return Signal(
                direction="LONG",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"EMA Crossover LONG: fast({self.FAST_PERIOD})={fast[-1]:.4f} crossed "
                    f"above slow({self.SLOW_PERIOD})={slow[-1]:.4f}, "
                    f"MACD hist={macd_hist:.4f}, vol ratio={volume_ratio:.2f}."
                ),
                metadata={**indicators, "ema_fast": float(fast[-1]),
                          "ema_slow": float(slow[-1])},
            )

        if bearish_cross and macd_hist < 0:
            stop_loss = current_price + sl_mult * atr
            take_profit = current_price - tp_mult * atr
            if take_profit <= 0:
                return None
            rr = (current_price - take_profit) / max(stop_loss - current_price, 1e-9)
            strength = float(np.clip(0.5 + 0.25 * min(1.0, volume_ratio - 1.0)
                                     + 0.25 * min(1.0, abs(curr_diff) / atr if atr > 0 else 0.0),
                                     0.0, 1.0))
            return Signal(
                direction="SHORT",
                strength=strength,
                strategy=self.strategy_type,
                symbol=df.attrs.get("symbol", "UNKNOWN"),
                timeframe=df.attrs.get("timeframe", "1h"),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr,
                reasoning=(
                    f"EMA Crossover SHORT: fast({self.FAST_PERIOD})={fast[-1]:.4f} crossed "
                    f"below slow({self.SLOW_PERIOD})={slow[-1]:.4f}, "
                    f"MACD hist={macd_hist:.4f}, vol ratio={volume_ratio:.2f}."
                ),
                metadata={**indicators, "ema_fast": float(fast[-1]),
                          "ema_slow": float(slow[-1])},
            )

        return None


# Strategy registry
STRATEGIES: Dict[StrategyType, BaseStrategy] = {
    StrategyType.MOMENTUM: MomentumStrategy(),
    StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
    StrategyType.BREAKOUT: BreakoutStrategy(),
    StrategyType.MULTI_FACTOR: MultiFactorStrategy(),
    StrategyType.KRONOS_MOMENTUM_CONFIRM: KronosMomentumConfirmStrategy(),
    StrategyType.KRONOS_DIVERGENCE: KronosDivergenceStrategy(),
    StrategyType.MULTI_TIMEFRAME_KRONOS: MultiTimeframeKronosStrategy(),
    StrategyType.VWAP_REVERSION: VWAPReversionStrategy(),
    StrategyType.BB_SQUEEZE: BollingerBandSqueezeStrategy(),
    StrategyType.EMA_CROSSOVER: EMACrossoverStrategy(),
}


def get_strategy(strategy_type: StrategyType) -> BaseStrategy:
    """Get a strategy instance by type."""
    return STRATEGIES[strategy_type]


def run_all_strategies(
    df: pd.DataFrame,
    enabled_strategies: Optional[List[StrategyType]] = None,
    indicator_data: Optional[Dict] = None,
    agent_reports: Optional[Dict[str, str]] = None,
    adaptive_params_by_strategy: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Signal]:
    """
    Run all enabled strategies and return their signals.

    ``adaptive_params_by_strategy`` — optional mapping
    ``{strategy_value: {param_name: value}}`` produced by the self-improver.
    When absent, each strategy uses its built-in defaults.
    """
    if enabled_strategies is None:
        enabled_strategies = list(StrategyType)

    signals = []
    for st in enabled_strategies:
        strategy = STRATEGIES.get(st)
        if strategy and strategy.enabled:
            try:
                params = None
                if adaptive_params_by_strategy is not None:
                    params = adaptive_params_by_strategy.get(st.value)
                signal = strategy.generate_signal(
                    df, indicator_data, agent_reports, adaptive_params=params
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in {st.value} strategy: {e}")

    return signals
