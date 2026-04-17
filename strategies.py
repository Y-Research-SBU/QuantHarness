"""
Pluggable strategy framework for QuantAgent.

Strategies:
1. Momentum — Follow trends on 4hr/daily. Enter on pullbacks in trend direction.
2. Mean Reversion — Fade RSI extremes (>70 short, <30 long) on 1hr timeframe.
3. Breakout — Pattern agent detects formation → enter on breakout with volume confirmation.
4. Multi-Factor — Weighted scoring from all 5 agents. Only trade when 4/5 agree.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from market_config import StrategyType

logger = logging.getLogger(__name__)


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
    ) -> Optional[Signal]:
        """
        Generate a trading signal from market data and optional agent reports.
        
        Args:
            df: OHLCV DataFrame
            indicator_data: Pre-computed indicators (RSI, MACD, etc.)
            agent_reports: Reports from indicator/pattern/trend/decision agents
        
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
        
        # Trend direction: price above both SMAs = uptrend
        uptrend = current_price > sma_20 and sma_20 > sma_50
        downtrend = current_price < sma_20 and sma_20 < sma_50
        
        # Pullback detection: RSI in moderate zone
        bullish_pullback = uptrend and 40 <= rsi <= 60 and macd_hist > 0
        bearish_pullback = downtrend and 40 <= rsi <= 60 and macd_hist < 0
        
        if bullish_pullback:
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 3 * atr
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
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 3 * atr
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
        
        # Overbought → SHORT
        if rsi > self.RSI_OVERBOUGHT and stoch_k > 80:
            stop_loss = current_price + 1.5 * atr
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
        elif rsi < self.RSI_OVERSOLD and stoch_k < 20:
            stop_loss = current_price - 1.5 * atr
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
    ) -> Optional[Signal]:
        if len(df) < 30:
            return None
        
        indicators = indicator_data or self._compute_indicators(df)
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        current_price = float(close[-1])
        
        atr = indicators.get("atr", current_price * 0.02)
        
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
            stop_loss = recent_low - 0.5 * atr
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
            stop_loss = recent_high + 0.5 * atr
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
    ) -> Optional[Signal]:
        if len(df) < 30:
            return None
        
        indicators = indicator_data or self._compute_indicators(df)
        close = df["Close"].values
        current_price = float(close[-1])
        atr = indicators.get("atr", current_price * 0.02)
        
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
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 3 * atr
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
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 3 * atr
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


# Strategy registry
STRATEGIES: Dict[StrategyType, BaseStrategy] = {
    StrategyType.MOMENTUM: MomentumStrategy(),
    StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
    StrategyType.BREAKOUT: BreakoutStrategy(),
    StrategyType.MULTI_FACTOR: MultiFactorStrategy(),
}


def get_strategy(strategy_type: StrategyType) -> BaseStrategy:
    """Get a strategy instance by type."""
    return STRATEGIES[strategy_type]


def run_all_strategies(
    df: pd.DataFrame,
    enabled_strategies: Optional[List[StrategyType]] = None,
    indicator_data: Optional[Dict] = None,
    agent_reports: Optional[Dict[str, str]] = None,
) -> List[Signal]:
    """
    Run all enabled strategies and return their signals.
    """
    if enabled_strategies is None:
        enabled_strategies = list(StrategyType)
    
    signals = []
    for st in enabled_strategies:
        strategy = STRATEGIES.get(st)
        if strategy and strategy.enabled:
            try:
                signal = strategy.generate_signal(df, indicator_data, agent_reports)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in {st.value} strategy: {e}")
    
    return signals
