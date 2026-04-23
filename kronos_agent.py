"""
Kronos forecast agent for QuantAgent.

Wraps the Kronos foundation model (https://github.com/shiyu-coder/Kronos) so it
can be plugged into the existing LangGraph trading pipeline as a 5th node.

If the Kronos model cannot be loaded (no network, missing weights, torch
incompatibility, etc.) the agent transparently falls back to a deterministic
statistical forecaster so the rest of the pipeline still produces signals.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kronos source path setup
# ---------------------------------------------------------------------------

_KRONOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos_forecast")
if os.path.isdir(_KRONOS_DIR) and _KRONOS_DIR not in sys.path:
    sys.path.insert(0, _KRONOS_DIR)


# ---------------------------------------------------------------------------
# Forecast result types
# ---------------------------------------------------------------------------


@dataclass
class KronosForecast:
    """Structured forecast returned by :class:`KronosForecastAgent`."""

    direction: str                       # "UP", "DOWN", "NEUTRAL"
    magnitude_pct: float                 # signed predicted percent change vs current close
    confidence: float                    # 0.0..1.0
    predicted_close: float               # last predicted close
    predicted_high: float
    predicted_low: float
    horizon: int                         # number of candles forecast
    last_close: float                    # close price at the end of the input data
    source: str                          # "kronos" or "fallback"
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Human/LLM-readable summary."""
        return (
            f"Kronos forecast ({self.source}, horizon={self.horizon}): "
            f"{self.direction} with predicted change {self.magnitude_pct:+.2f}% "
            f"(confidence {self.confidence:.2f}). "
            f"Last close={self.last_close:.4f}, predicted close={self.predicted_close:.4f}. "
            f"{self.reasoning}"
        )


# ---------------------------------------------------------------------------
# KronosForecastAgent
# ---------------------------------------------------------------------------


_DEFAULT_HORIZON_BY_TF = {
    "1m": 30,
    "5m": 24,
    "15m": 24,
    "30m": 24,
    "1h": 24,
    "4h": 12,
    "1d": 5,
    "1w": 4,
}


class KronosForecastAgent:
    """
    Wraps a Kronos predictor and produces a structured forecast from an OHLCV
    DataFrame.

    The agent is **lazy-loaded** and **thread-safe**: the underlying torch model
    is only constructed on the first ``predict`` call, and a process-wide
    singleton is reused so that we don't pay model-load cost per scan.
    """

    _LOAD_LOCK = threading.Lock()
    _SHARED_PREDICTOR: Any = None  # KronosPredictor or None
    _LOAD_FAILED = False
    _LOAD_ERROR: Optional[str] = None

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        max_context: int = 512,
        device: Optional[str] = None,
        default_horizon: int = 24,
        sample_count: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        enable_kronos: bool = True,
    ) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_context = max_context
        self.device = device
        self.default_horizon = default_horizon
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_p = top_p
        # Honour the QUANTAGENT_DISABLE_KRONOS env var so backtests and CI can
        # force the fast statistical fallback without code changes.
        if os.environ.get("QUANTAGENT_DISABLE_KRONOS", "").lower() in ("1", "true", "yes"):
            enable_kronos = False
        self.enable_kronos = enable_kronos

    # ------------------------------------------------------------------
    # Predictor loading
    # ------------------------------------------------------------------

    def _get_predictor(self) -> Optional[Any]:
        """Return a shared KronosPredictor, loading it on first use."""
        if not self.enable_kronos:
            return None
        if KronosForecastAgent._SHARED_PREDICTOR is not None:
            return KronosForecastAgent._SHARED_PREDICTOR
        if KronosForecastAgent._LOAD_FAILED:
            return None
        with KronosForecastAgent._LOAD_LOCK:
            if KronosForecastAgent._SHARED_PREDICTOR is not None:
                return KronosForecastAgent._SHARED_PREDICTOR
            if KronosForecastAgent._LOAD_FAILED:
                return None
            try:
                from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
            except Exception as exc:  # pragma: no cover - import-time failure
                KronosForecastAgent._LOAD_FAILED = True
                KronosForecastAgent._LOAD_ERROR = f"import failed: {exc}"
                logger.warning("Kronos import failed (%s); using fallback forecaster.", exc)
                return None
            try:
                tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
                model = Kronos.from_pretrained(self.model_name)
                predictor = KronosPredictor(
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    max_context=self.max_context,
                )
                KronosForecastAgent._SHARED_PREDICTOR = predictor
                logger.info("Kronos predictor loaded (%s).", self.model_name)
                return predictor
            except Exception as exc:
                KronosForecastAgent._LOAD_FAILED = True
                KronosForecastAgent._LOAD_ERROR = f"weights load failed: {exc}"
                logger.warning(
                    "Kronos weight load failed (%s); using fallback forecaster.", exc
                )
                return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        horizon: Optional[int] = None,
        timeframe: Optional[str] = None,
    ) -> KronosForecast:
        """
        Run a forecast on the provided OHLCV ``DataFrame``.

        Args:
            df: DataFrame with columns ``Open, High, Low, Close`` (and ideally
                ``Volume`` and ``Datetime``). The frame is expected to be ordered
                from oldest to newest.
            horizon: How many candles to predict. If ``None``, picks a sensible
                default for the timeframe (24 for crypto/intraday, 5 for daily).
            timeframe: Optional timeframe hint used both for default-horizon
                selection and for synthesising future timestamps.

        Returns:
            :class:`KronosForecast` — never raises; falls back to a statistical
            forecast on any error.
        """
        if df is None or len(df) == 0:
            return self._neutral_forecast(0.0, horizon or self.default_horizon, "empty input")

        tf = timeframe or df.attrs.get("timeframe")
        if horizon is None:
            horizon = _DEFAULT_HORIZON_BY_TF.get(tf or "", self.default_horizon)

        last_close = float(df["Close"].iloc[-1])

        predictor = self._get_predictor()
        if predictor is not None:
            try:
                return self._predict_with_kronos(predictor, df, horizon, tf, last_close)
            except Exception as exc:
                logger.warning("Kronos predict failed (%s); falling back.", exc)
                # fall through to fallback forecaster

        return self._predict_fallback(df, horizon, last_close)

    # ------------------------------------------------------------------
    # Kronos path
    # ------------------------------------------------------------------

    def _predict_with_kronos(
        self,
        predictor: Any,
        df: pd.DataFrame,
        horizon: int,
        timeframe: Optional[str],
        last_close: float,
    ) -> KronosForecast:
        x_df, x_ts = self._prepare_input(df)
        y_ts = self._build_future_timestamps(x_ts, horizon, timeframe)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=horizon,
            T=self.temperature,
            top_p=self.top_p,
            sample_count=max(1, self.sample_count),
            verbose=False,
        )

        predicted_close = float(pred_df["close"].iloc[-1])
        predicted_high = float(pred_df["high"].max())
        predicted_low = float(pred_df["low"].min())
        magnitude_pct = (predicted_close - last_close) / last_close * 100.0 if last_close else 0.0

        # Confidence: based on path stability — lower std-dev of predicted closes
        # relative to the move size means a more confident directional call.
        path = pred_df["close"].values.astype(float)
        path_std = float(np.std(path)) if len(path) > 1 else 0.0
        path_range = float(np.ptp(path)) if len(path) > 1 else 0.0
        consistency = 1.0 - min(1.0, path_std / (abs(predicted_close - last_close) + 1e-9))
        confidence = float(np.clip(0.4 + 0.6 * consistency, 0.0, 1.0))

        direction = self._direction_from_pct(magnitude_pct)

        return KronosForecast(
            direction=direction,
            magnitude_pct=magnitude_pct,
            confidence=confidence,
            predicted_close=predicted_close,
            predicted_high=predicted_high,
            predicted_low=predicted_low,
            horizon=horizon,
            last_close=last_close,
            source="kronos",
            reasoning=(
                f"Kronos path predicts close {predicted_close:.4f} from {last_close:.4f}; "
                f"path range {path_range:.4f}, std {path_std:.4f}."
            ),
            metadata={
                "predicted_path": path.tolist(),
                "model": self.model_name,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "sample_count": self.sample_count,
            },
        )

    def _prepare_input(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Build the lower-cased OHLCV frame Kronos expects."""
        cols = {c.lower(): c for c in df.columns}
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(f"Missing required OHLC columns: {missing}")

        x = pd.DataFrame({c: df[cols[c]].astype(float).values for c in required})
        if "volume" in cols:
            x["volume"] = df[cols["volume"]].astype(float).values
        else:
            x["volume"] = 0.0
        x["amount"] = x["volume"] * x[required].mean(axis=1)

        # Truncate to the predictor's max context.
        if len(x) > self.max_context:
            x = x.iloc[-self.max_context :].reset_index(drop=True)

        # Build / pull timestamps.
        if "Datetime" in df.columns:
            ts = pd.to_datetime(df["Datetime"].values)
        elif "datetime" in df.columns:
            ts = pd.to_datetime(df["datetime"].values)
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        else:
            ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="h")
        ts = pd.Series(pd.to_datetime(ts))[-len(x) :].reset_index(drop=True)
        return x, ts

    @staticmethod
    def _build_future_timestamps(
        x_ts: pd.Series, horizon: int, timeframe: Optional[str]
    ) -> pd.Series:
        if len(x_ts) >= 2:
            step = (x_ts.iloc[-1] - x_ts.iloc[-2]) or pd.Timedelta(hours=1)
        else:
            step = _timeframe_to_timedelta(timeframe)
        last = pd.Timestamp(x_ts.iloc[-1])
        future = [last + step * (i + 1) for i in range(horizon)]
        return pd.Series(future)

    # ------------------------------------------------------------------
    # Fallback path
    # ------------------------------------------------------------------

    def _predict_fallback(
        self, df: pd.DataFrame, horizon: int, last_close: float
    ) -> KronosForecast:
        """
        Deterministic statistical forecaster used when Kronos is unavailable.

        Combines a short-window log-return drift with realised volatility to
        produce a directional view that follows the same data shape as the
        Kronos path. This keeps the rest of the pipeline working in CI / offline
        environments.
        """
        close = df["Close"].astype(float).values
        if len(close) < 5:
            return self._neutral_forecast(last_close, horizon, "insufficient history")

        log_ret = np.diff(np.log(close + 1e-12))
        window = min(30, len(log_ret))
        recent = log_ret[-window:]
        mu = float(np.mean(recent))
        sigma = float(np.std(recent)) or 1e-6

        # Linear trend slope (log-price) over the window for stability.
        idx = np.arange(window)
        slope = float(np.polyfit(idx, np.log(close[-window:] + 1e-12), 1)[0]) if window >= 2 else 0.0
        drift = 0.5 * mu + 0.5 * slope

        # Project a path forward using the drift; high/low envelope from sigma.
        path = last_close * np.exp(drift * np.arange(1, horizon + 1))
        predicted_close = float(path[-1])
        predicted_high = float(np.max(path) * np.exp(sigma))
        predicted_low = float(np.min(path) * np.exp(-sigma))

        magnitude_pct = (predicted_close - last_close) / last_close * 100.0 if last_close else 0.0

        # Confidence: magnitude divided by per-step vol — clamp to [0.1, 0.85].
        snr = abs(drift) / sigma if sigma else 0.0
        confidence = float(np.clip(0.1 + snr * 0.5, 0.1, 0.85))

        direction = self._direction_from_pct(magnitude_pct)

        return KronosForecast(
            direction=direction,
            magnitude_pct=magnitude_pct,
            confidence=confidence,
            predicted_close=predicted_close,
            predicted_high=predicted_high,
            predicted_low=predicted_low,
            horizon=horizon,
            last_close=last_close,
            source="fallback",
            reasoning=(
                f"Fallback drift={drift:.5f}, sigma={sigma:.5f} (window={window}); "
                f"projected {magnitude_pct:+.2f}%."
            ),
            metadata={
                "drift": drift,
                "sigma": sigma,
                "window": window,
                "predicted_path": path.tolist(),
                "load_error": KronosForecastAgent._LOAD_ERROR,
            },
        )

    def _neutral_forecast(self, last_close: float, horizon: int, reason: str) -> KronosForecast:
        return KronosForecast(
            direction="NEUTRAL",
            magnitude_pct=0.0,
            confidence=0.0,
            predicted_close=last_close,
            predicted_high=last_close,
            predicted_low=last_close,
            horizon=horizon,
            last_close=last_close,
            source="fallback",
            reasoning=f"Neutral forecast: {reason}.",
            metadata={"load_error": KronosForecastAgent._LOAD_ERROR},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    UP_THRESHOLD_PCT = 0.25       # any predicted move beyond this counts as directional
    STRONG_THRESHOLD_PCT = 1.0

    @classmethod
    def _direction_from_pct(cls, magnitude_pct: float) -> str:
        if magnitude_pct > cls.UP_THRESHOLD_PCT:
            return "UP"
        if magnitude_pct < -cls.UP_THRESHOLD_PCT:
            return "DOWN"
        return "NEUTRAL"


def _timeframe_to_timedelta(timeframe: Optional[str]) -> pd.Timedelta:
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
        "1w": pd.Timedelta(weeks=1),
    }
    return mapping.get(timeframe or "", pd.Timedelta(hours=1))


# ---------------------------------------------------------------------------
# LangGraph node factory
# ---------------------------------------------------------------------------


def create_kronos_agent(
    agent: Optional[KronosForecastAgent] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a LangGraph node that runs Kronos and writes a forecast into state.

    The node consumes ``kline_data`` (the dict produced by ``prepare_kline_dict``)
    plus ``time_frame`` from the graph state and writes:

    - ``kronos_forecast``: human-readable summary string used by the decision
      agent prompt.
    - ``kronos_forecast_data``: the structured :class:`KronosForecast` as a
      dict for any downstream consumer.
    """
    forecaster = agent or KronosForecastAgent()

    def kronos_node(state: Dict[str, Any]) -> Dict[str, Any]:
        kline = state.get("kline_data") or {}
        timeframe = state.get("time_frame")
        try:
            df = _kline_dict_to_df(kline)
            forecast = forecaster.predict(df, timeframe=timeframe)
        except Exception as exc:  # belt-and-braces — never crash the graph
            logger.error("Kronos node failed: %s", exc)
            last_close = float(kline.get("Close", [0.0])[-1]) if kline.get("Close") else 0.0
            forecast = KronosForecast(
                direction="NEUTRAL",
                magnitude_pct=0.0,
                confidence=0.0,
                predicted_close=last_close,
                predicted_high=last_close,
                predicted_low=last_close,
                horizon=0,
                last_close=last_close,
                source="fallback",
                reasoning=f"Kronos node error: {exc}",
            )

        return {
            "kronos_forecast": forecast.summary(),
            "kronos_forecast_data": forecast.to_dict(),
        }

    return kronos_node


def _kline_dict_to_df(kline: Dict[str, Any]) -> pd.DataFrame:
    """Convert ``prepare_kline_dict`` output back into an OHLCV DataFrame."""
    if not kline:
        return pd.DataFrame()
    cols = {}
    for key in ("Datetime", "Open", "High", "Low", "Close", "Volume"):
        if key in kline:
            cols[key] = kline[key]
    df = pd.DataFrame(cols)
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df
