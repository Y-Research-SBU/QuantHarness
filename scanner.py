"""
Automated market scanner for QuantAgent.
Runs on a configurable schedule, analyzing all markets and executing paper trades.
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from data_fetcher import fetch_market_data, prepare_kline_dict
from kronos_agent import KronosForecastAgent
from market_config import MARKETS, MarketConfig, StrategyType
from paper_trading import PaperTradingEngine
from position_sizing import calculate_position_size, calculate_stop_loss
from performance_tracker import calculate_performance
from self_improver import SelfImprover, WEIGHT_DISABLED
from strategies import Signal, run_all_strategies

logger = logging.getLogger(__name__)


# Minimum meta-model probability for a signal to pass L3 quality gating.
SIGNAL_QUALITY_THRESHOLD = 0.35


class MarketScanner:
    """
    Automated market scanner that:
    1. Fetches data for all markets
    2. Runs enabled strategies
    3. Executes paper trades for valid signals
    4. Checks stop-losses and take-profits
    5. Takes portfolio snapshots
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        use_agents: bool = False,
        agent_config: Optional[Dict] = None,
        use_kronos: bool = True,
        kronos_agent: Optional[KronosForecastAgent] = None,
        use_self_improvement: bool = True,
        self_improver: Optional[SelfImprover] = None,
    ):
        self.engine = PaperTradingEngine(db_path=db_path)
        self.use_agents = use_agents
        self.agent_config = agent_config or {}
        self._trading_graph = None
        self.use_kronos = use_kronos
        self._kronos_agent = kronos_agent

        # ── Self-improvement (L1–L5) — additive; failures must never crash trading.
        self.use_self_improvement = use_self_improvement
        self.self_improver: Optional[SelfImprover] = None
        if use_self_improvement:
            try:
                self.self_improver = self_improver or SelfImprover(db_path=db_path)
            except Exception as exc:
                logger.warning("SelfImprover init failed — self-improvement disabled: %s", exc)
                self.self_improver = None
                self.use_self_improvement = False

        # Cache for OHLCV frames collected during a scan cycle; used by the
        # improvement cycle at the end so we can feed real bars into L2/L4.
        self._cycle_bars: Dict[str, pd.DataFrame] = {}
        # Per-symbol latest detected regime in the current cycle.
        self._cycle_regimes: Dict[str, str] = {}
    
    def _get_trading_graph(self):
        """Lazy-load the trading graph (expensive to initialize)."""
        if self._trading_graph is None and self.use_agents:
            try:
                from trading_graph import TradingGraph
                config = {
                    "agent_llm_provider": "anthropic",
                    "agent_llm_model": "claude-haiku-4-5-20251001",
                    "graph_llm_provider": "anthropic",
                    "graph_llm_model": "claude-sonnet-4-20250514",
                    "agent_llm_temperature": 0.1,
                    "graph_llm_temperature": 0.1,
                }
                config.update(self.agent_config)
                self._trading_graph = TradingGraph(config=config)
            except Exception as e:
                logger.error(f"Failed to initialize trading graph: {e}")
        return self._trading_graph

    def _get_kronos_agent(self) -> Optional[KronosForecastAgent]:
        """Lazy-construct a shared Kronos forecaster for the scan loop."""
        if not self.use_kronos:
            return None
        if self._kronos_agent is None:
            try:
                self._kronos_agent = KronosForecastAgent()
            except Exception as exc:
                logger.error(f"Failed to initialize Kronos agent: {exc}")
                self.use_kronos = False
                return None
        return self._kronos_agent

    def _run_kronos_forecast(
        self, symbol: str, timeframe: str, df: pd.DataFrame
    ) -> Optional[Dict]:
        """Run Kronos against the latest bars; never raise.

        Also logs the prediction to the self-improvement DB (L5) and applies
        the learned per-market confidence adjustment so downstream strategies
        see calibrated confidence. The log row's ID is attached to the returned
        forecast dict so trade close handlers can evaluate the outcome.
        """
        agent = self._get_kronos_agent()
        if agent is None:
            return None
        try:
            forecast = agent.predict(df, timeframe=timeframe)
            data = forecast.to_dict()
        except Exception as exc:
            logger.warning(f"Kronos forecast failed for {symbol} ({timeframe}): {exc}")
            return None

        # L5: per-market confidence calibration (never raise).
        if self.self_improver is not None:
            try:
                adj = self.self_improver.get_kronos_confidence_adjustment(symbol)
                if adj != 1.0:
                    data["raw_confidence"] = data.get("confidence")
                    data["confidence"] = float(min(1.0, max(0.0, data.get("confidence", 0.0) * adj)))
                    data["confidence_adjustment"] = adj
            except Exception as exc:
                logger.debug("Kronos confidence adjustment failed for %s: %s", symbol, exc)

            # L5: log the prediction for later outcome evaluation.
            try:
                pred_id = self.self_improver.log_kronos_prediction(
                    symbol=symbol,
                    timeframe=timeframe or "",
                    predicted_direction=str(data.get("direction", "NEUTRAL")),
                    predicted_magnitude=float(data.get("magnitude_pct", 0.0) or 0.0),
                    confidence=float(data.get("confidence", 0.0) or 0.0),
                    horizon=int(data.get("horizon", 0) or 0),
                    predicted_price=float(data.get("predicted_close") or 0.0) or None,
                )
                data["prediction_id"] = pred_id
            except Exception as exc:
                logger.debug("log_kronos_prediction failed for %s: %s", symbol, exc)
        return data
    
    def scan_market(
        self,
        symbol: str,
        config: MarketConfig,
    ) -> List[Signal]:
        """
        Scan a single market across all its timeframes and strategies.

        Also feeds the latest OHLCV frame to the self-improver for L4 regime
        detection and caches the bars so L2 optimization can run on them at
        the end of the cycle.

        Returns:
            List of trading signals
        """
        all_signals = []

        # Pick the longest-horizon frame we scan as the canonical cycle bars
        # for this symbol (typically "4h"). Store the first non-empty frame.
        cycle_df: Optional[pd.DataFrame] = None

        # Pull adaptive params (L2) and current regime (L4) once per symbol so
        # we can tune strategies + attach context to every signal.
        adaptive_by_strategy: Dict[str, Dict[str, float]] = {}
        current_regime: Optional[str] = None
        if self.self_improver is not None:
            try:
                for st in config.enabled_strategies:
                    adaptive_by_strategy[st.value] = self.self_improver.get_adaptive_params(
                        st.value, symbol
                    )
            except Exception as exc:
                logger.debug("get_adaptive_params failed for %s: %s", symbol, exc)

        for timeframe in config.timeframes:
            try:
                df = fetch_market_data(symbol, timeframe)
                if df.empty:
                    logger.warning(f"No data for {symbol} ({timeframe})")
                    continue

                # Attach metadata
                df.attrs["symbol"] = symbol
                df.attrs["timeframe"] = timeframe

                if cycle_df is None:
                    cycle_df = df

                # L4: detect + log regime once per symbol (use first good frame).
                if current_regime is None and self.self_improver is not None:
                    try:
                        current_regime = self.self_improver.log_regime(
                            symbol, df, timeframe=timeframe
                        )
                        self._cycle_regimes[symbol] = current_regime
                    except Exception as exc:
                        logger.debug("regime detection failed for %s: %s", symbol, exc)

                # Run agent analysis if enabled
                agent_reports: Optional[Dict] = None
                if self.use_agents:
                    agent_reports = self._run_agent_analysis(symbol, timeframe, df)

                # Always include Kronos forecast if enabled (cheap, no LLM call).
                if self.use_kronos:
                    kronos_data = self._run_kronos_forecast(symbol, timeframe, df)
                    if kronos_data is not None:
                        agent_reports = dict(agent_reports or {})
                        agent_reports["kronos_forecast_data"] = kronos_data

                # Run strategies
                signals = run_all_strategies(
                    df=df,
                    enabled_strategies=config.enabled_strategies,
                    agent_reports=agent_reports,
                    adaptive_params_by_strategy=adaptive_by_strategy or None,
                )

                for sig in signals:
                    sig.symbol = symbol
                    sig.timeframe = timeframe
                    # Attach regime + category to metadata for L3 feature extraction
                    # and downstream L4 affinity scoring / outcome analysis.
                    sig.metadata = dict(sig.metadata or {})
                    if current_regime:
                        sig.metadata.setdefault("regime", current_regime)
                    sig.metadata.setdefault("category", getattr(config.category, "value", str(config.category)))
                    # If Kronos predicted, carry the prediction ID so close_trade
                    # can mark the outcome.
                    if agent_reports and isinstance(agent_reports.get("kronos_forecast_data"), dict):
                        pred_id = agent_reports["kronos_forecast_data"].get("prediction_id")
                        if pred_id is not None:
                            sig.metadata.setdefault("kronos_prediction_id", pred_id)
                            sig.metadata.setdefault(
                                "kronos_confidence",
                                agent_reports["kronos_forecast_data"].get("confidence"),
                            )
                            sig.metadata.setdefault(
                                "kronos_entry_price",
                                float(df["Close"].iloc[-1]),
                            )

                all_signals.extend(signals)

            except Exception as e:
                logger.error(f"Error scanning {symbol} ({timeframe}): {e}")

        if cycle_df is not None:
            self._cycle_bars[symbol] = cycle_df

        return all_signals
    
    def _run_agent_analysis(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> Optional[Dict[str, str]]:
        """Run the full agent pipeline on market data."""
        graph = self._get_trading_graph()
        if not graph:
            return None
        
        try:
            kline_dict = prepare_kline_dict(df.tail(45))
            
            # Import static utilities for image generation
            import static_util
            p_image = static_util.generate_kline_image(kline_dict)
            t_image = static_util.generate_trend_image(kline_dict)
            
            initial_state = {
                "kline_data": kline_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": timeframe,
                "stock_name": symbol,
                "pattern_image": p_image.get("pattern_image"),
                "trend_image": t_image.get("trend_image"),
            }
            
            final_state = graph.graph.invoke(initial_state)
            
            reports = {
                "indicator_report": final_state.get("indicator_report", ""),
                "pattern_report": final_state.get("pattern_report", ""),
                "trend_report": final_state.get("trend_report", ""),
                "decision": final_state.get("final_trade_decision", ""),
            }
            
            # Log API costs (rough estimates)
            self.engine.log_api_cost(
                symbol=symbol,
                timeframe=timeframe,
                model=self.agent_config.get("agent_llm_model", "claude-haiku-4-5"),
                prompt_tokens=2000,
                completion_tokens=1000,
                operation="agent_analysis",
            )
            
            return reports
            
        except Exception as e:
            logger.error(f"Agent analysis failed for {symbol} ({timeframe}): {e}")
            return None
    
    def execute_signals(
        self,
        signals: List[Signal],
    ) -> List[int]:
        """
        Execute paper trades for the strongest signals against the master portfolio.

        Applies (before sizing):
          • L1: strategy weight — disabled strategies are skipped; weighted
            strength is used for ranking.
          • L4: regime affinity — the signal's detected regime multiplies the
            weight.
          • L3: signal-quality meta-model — if a model has been trained, we
            skip signals whose predicted probability of profit is below
            SIGNAL_QUALITY_THRESHOLD.

        Signals are sorted by weighted strength (strongest first) across ALL
        markets; we take up to (MAX_POSITIONS - currently_open) trades.
        """
        if not signals:
            return []

        master = self.engine.get_master_portfolio()
        master_balance = float(master["initial_balance"]) or self.engine.MASTER_INITIAL_BALANCE

        # Pre-compute strategy weights once (L1) and affinity once (L4).
        strategy_weights: Dict[str, float] = {}
        if self.self_improver is not None:
            try:
                strategy_weights = self.self_improver.get_strategy_weights()
            except Exception as exc:
                logger.debug("get_strategy_weights failed: %s", exc)

        def _weighted_strength(signal: Signal) -> float:
            """Combine base strength with L1 weight and L4 regime affinity."""
            base = float(signal.strength or 0.0)
            strat = signal.strategy.value if hasattr(signal.strategy, "value") else str(signal.strategy)
            weight = strategy_weights.get(strat, 1.0) if strategy_weights else 1.0
            regime = None
            if isinstance(signal.metadata, dict):
                regime = signal.metadata.get("regime")
            if self.self_improver is not None and regime:
                try:
                    weight = self.self_improver.get_regime_adjusted_weight(strat, regime)
                except Exception:
                    pass
            return base * float(weight)

        # Drop disabled strategies up-front (L1) + trend-conflict filter.
        filtered: List[Signal] = []
        for s in signals:
            strat = s.strategy.value if hasattr(s.strategy, "value") else str(s.strategy)
            if strategy_weights.get(strat, 1.0) == WEIGHT_DISABLED:
                logger.debug("Skipping disabled strategy signal: %s %s", strat, s.symbol)
                continue

            # Trend-conflict filter: never SHORT a trending-up market or
            # LONG a trending-down market. These counter-trend trades have
            # a terrible hit rate (the INJ-USD problem).
            regime = (s.metadata or {}).get("regime") if isinstance(s.metadata, dict) else None
            if regime == "trending_up" and s.direction == "SHORT":
                logger.info(
                    "Trend filter: skipping SHORT %s — regime is trending_up", s.symbol
                )
                continue
            if regime == "trending_down" and s.direction == "LONG":
                logger.info(
                    "Trend filter: skipping LONG %s — regime is trending_down", s.symbol
                )
                continue

            filtered.append(s)

        # Rank by weighted strength, strongest first.
        ranked = sorted(filtered, key=_weighted_strength, reverse=True)

        current_open = len(self.engine.get_open_positions())
        slots_remaining = max(0, self.engine.MAX_POSITIONS - current_open)
        if slots_remaining == 0:
            return []

        trade_ids: List[int] = []
        for signal in ranked:
            if len(trade_ids) >= slots_remaining:
                break

            # L3: signal-quality meta-model. Neutral 0.5 if no model trained,
            # so we keep trading while the model bootstraps.
            if self.self_improver is not None:
                try:
                    features = dict(signal.metadata or {})
                    features.setdefault("strength", float(signal.strength or 0.0))
                    strat = signal.strategy.value if hasattr(signal.strategy, "value") else str(signal.strategy)
                    category = features.get("category") or ""
                    quality = self.self_improver.predict_signal_quality(
                        features, strategy=strat, category=str(category)
                    )
                    if quality < SIGNAL_QUALITY_THRESHOLD:
                        logger.debug(
                            "Signal below L3 quality threshold (%.2f < %.2f): %s %s",
                            quality, SIGNAL_QUALITY_THRESHOLD, strat, signal.symbol,
                        )
                        continue
                    # Persist the quality score so downstream reporting can see it.
                    if isinstance(signal.metadata, dict):
                        signal.metadata.setdefault("signal_quality", quality)
                except Exception as exc:
                    logger.debug("predict_signal_quality failed: %s", exc)

            pos = calculate_position_size(
                portfolio_balance=master_balance,
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss,
                direction=signal.direction,
                signal_strength=signal.strength,
                max_risk_pct=0.02,
                risk_reward_ratio=signal.risk_reward_ratio,
                min_position_size=self.engine.MIN_POSITION_SIZE,
                max_position_size=self.engine.MAX_POSITION_SIZE,
            )

            if pos.position_size_usd <= 0:
                continue

            trade_id = self.engine.execute_trade(signal, pos)
            if trade_id:
                trade_ids.append(trade_id)

        return trade_ids
    
    def check_all_stops(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check all open positions for stop-loss/take-profit."""
        return self.engine.check_stops(current_prices)
    
    def run_scan_cycle(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run a full scan cycle:
        1. Fetch current prices and check stops
        2. Scan all markets for signals (including regime detection + Kronos logging)
        3. Execute best signals (weighted + quality-gated)
        4. Take snapshots
        5. Run the self-improvement cycle (gated by closed-trade count)

        Returns:
            Summary of the cycle
        """
        if symbols is None:
            symbols = list(MARKETS.keys())

        cycle_start = datetime.utcnow()
        results = {
            "cycle_time": cycle_start.isoformat(),
            "markets_scanned": 0,
            "signals_found": 0,
            "trades_opened": 0,
            "stops_triggered": 0,
            "signals": [],
            "improvement": None,
            "regimes": {},
        }

        # Fresh cycle state for bars + regimes collected during this run.
        self._cycle_bars = {}
        self._cycle_regimes = {}

        # Reset daily P&L if needed
        self.engine.reset_daily_pnl()

        # Get current prices and check stops
        current_prices = {}
        for symbol in symbols:
            try:
                df = fetch_market_data(symbol, "1d", bars=2)
                if not df.empty:
                    current_prices[symbol] = float(df["Close"].iloc[-1])
            except Exception:
                pass

        closed = self.check_all_stops(current_prices)
        results["stops_triggered"] = len(closed)

        # Mark-to-market: refresh unrealised P&L on still-open positions so the
        # dashboard and analytics see live numbers (not 0 until close).
        try:
            mtm = self.engine.mark_to_market(current_prices)
            results["unrealized_pnl"] = mtm.get("total_unrealized_pnl", 0.0)
            results["positions_marked"] = mtm.get("positions_marked", 0)
        except Exception as exc:
            logger.warning("mark_to_market failed (non-fatal): %s", exc)

        # Scan markets
        all_signals: List[Signal] = []
        for symbol in symbols:
            config = MARKETS.get(symbol)
            if not config:
                continue

            signals = self.scan_market(symbol, config)
            all_signals.extend(signals)
            results["markets_scanned"] += 1

        results["signals_found"] = len(all_signals)
        results["signals"] = [
            {
                "symbol": s.symbol,
                "direction": s.direction,
                "strategy": s.strategy.value,
                "strength": s.strength,
                "entry_price": s.entry_price,
                "regime": (s.metadata or {}).get("regime") if isinstance(s.metadata, dict) else None,
            }
            for s in all_signals
        ]
        results["regimes"] = dict(self._cycle_regimes)

        # Execute best signals (weighted by L1 + regime via execute_signals).
        trade_ids = self.execute_signals(all_signals)
        results["trades_opened"] = len(trade_ids)

        # Take snapshots: master for overall equity curve, per-symbol for per-market P&L curves.
        self.engine.take_snapshot(self.engine.MASTER_SYMBOL)
        for symbol in symbols:
            self.engine.take_snapshot(symbol)

        # ── Self-improvement cycle (L1/L2/L3/L5 gated by closed-trade count). ─
        if self.self_improver is not None:
            try:
                results["improvement"] = self.self_improver.run_improvement_cycle(
                    symbol_bars=self._cycle_bars,
                )
            except Exception as exc:
                logger.warning("self-improvement cycle failed (non-fatal): %s", exc)
                results["improvement"] = {"error": str(exc)}

        logger.info(
            f"Scan cycle complete: {results['markets_scanned']} markets, "
            f"{results['signals_found']} signals, {results['trades_opened']} trades, "
            f"{results['stops_triggered']} stops"
        )

        return results


def run_scanner_loop(
    interval_seconds: int = 14400,  # 4 hours default
    db_path: Optional[str] = None,
    use_agents: bool = False,
):
    """
    Run the scanner in a loop.
    
    Args:
        interval_seconds: Seconds between scan cycles
        db_path: Path to SQLite database
        use_agents: Whether to use LLM agent analysis
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    scanner = MarketScanner(db_path=db_path, use_agents=use_agents)
    
    logger.info(f"Starting scanner loop (interval: {interval_seconds}s, agents: {use_agents})")
    
    while True:
        try:
            results = scanner.run_scan_cycle()
            logger.info(f"Cycle results: {json.dumps(results, indent=2)}")
        except Exception as e:
            logger.error(f"Scan cycle error: {e}", exc_info=True)
        
        logger.info(f"Sleeping {interval_seconds}s until next cycle...")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantAgent Market Scanner")
    parser.add_argument("--interval", type=int, default=14400, help="Scan interval in seconds")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--agents", action="store_true", help="Enable LLM agent analysis")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    
    args = parser.parse_args()
    
    if args.once:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        scanner = MarketScanner(db_path=args.db, use_agents=args.agents)
        results = scanner.run_scan_cycle()
        print(json.dumps(results, indent=2))
    else:
        run_scanner_loop(
            interval_seconds=args.interval,
            db_path=args.db,
            use_agents=args.agents,
        )
