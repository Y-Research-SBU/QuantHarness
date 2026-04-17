"""
Automated market scanner for QuantAgent.
Runs on a configurable schedule, analyzing all markets and executing paper trades.
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from data_fetcher import fetch_market_data, prepare_kline_dict
from market_config import MARKETS, MarketConfig, StrategyType
from paper_trading import PaperTradingEngine
from position_sizing import calculate_position_size, calculate_stop_loss
from performance_tracker import calculate_performance
from strategies import Signal, run_all_strategies

logger = logging.getLogger(__name__)


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
    ):
        self.engine = PaperTradingEngine(db_path=db_path)
        self.use_agents = use_agents
        self.agent_config = agent_config or {}
        self._trading_graph = None
    
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
    
    def scan_market(
        self,
        symbol: str,
        config: MarketConfig,
    ) -> List[Signal]:
        """
        Scan a single market across all its timeframes and strategies.
        
        Returns:
            List of trading signals
        """
        all_signals = []
        
        for timeframe in config.timeframes:
            try:
                df = fetch_market_data(symbol, timeframe)
                if df.empty:
                    logger.warning(f"No data for {symbol} ({timeframe})")
                    continue
                
                # Attach metadata
                df.attrs["symbol"] = symbol
                df.attrs["timeframe"] = timeframe
                
                # Run agent analysis if enabled
                agent_reports = None
                if self.use_agents:
                    agent_reports = self._run_agent_analysis(symbol, timeframe, df)
                
                # Run strategies
                signals = run_all_strategies(
                    df=df,
                    enabled_strategies=config.enabled_strategies,
                    agent_reports=agent_reports,
                )
                
                for sig in signals:
                    sig.symbol = symbol
                    sig.timeframe = timeframe
                
                all_signals.extend(signals)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol} ({timeframe}): {e}")
        
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
        Execute paper trades for valid signals.
        
        Returns:
            List of trade IDs
        """
        trade_ids = []
        
        for signal in signals:
            portfolio = self.engine.get_portfolio(signal.symbol)
            if not portfolio:
                continue
            
            # Calculate position size
            pos = calculate_position_size(
                portfolio_balance=portfolio["current_balance"],
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss,
                direction=signal.direction,
                max_risk_pct=0.02,
                risk_reward_ratio=signal.risk_reward_ratio,
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
        2. Scan all markets for signals
        3. Execute best signals
        4. Take snapshots
        
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
        }
        
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
            }
            for s in all_signals
        ]
        
        # Execute best signals (strongest first)
        all_signals.sort(key=lambda s: s.strength, reverse=True)
        trade_ids = self.execute_signals(all_signals)
        results["trades_opened"] = len(trade_ids)
        
        # Take snapshots
        for symbol in symbols:
            self.engine.take_snapshot(symbol)
        
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
