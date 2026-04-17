"""
Scanner controller: manages a background thread that runs MarketScanner on an interval.
Provides start/stop/status for the web UI.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from scanner import MarketScanner

logger = logging.getLogger(__name__)


class ScannerController:
    """Thread-safe controller for starting/stopping a background market scanner."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._scanner: Optional[MarketScanner] = None
        self._last_cycle: Optional[Dict[str, Any]] = None
        self._last_cycle_time: Optional[str] = None
        self._cycles_run: int = 0
        self._started_at: Optional[str] = None
        self._interval_seconds: int = 14400
        self._use_agents: bool = False
        self._last_error: Optional[str] = None

    def start(
        self,
        interval_seconds: int = 14400,
        use_agents: bool = False,
    ) -> Dict[str, Any]:
        """Start the scanner loop in a background thread."""
        with self._lock:
            if self.is_running():
                return {"started": False, "reason": "already_running"}

            self._interval_seconds = int(interval_seconds)
            self._use_agents = bool(use_agents)
            self._stop_event.clear()
            self._scanner = MarketScanner(db_path=self.db_path, use_agents=self._use_agents)
            self._started_at = datetime.utcnow().isoformat()
            self._cycles_run = 0
            self._last_error = None

            self._thread = threading.Thread(
                target=self._run_loop,
                name="ScannerLoop",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                f"Scanner started (interval={self._interval_seconds}s, agents={self._use_agents})"
            )
            return {
                "started": True,
                "interval_seconds": self._interval_seconds,
                "use_agents": self._use_agents,
            }

    def stop(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Signal the scanner to stop and wait briefly for the thread to exit."""
        with self._lock:
            if not self.is_running():
                return {"stopped": False, "reason": "not_running"}

            self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)

        logger.info("Scanner stopped")
        return {"stopped": True}

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> Dict[str, Any]:
        """Run a single scan cycle synchronously (does not start the loop)."""
        scanner = MarketScanner(db_path=self.db_path, use_agents=False)
        results = scanner.run_scan_cycle()
        self._last_cycle = results
        self._last_cycle_time = results.get("cycle_time")
        return results

    def status(self) -> Dict[str, Any]:
        """Get current scanner state and a portfolio summary."""
        scanner_state = {
            "running": self.is_running(),
            "started_at": self._started_at,
            "interval_seconds": self._interval_seconds,
            "use_agents": self._use_agents,
            "cycles_run": self._cycles_run,
            "last_cycle_time": self._last_cycle_time,
            "last_cycle": self._last_cycle,
            "last_error": self._last_error,
        }

        # Build a lightweight portfolio + P&L summary (no heavy agent work).
        try:
            from paper_trading import PaperTradingEngine

            engine = PaperTradingEngine(db_path=self.db_path)
            portfolios = engine.get_all_portfolios()
            open_positions = engine.get_open_positions()

            total_balance = sum(p["current_balance"] for p in portfolios)
            total_initial = sum(p["initial_balance"] for p in portfolios)
            total_pnl = sum(p["total_pnl"] for p in portfolios)
            total_trades = sum(p["total_trades"] for p in portfolios)
            total_wins = sum(p["winning_trades"] for p in portfolios)
            win_rate = total_wins / total_trades if total_trades > 0 else 0.0

            summary = {
                "total_balance": total_balance,
                "total_initial": total_initial,
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "winning_trades": total_wins,
                "win_rate": win_rate,
                "open_positions": len(open_positions),
                "portfolios": len(portfolios),
            }
        except Exception as e:
            logger.warning(f"Failed to build portfolio summary: {e}")
            summary = {"error": str(e)}

        return {"scanner": scanner_state, "summary": summary}

    def _run_loop(self):
        """Thread target: run scan cycles until stop is signaled."""
        assert self._scanner is not None
        while not self._stop_event.is_set():
            try:
                results = self._scanner.run_scan_cycle()
                self._last_cycle = results
                self._last_cycle_time = results.get("cycle_time")
                self._cycles_run += 1
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Scan cycle error: {e}", exc_info=True)

            # Wait in small slices so stop() is responsive.
            waited = 0.0
            slice_sec = 1.0
            while waited < self._interval_seconds and not self._stop_event.is_set():
                time.sleep(slice_sec)
                waited += slice_sec


# Module-level singleton — one controller per process.
_controller: Optional[ScannerController] = None


def get_controller(db_path: Optional[str] = None) -> ScannerController:
    """Return the shared ScannerController singleton."""
    global _controller
    if _controller is None:
        _controller = ScannerController(db_path=db_path)
    return _controller


def reset_controller():
    """Reset the singleton (for tests)."""
    global _controller
    if _controller is not None and _controller.is_running():
        _controller.stop(timeout=2.0)
    _controller = None
