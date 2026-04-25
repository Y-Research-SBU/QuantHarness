"""
Tournament instance runner.

Wraps :mod:`run_continuous` so each tournament instance writes to its own
SQLite DB and applies its profile-specific filters/multipliers.

Usage::

    python3 run_instance.py --profile baseline
    python3 run_instance.py --profile crypto_aggro --max-cycles 5

The profile name maps to ``instances/<name>/profile.py`` and that module
must expose a ``PROFILE`` instance of :class:`instances.profile.Profile`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from market_config import MARKETS
from run_continuous import ContinuousRunner, _MarketSchedule
from scanner import MarketScanner

logger = logging.getLogger(__name__)


# Repo root (where instances/ lives).
REPO_ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = REPO_ROOT / "instances"


def load_profile(name: str):
    """Import ``instances.<name>.profile`` and return its ``PROFILE``.

    Raises :class:`FileNotFoundError` if the profile directory or
    ``profile.py`` is missing, and :class:`AttributeError` if the module
    does not export ``PROFILE``.
    """
    profile_path = INSTANCES_DIR / name / "profile.py"
    if not profile_path.exists():
        raise FileNotFoundError(f"profile not found: {profile_path}")

    module_name = f"instances.{name}.profile"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Attempt a fresh import after invalidating caches (helpful in tests
        # that create instance dirs at runtime).
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)

    if not hasattr(module, "PROFILE"):
        raise AttributeError(
            f"profile module {module_name} does not define PROFILE"
        )
    return module.PROFILE


def state_file_path(profile_name: str) -> Path:
    """Return the path to ``instances/<name>/state.json``."""
    return INSTANCES_DIR / profile_name / "state.json"


def write_state_file(profile_name: str, state: dict) -> None:
    """Write a heartbeat file at ``instances/<name>/state.json``.

    Non-fatal — any error is logged and swallowed.
    """
    try:
        state_dir = INSTANCES_DIR / profile_name
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "state.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        tmp.replace(path)
    except Exception as exc:  # pragma: no cover — best-effort
        logger.warning("failed to write state.json for %s: %s", profile_name, exc)


class HeartbeatRunner(ContinuousRunner):
    """ContinuousRunner that updates instances/<name>/state.json each cycle."""

    profile_name: str = ""
    db_path_str: str = ""
    started_at_iso: str = ""

    def _run_one(self, symbol: str, sched: _MarketSchedule, force: bool = False) -> None:  # type: ignore[override]
        super()._run_one(symbol, sched, force=force)
        try:
            self._write_heartbeat()
        except Exception as exc:  # pragma: no cover
            logger.debug("heartbeat write failed: %s", exc)

    def _write_heartbeat(self) -> None:
        if not self.profile_name:
            return
        try:
            engine = self.scanner.engine
            master = engine.get_master_portfolio() if hasattr(engine, "get_master_portfolio") else None
            open_positions = engine.get_open_positions() if hasattr(engine, "get_open_positions") else []
            try:
                exposure = engine.get_total_exposure() if hasattr(engine, "get_total_exposure") else 0.0
            except Exception:
                exposure = 0.0
            equity = float(master.get("current_balance") or 0.0) + float(exposure or 0.0) if master else 0.0
            state = {
                "name": self.profile_name,
                "pid": os.getpid(),
                "started_at": self.started_at_iso,
                "db_path": self.db_path_str,
                "last_heartbeat": datetime.utcnow().isoformat(),
                "equity": equity,
                "realized_pnl": float(master.get("total_pnl") or 0.0) if master else 0.0,
                "open_positions": len(open_positions),
                "total_trades": int(master.get("total_trades") or 0) if master else 0,
                "last_scan": {
                    sym: {
                        "total_cycles": s.total_cycles,
                        "total_signals": s.total_signals,
                        "total_trades": s.total_trades,
                        "last_regime": s.last_regime,
                        "last_error": s.last_error,
                    }
                    for sym, s in self._schedules.items()
                },
            }
            write_state_file(self.profile_name, state)
        except Exception as exc:  # pragma: no cover
            logger.debug("heartbeat snapshot failed: %s", exc)


def resolve_symbols(profile, override: Optional[List[str]] = None) -> List[str]:
    """Compute the symbol list this instance should scan.

    ``--symbols`` from the CLI takes precedence; otherwise the profile's
    universe / cell_overrides / asset_categories filters apply.
    """
    if override:
        return list(override)

    all_symbols = list(MARKETS.keys())
    filtered = profile.filter_symbols(all_symbols)

    if profile.asset_categories:
        wanted = set(profile.asset_categories)
        filtered = [
            s for s in filtered
            if MARKETS[s].category.value in wanted
        ]

    return filtered


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantAgent tournament instance runner")
    parser.add_argument("--profile", required=True, help="Profile name under instances/")
    parser.add_argument("--symbols", nargs="*", default=None, help="Override symbols")
    parser.add_argument(
        "--summary-every-minutes",
        type=int,
        default=30,
        help="Minutes between performance summaries.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after N total scan cycles.",
    )
    parser.add_argument(
        "--no-kronos",
        action="store_true",
        help="Disable Kronos forecasting.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan for every market and exit.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Override the profile's db_path.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    db_path = args.db or profile.db_path

    logger.info(
        "Starting instance '%s' \u2014 db=%s, regime_gate=%s, size_x=%.2f, entry_x=%.2f",
        profile.name, db_path, profile.regime_filter_enabled,
        profile.position_size_multiplier, profile.entry_threshold_multiplier,
    )

    scanner = MarketScanner(
        db_path=db_path,
        use_kronos=not args.no_kronos,
        profile=profile,
    )
    symbols = resolve_symbols(profile, args.symbols)
    started_at_iso = datetime.utcnow().isoformat()
    runner = HeartbeatRunner(
        scanner=scanner,
        symbols=symbols,
        summary_every_seconds=args.summary_every_minutes * 60,
    )
    runner.profile_name = profile.name
    runner.db_path_str = db_path
    runner.started_at_iso = started_at_iso
    runner.install_signal_handlers()

    write_state_file(profile.name, {
        "name": profile.name,
        "pid": os.getpid(),
        "started_at": started_at_iso,
        "db_path": db_path,
        "symbols": symbols,
        "profile": {
            "universe": profile.universe,
            "asset_categories": profile.asset_categories,
            "strategies": profile.strategies,
            "cell_overrides": [list(c) for c in profile.cell_overrides],
            "position_size_multiplier": profile.position_size_multiplier,
            "entry_threshold_multiplier": profile.entry_threshold_multiplier,
            "regime_filter_enabled": profile.regime_filter_enabled,
            "use_l0_whitelist": profile.use_l0_whitelist,
            "use_l1_evolution": profile.use_l1_evolution,
        },
    })

    if args.once:
        runner.run_once()
    else:
        runner.run(max_cycles=args.max_cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
