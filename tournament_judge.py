"""
Tournament judge — weekly auto-promotion.

Reads every tournament instance, computes 7-day Sharpe from each
instance's master portfolio snapshots, picks the winner, and writes a
new ``instances/champion-YYYYMMDD/profile.py`` that copies the winning
profile's settings under a new name + db_path.

Decision is logged to ``tournament_log.json`` (in the repo root) and to
``~/brain/quantagent-tournaments/<date>.json``.

Run weekly via cron / OpenClaw scheduler. ``--dry-run`` prints what would
happen without writing anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = REPO_ROOT / "instances"
TOURNAMENT_LOG = REPO_ROOT / "tournament_log.json"
BRAIN_DIR = Path.home() / "brain" / "quantagent-tournaments"


# ──────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────


def _equity_series(db_path: str, days: int = 7) -> List[float]:
    """Return master equity snapshot values for the last ``days``."""
    if not Path(db_path).exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT balance FROM portfolio_snapshots "
            "WHERE symbol = '__MASTER__' "
            "AND snapshot_time >= datetime('now', ?) "
            "ORDER BY snapshot_time ASC",
            (f"-{int(days)} day",),
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("equity_series failed for %s: %s", db_path, exc)
        return []
    return [float(r["balance"]) for r in rows]


def compute_sharpe(equity: List[float]) -> float:
    """Annualised Sharpe of period returns derived from snapshots.

    Returns 0.0 when the series is too short or has no variance.
    """
    if len(equity) < 3:
        return 0.0
    rets = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        cur = equity[i]
        if prev > 0:
            rets.append((cur - prev) / prev)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(252 * 24)


# ──────────────────────────────────────────────────────────────────────
# Judging
# ──────────────────────────────────────────────────────────────────────


def evaluate_instance(name: str, instances_dir: Path) -> Dict[str, Any]:
    """Return a metrics dict for one instance, or {} on failure."""
    state_path = instances_dir / name / "state.json"
    if not state_path.exists():
        return {"name": name, "error": "no state.json"}
    try:
        state = json.loads(state_path.read_text())
    except Exception as exc:
        return {"name": name, "error": f"bad state.json: {exc}"}
    db_path = state.get("db_path") or ""
    equity = _equity_series(db_path) if db_path else []
    sharpe = compute_sharpe(equity)
    return {
        "name": name,
        "db_path": db_path,
        "equity": state.get("equity"),
        "realized_pnl": state.get("realized_pnl"),
        "total_trades": state.get("total_trades"),
        "sharpe_7d": round(sharpe, 4),
        "snapshots": len(equity),
    }


def select_winner(metrics: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the highest 7d Sharpe with at least a couple data points."""
    candidates = [m for m in metrics if "error" not in m and m.get("snapshots", 0) >= 3]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.get("sharpe_7d") or float("-inf"))


# ──────────────────────────────────────────────────────────────────────
# Promotion
# ──────────────────────────────────────────────────────────────────────


def champion_name(today: Optional[datetime] = None) -> str:
    today = today or datetime.now(timezone.utc)
    return f"champion-{today.strftime('%Y%m%d')}"


def promote(
    winner_name: str,
    new_name: str,
    instances_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Create a new instance directory copying the winner's profile.

    The new profile keeps every setting except ``name`` and ``db_path``.
    Returns the path to the new ``profile.py``.
    """
    src = instances_dir / winner_name / "profile.py"
    if not src.exists():
        raise FileNotFoundError(f"winner profile not found: {src}")
    dst_dir = instances_dir / new_name
    dst = dst_dir / "profile.py"

    body = (
        f'"""Auto-promoted champion (winner of weekly tournament).'
        f' Copied from {winner_name}."""\n\n'
        f'from instances.{winner_name}.profile import PROFILE as _PARENT\n'
        f'from dataclasses import replace\n\n'
        f'PROFILE = replace(_PARENT, name="{new_name}", db_path="paper_trades_{new_name}.db")\n'
    )

    if dry_run:
        logger.info("[dry-run] would write %s", dst)
        return dst

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_path = dst_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")
    dst.write_text(body)
    return dst


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────


def append_log(record: Dict[str, Any], path: Path = TOURNAMENT_LOG) -> None:
    log: List[Dict[str, Any]] = []
    if path.exists():
        try:
            log = json.loads(path.read_text())
            if not isinstance(log, list):
                log = []
        except Exception:
            log = []
    log.append(record)
    path.write_text(json.dumps(log, indent=2, default=str))


def write_brain_record(record: Dict[str, Any], brain_dir: Path = BRAIN_DIR) -> Optional[Path]:
    try:
        brain_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        path = brain_dir / f"{ts}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        return path
    except Exception as exc:
        logger.warning("failed to write brain record: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────


def run(
    instances_dir: Path = INSTANCES_DIR,
    dry_run: bool = False,
    today: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Top-level run: evaluate all instances, pick winner, promote, log.

    Returns the decision record (also written to disk).
    """
    if not instances_dir.exists():
        raise FileNotFoundError(f"instances dir missing: {instances_dir}")

    names = sorted(
        d.name for d in instances_dir.iterdir()
        if d.is_dir() and (d / "profile.py").exists() and not d.name.startswith("champion-")
    )
    metrics = [evaluate_instance(n, instances_dir) for n in names]
    winner = select_winner(metrics)

    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluated": metrics,
        "winner": None,
        "promoted_to": None,
        "dry_run": dry_run,
    }

    if winner is None:
        logger.warning("tournament judge: no eligible winner.")
    else:
        new_name = champion_name(today)
        record["winner"] = winner
        record["promoted_to"] = new_name
        try:
            path = promote(winner["name"], new_name, instances_dir, dry_run=dry_run)
            record["champion_profile_path"] = str(path)
        except Exception as exc:
            logger.exception("promotion failed: %s", exc)
            record["error"] = str(exc)

    # Read module-level paths at call time so tests can monkeypatch them.
    append_log(record, TOURNAMENT_LOG)
    write_brain_record(record, BRAIN_DIR)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="QuantAgent weekly tournament judge")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    record = run(dry_run=args.dry_run)
    print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
