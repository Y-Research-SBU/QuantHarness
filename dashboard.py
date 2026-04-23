"""
Real-time web monitoring dashboard for the QuantAgent trading system.

A standalone Flask app (separate from web_interface.py) that renders a
single-page dashboard with:
  - Overview panel: total balance, total P&L, open positions, markets
  - Market grid: per-market cards with sparklines
  - Strategy performance table
  - Recent trade log
  - Scanner status (last scan, cycles, errors)
  - Backtest results summary from backtest_results/*.json

Data source: paper_trades.db (SQLite).

Run with:
    python3 dashboard.py

Then open http://127.0.0.1:5001/.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request as flask_request, send_from_directory


logger = logging.getLogger(__name__)


# Default paths (can be overridden by tests via create_app()).
DEFAULT_DB_PATH = "paper_trades.db"
DEFAULT_BACKTEST_DIR = "backtest_results"


# Static category mapping for market metadata (avoids importing the full
# market_config module if it isn't available — but we prefer the real one).
_FALLBACK_MARKETS = {
    "BTC-USD": {"display_name": "Bitcoin", "category": "crypto"},
    "ETH-USD": {"display_name": "Ethereum", "category": "crypto"},
    "SOL-USD": {"display_name": "Solana", "category": "crypto"},
    "SPY": {"display_name": "S&P 500 ETF", "category": "stocks"},
    "QQQ": {"display_name": "Nasdaq 100 ETF", "category": "stocks"},
    "AAPL": {"display_name": "Apple Inc.", "category": "stocks"},
    "TSLA": {"display_name": "Tesla Inc.", "category": "stocks"},
    "NVDA": {"display_name": "NVIDIA Corp.", "category": "stocks"},
    "GC=F": {"display_name": "Gold Futures", "category": "commodities"},
    "CL=F": {"display_name": "Crude Oil Futures", "category": "commodities"},
    "EURUSD=X": {"display_name": "EUR/USD", "category": "forex"},
    "GBPUSD=X": {"display_name": "GBP/USD", "category": "forex"},
}


def _load_markets_meta() -> Dict[str, Dict[str, Any]]:
    """Load per-symbol metadata. Prefers market_config.MARKETS when available."""
    try:
        from market_config import MARKETS  # type: ignore

        out: Dict[str, Dict[str, Any]] = {}
        for symbol, cfg in MARKETS.items():
            category = getattr(cfg.category, "value", str(cfg.category))
            out[symbol] = {
                "display_name": cfg.display_name,
                "category": category,
                "timeframes": list(cfg.timeframes),
                "scan_interval_hours": cfg.scan_interval_hours,
            }
        return out
    except Exception:
        return {k: dict(v) for k, v in _FALLBACK_MARKETS.items()}


# ───────────────────────── Database helpers ─────────────────────────


def _connect(db_path: str) -> sqlite3.Connection:
    """Open a read-only connection to the SQLite DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _query_portfolios(db_path: str) -> List[Dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM portfolios ORDER BY symbol").fetchall()
        return _rows_to_dicts(rows)


def _query_trades(
    db_path: str,
    limit: int = 50,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    with _connect(db_path) as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = ? ORDER BY entry_time DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return _rows_to_dicts(rows)


def _query_strategy_performance(db_path: str) -> List[Dict[str, Any]]:
    """Aggregate strategy_performance across all symbols/timeframes per strategy."""
    if not Path(db_path).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM strategy_performance").fetchall()
        raw = _rows_to_dicts(rows)

    agg: Dict[str, Dict[str, Any]] = {}
    for r in raw:
        s = r["strategy"]
        bucket = agg.setdefault(
            s,
            {
                "strategy": s,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "_win_pnls": [],
                "_loss_pnls": [],
            },
        )
        bucket["total_trades"] += int(r.get("total_trades") or 0)
        bucket["winning_trades"] += int(r.get("winning_trades") or 0)
        bucket["losing_trades"] += int(r.get("losing_trades") or 0)
        bucket["total_pnl"] += float(r.get("total_pnl") or 0.0)
        # avg_win / avg_loss are per-row averages; track weighted sums via counts
        w = int(r.get("winning_trades") or 0)
        l = int(r.get("losing_trades") or 0)
        if w > 0:
            bucket["_win_pnls"].append((float(r.get("avg_win") or 0.0), w))
        if l > 0:
            bucket["_loss_pnls"].append((float(r.get("avg_loss") or 0.0), l))

    out: List[Dict[str, Any]] = []
    for s, bucket in agg.items():
        total = bucket["total_trades"]
        win_rate = (bucket["winning_trades"] / total) if total > 0 else 0.0

        total_w = sum(c for _, c in bucket["_win_pnls"])
        total_l = sum(c for _, c in bucket["_loss_pnls"])
        avg_win = (
            sum(v * c for v, c in bucket["_win_pnls"]) / total_w if total_w > 0 else 0.0
        )
        avg_loss = (
            sum(v * c for v, c in bucket["_loss_pnls"]) / total_l if total_l > 0 else 0.0
        )

        out.append(
            {
                "strategy": s,
                "total_trades": total,
                "winning_trades": bucket["winning_trades"],
                "losing_trades": bucket["losing_trades"],
                "total_pnl": bucket["total_pnl"],
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
            }
        )

    out.sort(key=lambda x: x["total_pnl"], reverse=True)
    return out


def _query_snapshots_for_sparkline(
    db_path: str, symbol: str, limit: int = 50
) -> List[float]:
    if not Path(db_path).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT balance FROM portfolio_snapshots WHERE symbol = ? "
            "ORDER BY snapshot_time DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
    # Reverse so the series runs oldest → newest (left to right).
    return list(reversed([float(r["balance"]) for r in rows]))


def _query_latest_signal(db_path: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Return the latest trade row for a symbol (serves as the most recent signal)."""
    if not Path(db_path).exists():
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT symbol, strategy, direction, entry_time, entry_price "
            "FROM trades WHERE symbol = ? ORDER BY entry_time DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    return dict(row) if row else None


def _query_scanner_status(db_path: str) -> Dict[str, Any]:
    """Derive scanner activity from portfolio_snapshots + recent trades."""
    if not Path(db_path).exists():
        return {
            "last_scan_time": None,
            "recent_scans": [],
            "total_snapshots": 0,
            "snapshots_last_hour": 0,
            "signals_last_hour": 0,
            "markets_scanned_last_hour": [],
        }

    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(snapshot_time) AS last FROM portfolio_snapshots"
        ).fetchone()
        last_scan_time = row["last"] if row else None

        total = conn.execute(
            "SELECT COUNT(*) AS n FROM portfolio_snapshots"
        ).fetchone()["n"]

        # snapshots in the last hour (approximate: 1h window from now)
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        snapshots_last_hour = conn.execute(
            "SELECT COUNT(*) AS n FROM portfolio_snapshots WHERE snapshot_time >= ?",
            (one_hour_ago,),
        ).fetchone()["n"]

        markets_last_hour = [
            r["symbol"]
            for r in conn.execute(
                "SELECT DISTINCT symbol FROM portfolio_snapshots "
                "WHERE snapshot_time >= ? ORDER BY symbol",
                (one_hour_ago,),
            ).fetchall()
        ]

        # recent distinct market/time pairs
        recent_rows = conn.execute(
            "SELECT symbol, MAX(snapshot_time) AS ts, COUNT(*) AS cnt "
            "FROM portfolio_snapshots GROUP BY symbol ORDER BY ts DESC LIMIT 20"
        ).fetchall()
        recent_scans = [
            {"symbol": r["symbol"], "last_scan": r["ts"], "scan_count": r["cnt"]}
            for r in recent_rows
        ]

        # signals = trades in the last hour
        signals_last_hour = conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE entry_time >= ?",
            (one_hour_ago.replace(" ", "T"),),
        ).fetchone()["n"]

    return {
        "last_scan_time": last_scan_time,
        "recent_scans": recent_scans,
        "total_snapshots": total,
        "snapshots_last_hour": snapshots_last_hour,
        "signals_last_hour": signals_last_hour,
        "markets_scanned_last_hour": markets_last_hour,
    }


# ───────────────────────── Aggregation helpers ─────────────────────────


def build_overview(db_path: str) -> Dict[str, Any]:
    portfolios = _query_portfolios(db_path)
    total_balance = sum(float(p["current_balance"]) for p in portfolios)
    total_initial = sum(float(p["initial_balance"]) for p in portfolios)
    total_pnl = total_balance - total_initial
    pnl_pct = (total_pnl / total_initial * 100.0) if total_initial > 0 else 0.0

    open_positions = _query_trades(db_path, limit=1000, status="OPEN")

    markets_meta = _load_markets_meta()

    scanner = _query_scanner_status(db_path)

    return {
        "total_balance": total_balance,
        "total_initial": total_initial,
        "total_pnl": total_pnl,
        "total_pnl_pct": pnl_pct,
        "open_positions": len(open_positions),
        "markets_tracked": len(markets_meta),
        "portfolios": len(portfolios),
        "last_scan_time": scanner.get("last_scan_time"),
        "signals_last_hour": scanner.get("signals_last_hour", 0),
        "is_profitable": total_pnl >= 0,
    }


def build_market_grid(db_path: str) -> List[Dict[str, Any]]:
    """Per-market card data: balance, P&L, drawdown, sparkline, latest signal."""
    portfolios = _query_portfolios(db_path)
    markets_meta = _load_markets_meta()

    by_symbol = {p["symbol"]: p for p in portfolios}

    cards: List[Dict[str, Any]] = []
    for symbol, meta in markets_meta.items():
        pf = by_symbol.get(symbol)
        initial_balance = float(pf["initial_balance"]) if pf else 10000.0
        current_balance = float(pf["current_balance"]) if pf else initial_balance
        total_pnl = float(pf["total_pnl"]) if pf else 0.0
        total_trades = int(pf["total_trades"]) if pf else 0
        max_dd = float(pf["max_drawdown"]) if pf else 0.0
        circuit_breaker = bool(int(pf["is_circuit_breaker_active"])) if pf else False
        pnl_pct = (total_pnl / initial_balance * 100.0) if initial_balance > 0 else 0.0

        sparkline = _query_snapshots_for_sparkline(db_path, symbol, limit=50)
        if not sparkline:
            sparkline = [initial_balance]

        latest_signal = _query_latest_signal(db_path, symbol)

        cards.append(
            {
                "symbol": symbol,
                "display_name": meta.get("display_name", symbol),
                "category": meta.get("category", "unknown"),
                "current_balance": current_balance,
                "initial_balance": initial_balance,
                "total_pnl": total_pnl,
                "pnl_pct": pnl_pct,
                "total_trades": total_trades,
                "max_drawdown_pct": max_dd * 100.0,
                "circuit_breaker": circuit_breaker,
                "sparkline": sparkline,
                "latest_signal": latest_signal,
            }
        )

    cards.sort(key=lambda c: (-c["total_pnl"], c["symbol"]))
    return cards


def build_strategy_table(db_path: str) -> List[Dict[str, Any]]:
    rows = _query_strategy_performance(db_path)
    if not rows:
        return []
    best = max(rows, key=lambda r: r["total_pnl"])["strategy"]
    worst = min(rows, key=lambda r: r["total_pnl"])["strategy"]
    for r in rows:
        r["is_best"] = r["strategy"] == best
        r["is_worst"] = r["strategy"] == worst and best != worst
    return rows


def build_trade_log(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    trades = _query_trades(db_path, limit=limit)
    slim = []
    for t in trades:
        slim.append(
            {
                "id": t["id"],
                "symbol": t["symbol"],
                "timeframe": t["timeframe"],
                "strategy": t["strategy"],
                "direction": t["direction"],
                "entry_price": t["entry_price"],
                "exit_price": t.get("exit_price"),
                "pnl": t.get("pnl") or 0.0,
                "pnl_pct": t.get("pnl_pct") or 0.0,
                "status": t["status"],
                "entry_time": t["entry_time"],
                "exit_time": t.get("exit_time"),
            }
        )
    return slim


def build_backtest_summary(backtest_dir: str) -> List[Dict[str, Any]]:
    """Summarize backtest JSON files in ``backtest_dir``.

    Expected shape (from backtest.py):
        {"generated_at": ..., "results": [{"symbol", "strategy", "timeframe",
         "total_return_pct", "sharpe_ratio", "max_drawdown_pct", ...}, ...]}
    """
    path = Path(backtest_dir)
    if not path.exists() or not path.is_dir():
        return []

    summaries: List[Dict[str, Any]] = []
    for file in sorted(path.glob("*.json"), reverse=True):
        try:
            with open(file, "r") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.warning("Skipping unreadable backtest file %s: %s", file, e)
            continue

        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            continue

        for r in results:
            summaries.append(
                {
                    "file": file.name,
                    "generated_at": data.get("generated_at"),
                    "symbol": r.get("symbol"),
                    "strategy": r.get("strategy"),
                    "timeframe": r.get("timeframe"),
                    "total_return_pct": r.get("total_return_pct", 0.0),
                    "sharpe_ratio": r.get("sharpe_ratio", 0.0),
                    "max_drawdown_pct": r.get("max_drawdown_pct", 0.0),
                    "win_rate": r.get("win_rate", 0.0),
                    "total_trades": r.get("total_trades", 0),
                    "profit_factor": r.get("profit_factor", 0.0),
                }
            )
        # Only the most recent file by default (summaries sorted reverse).
        if summaries:
            break

    summaries.sort(key=lambda s: s["total_return_pct"], reverse=True)
    return summaries


# ───────────────────────── Flask app factory ─────────────────────────


def create_app(
    db_path: Optional[str] = None,
    backtest_dir: Optional[str] = None,
) -> Flask:
    """Create the dashboard Flask application.

    ``db_path`` and ``backtest_dir`` are captured in closures so tests can
    point the app at a temp DB without monkeypatching module-level state.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["DB_PATH"] = db_path or DEFAULT_DB_PATH
    app.config["BACKTEST_DIR"] = backtest_dir or DEFAULT_BACKTEST_DIR
    app.config["STARTED_AT"] = datetime.utcnow().isoformat()

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/api/overview")
    def api_overview():
        data = build_overview(app.config["DB_PATH"])
        data["runner_started_at"] = app.config["STARTED_AT"]
        now = datetime.utcnow()
        try:
            started = datetime.fromisoformat(app.config["STARTED_AT"])
            uptime_seconds = (now - started).total_seconds()
        except Exception:
            uptime_seconds = 0.0
        data["uptime_seconds"] = uptime_seconds
        data["current_time"] = now.isoformat()
        return jsonify(data)

    @app.route("/api/markets")
    def api_markets():
        return jsonify(build_market_grid(app.config["DB_PATH"]))

    @app.route("/api/strategies")
    def api_strategies():
        return jsonify(build_strategy_table(app.config["DB_PATH"]))

    @app.route("/api/trades")
    def api_trades():
        from flask import request

        try:
            limit = min(int(request.args.get("limit", 50)), 500)
        except ValueError:
            limit = 50
        return jsonify(build_trade_log(app.config["DB_PATH"], limit=limit))

    @app.route("/api/scanner")
    def api_scanner():
        return jsonify(_query_scanner_status(app.config["DB_PATH"]))

    @app.route("/api/backtests")
    def api_backtests():
        return jsonify(build_backtest_summary(app.config["BACKTEST_DIR"]))

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "db_exists": Path(app.config["DB_PATH"]).exists()})

    @app.route("/api/sync", methods=["POST"])
    def api_sync():
        """Accept a SQLite DB upload from the local runner for remote sync."""
        sync_token = os.environ.get("SYNC_TOKEN", "")
        if sync_token and flask_request.headers.get("X-Sync-Token") != sync_token:
            return jsonify({"error": "unauthorized"}), 401
        if "db" not in flask_request.files:
            return jsonify({"error": "no db file in request"}), 400
        db_file = flask_request.files["db"]
        db_path = Path(app.config["DB_PATH"])
        db_file.save(str(db_path))
        logger.info("DB synced from remote (%d bytes)", db_path.stat().st_size)
        return jsonify({"status": "ok", "size": db_path.stat().st_size})

    @app.route("/static/<path:filename>")
    def static_files(filename):
        return send_from_directory(app.static_folder, filename)

    return app


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Ensure static/templates dirs exist so Flask doesn't complain.
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)

    app = create_app()
    logger.info("QuantAgent Dashboard starting on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    main()
