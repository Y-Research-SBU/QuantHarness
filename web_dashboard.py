"""
Flask blueprint for paper trading dashboard, strategy comparison, and trade journal.
"""

import json
from datetime import datetime
from typing import Optional

from flask import Blueprint, jsonify, render_template_string, request

from db_schema import get_connection
from market_config import MARKETS, MarketCategory
from paper_trading import PaperTradingEngine
from performance_tracker import (
    calculate_performance,
    get_api_cost_summary,
    get_portfolio_snapshots,
    get_strategy_performance,
)

dashboard_bp = Blueprint("dashboard", __name__)

# Lazy-init engine
_engine: Optional[PaperTradingEngine] = None


def _get_engine() -> PaperTradingEngine:
    global _engine
    if _engine is None:
        _engine = PaperTradingEngine()
    return _engine


# ───────────────────────────── Dashboard Page ─────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>QuantAgent — Paper Trading Dashboard</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        h2 { color: #58a6ff; margin: 20px 0 10px; font-size: 1.2em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
        .card h3 { color: #8b949e; font-size: 0.85em; margin-bottom: 6px; text-transform: uppercase; }
        .card .value { font-size: 1.6em; font-weight: 700; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral { color: #8b949e; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }
        th { color: #8b949e; font-size: 0.85em; text-transform: uppercase; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
        .badge-long { background: #0d419d; color: #58a6ff; }
        .badge-short { background: #6e1010; color: #f85149; }
        .nav { display: flex; gap: 15px; margin-bottom: 20px; }
        .nav a { color: #58a6ff; text-decoration: none; padding: 8px 16px; border: 1px solid #30363d;
                 border-radius: 6px; }
        .nav a:hover { background: #161b22; }
        .nav a.active { background: #1f6feb; color: white; border-color: #1f6feb; }
    </style>
</head>
<body>
    <h1>📊 QuantAgent Paper Trading Dashboard</h1>
    <div class="nav">
        <a href="/dashboard" class="active">Dashboard</a>
        <a href="/dashboard/strategies">Strategies</a>
        <a href="/dashboard/journal">Trade Journal</a>
        <a href="/">Analysis</a>
    </div>
    <div id="content">Loading...</div>
    <script>
        async function load() {
            const [portfolios, summary] = await Promise.all([
                fetch('/api/dashboard/portfolios').then(r => r.json()),
                fetch('/api/dashboard/summary').then(r => r.json()),
            ]);
            let html = '<h2>Portfolio Summary</h2><div class="grid">';
            html += card('Total P&L', '$' + summary.total_pnl.toFixed(2), summary.total_pnl >= 0 ? 'positive' : 'negative');
            html += card('Total Trades', summary.total_trades, 'neutral');
            html += card('Win Rate', (summary.win_rate * 100).toFixed(1) + '%', summary.win_rate >= 0.5 ? 'positive' : 'negative');
            html += card('Open Positions', summary.open_positions, 'neutral');
            html += card('API Cost', '$' + summary.api_cost.toFixed(4), 'neutral');
            html += '</div>';
            html += '<h2>Market Portfolios</h2><table><tr><th>Market</th><th>Balance</th><th>P&L</th><th>Trades</th><th>Win Rate</th><th>Drawdown</th><th>Status</th></tr>';
            for (const p of portfolios) {
                const wr = p.total_trades > 0 ? (p.winning_trades / p.total_trades * 100).toFixed(1) + '%' : '-';
                const pnlClass = p.total_pnl >= 0 ? 'positive' : 'negative';
                const status = p.is_circuit_breaker_active ? '🔴 Halted' : '🟢 Active';
                html += '<tr><td><strong>' + p.symbol + '</strong></td><td>$' + p.current_balance.toFixed(2) + '</td>';
                html += '<td class="' + pnlClass + '">$' + p.total_pnl.toFixed(2) + '</td>';
                html += '<td>' + p.total_trades + '</td><td>' + wr + '</td>';
                html += '<td>' + (p.max_drawdown * 100).toFixed(1) + '%</td><td>' + status + '</td></tr>';
            }
            html += '</table>';
            document.getElementById('content').innerHTML = html;
        }
        function card(title, value, cls) {
            return '<div class="card"><h3>' + title + '</h3><div class="value ' + cls + '">' + value + '</div></div>';
        }
        load();
    </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard")
def dashboard_page():
    return render_template_string(DASHBOARD_HTML)


# ───────────────────────────── Strategy Comparison Page ─────────────────────────────

STRATEGIES_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>QuantAgent — Strategy Comparison</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        h2 { color: #58a6ff; margin: 20px 0 10px; font-size: 1.2em; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }
        th { color: #8b949e; font-size: 0.85em; text-transform: uppercase; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .nav { display: flex; gap: 15px; margin-bottom: 20px; }
        .nav a { color: #58a6ff; text-decoration: none; padding: 8px 16px; border: 1px solid #30363d;
                 border-radius: 6px; }
        .nav a:hover { background: #161b22; }
        .nav a.active { background: #1f6feb; color: white; border-color: #1f6feb; }
    </style>
</head>
<body>
    <h1>📈 Strategy Comparison</h1>
    <div class="nav">
        <a href="/dashboard">Dashboard</a>
        <a href="/dashboard/strategies" class="active">Strategies</a>
        <a href="/dashboard/journal">Trade Journal</a>
        <a href="/">Analysis</a>
    </div>
    <div id="content">Loading...</div>
    <script>
        async function load() {
            const data = await fetch('/api/dashboard/strategy-performance').then(r => r.json());
            let html = '<table><tr><th>Strategy</th><th>Market</th><th>TF</th><th>Trades</th><th>Win Rate</th><th>P&L</th><th>Avg Win</th><th>Avg Loss</th><th>Profit Factor</th></tr>';
            for (const s of data) {
                const pnlClass = s.total_pnl >= 0 ? 'positive' : 'negative';
                html += '<tr><td><strong>' + s.strategy + '</strong></td><td>' + s.symbol + '</td><td>' + s.timeframe + '</td>';
                html += '<td>' + s.total_trades + '</td><td>' + (s.win_rate * 100).toFixed(1) + '%</td>';
                html += '<td class="' + pnlClass + '">$' + s.total_pnl.toFixed(2) + '</td>';
                html += '<td class="positive">$' + s.avg_win.toFixed(2) + '</td>';
                html += '<td class="negative">$' + s.avg_loss.toFixed(2) + '</td>';
                html += '<td>' + s.profit_factor.toFixed(2) + '</td></tr>';
            }
            html += '</table>';
            if (data.length === 0) html = '<p>No strategy data yet. Run some paper trades first!</p>';
            document.getElementById('content').innerHTML = html;
        }
        load();
    </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard/strategies")
def strategies_page():
    return render_template_string(STRATEGIES_HTML)


# ───────────────────────────── Trade Journal Page ─────────────────────────────

JOURNAL_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>QuantAgent — Trade Journal</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; font-size: 0.9em; }
        th { color: #8b949e; font-size: 0.8em; text-transform: uppercase; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
        .badge-long { background: #0d419d; color: #58a6ff; }
        .badge-short { background: #6e1010; color: #f85149; }
        .badge-open { background: #1a4730; color: #3fb950; }
        .reasoning { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
        .reasoning:hover { white-space: normal; }
        .nav { display: flex; gap: 15px; margin-bottom: 20px; }
        .nav a { color: #58a6ff; text-decoration: none; padding: 8px 16px; border: 1px solid #30363d;
                 border-radius: 6px; }
        .nav a:hover { background: #161b22; }
        .nav a.active { background: #1f6feb; color: white; border-color: #1f6feb; }
    </style>
</head>
<body>
    <h1>📝 Trade Journal</h1>
    <div class="nav">
        <a href="/dashboard">Dashboard</a>
        <a href="/dashboard/strategies">Strategies</a>
        <a href="/dashboard/journal" class="active">Trade Journal</a>
        <a href="/">Analysis</a>
    </div>
    <div id="content">Loading...</div>
    <script>
        async function load() {
            const data = await fetch('/api/dashboard/trades?limit=100').then(r => r.json());
            let html = '<table><tr><th>#</th><th>Symbol</th><th>Dir</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>Size</th><th>P&L</th><th>Status</th><th>Reasoning</th></tr>';
            for (const t of data) {
                const dirClass = t.direction === 'LONG' ? 'badge-long' : 'badge-short';
                const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
                const statusClass = t.status === 'OPEN' ? 'badge-open' : '';
                html += '<tr><td>' + t.id + '</td><td><strong>' + t.symbol + '</strong></td>';
                html += '<td><span class="badge ' + dirClass + '">' + t.direction + '</span></td>';
                html += '<td>' + t.strategy + '</td>';
                html += '<td>$' + (t.entry_price || 0).toFixed(2) + '</td>';
                html += '<td>' + (t.exit_price ? '$' + t.exit_price.toFixed(2) : '-') + '</td>';
                html += '<td>$' + (t.position_size || 0).toFixed(2) + '</td>';
                html += '<td class="' + pnlClass + '">$' + (t.pnl || 0).toFixed(2) + '</td>';
                html += '<td><span class="badge ' + statusClass + '">' + t.status + '</span></td>';
                html += '<td class="reasoning">' + (t.agent_reasoning || '-') + '</td></tr>';
            }
            html += '</table>';
            if (data.length === 0) html = '<p>No trades yet. Run the scanner to generate paper trades!</p>';
            document.getElementById('content').innerHTML = html;
        }
        load();
    </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard/journal")
def journal_page():
    return render_template_string(JOURNAL_HTML)


# ───────────────────────────── API Endpoints ─────────────────────────────

@dashboard_bp.route("/api/dashboard/portfolios")
def api_portfolios():
    engine = _get_engine()
    return jsonify(engine.get_all_portfolios())


@dashboard_bp.route("/api/dashboard/summary")
def api_summary():
    engine = _get_engine()
    portfolios = engine.get_all_portfolios()
    
    total_pnl = sum(p["total_pnl"] for p in portfolios)
    total_trades = sum(p["total_trades"] for p in portfolios)
    total_wins = sum(p["winning_trades"] for p in portfolios)
    open_positions = len(engine.get_open_positions())
    win_rate = total_wins / total_trades if total_trades > 0 else 0
    api_cost = get_api_cost_summary().get("total_cost", 0) or 0
    
    return jsonify({
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "open_positions": open_positions,
        "api_cost": api_cost,
    })


@dashboard_bp.route("/api/dashboard/strategy-performance")
def api_strategy_performance():
    return jsonify(get_strategy_performance())


@dashboard_bp.route("/api/dashboard/trades")
def api_trades():
    engine = _get_engine()
    symbol = request.args.get("symbol")
    strategy = request.args.get("strategy")
    limit = request.args.get("limit", 100, type=int)
    
    # Get both open and closed trades
    with get_connection() as conn:
        query = "SELECT * FROM trades"
        params: list = []
        conditions = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        return jsonify([dict(r) for r in rows])


@dashboard_bp.route("/api/dashboard/snapshots/<symbol>")
def api_snapshots(symbol):
    snapshots = get_portfolio_snapshots(symbol=symbol)
    return jsonify(snapshots)
