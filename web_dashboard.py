"""
Flask blueprint for the QuantAgent trading terminal.

Pages:
  /dashboard             — live portfolio overview with open positions and allocation
  /dashboard/charts      — TradingView candlesticks with trade markers
  /dashboard/markets     — market grid with sparklines and 24h change
  /dashboard/strategies  — strategy comparison and cards
  /dashboard/equity      — equity curve + drawdown + risk ratios
  /dashboard/journal     — searchable trade journal with CSV export

All HTML is inline (render_template_string) and all client code is vanilla JS.
"""

import csv
import io
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from flask import Blueprint, Response, jsonify, render_template_string, request

from db_schema import get_connection
from market_config import MARKETS
from paper_trading import PaperTradingEngine
from performance_tracker import (
    calculate_performance,
    get_api_cost_summary,
    get_portfolio_snapshots,
    get_strategy_performance,
)

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)

_engine: Optional[PaperTradingEngine] = None


def _get_engine() -> PaperTradingEngine:
    global _engine
    if _engine is None:
        _engine = PaperTradingEngine()
    return _engine


# ───────────────────────────── Live price cache ─────────────────────────────

_PRICE_CACHE_TTL = 30.0          # seconds
_CANDLE_CACHE_TTL = 60.0         # seconds
_price_cache: Dict[str, Any] = {"ts": 0.0, "prices": {}, "change24h": {}, "volumes": {}, "sparks": {}}
_candle_cache: Dict[str, Any] = {}


def _safe_last_close(series: pd.Series) -> Optional[float]:
    try:
        s = series.dropna()
        if len(s) == 0:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def _fetch_bulk_prices(symbols: List[str]) -> None:
    """Populate the price cache for all symbols at once. Silent on failure."""
    if not symbols:
        return
    try:
        data = yf.download(
            tickers=" ".join(symbols),
            period="2d",
            interval="15m",
            progress=False,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
    except Exception as e:
        logger.warning("yfinance bulk price fetch failed: %s", e)
        return

    if data is None or data.empty:
        return

    prices: Dict[str, float] = {}
    change: Dict[str, float] = {}
    volumes: Dict[str, float] = {}
    sparks: Dict[str, List[float]] = {}

    def _extract(df: pd.DataFrame, sym: str):
        try:
            closes = df["Close"].dropna()
            if len(closes) == 0:
                return
            last = float(closes.iloc[-1])
            # 24h change — walk back ~24h (96 x 15m bars)
            idx_24h = max(len(closes) - 96, 0)
            prev = float(closes.iloc[idx_24h])
            prices[sym] = last
            change[sym] = ((last - prev) / prev * 100.0) if prev else 0.0
            try:
                vol = float(df["Volume"].dropna().tail(96).sum())
                volumes[sym] = vol
            except Exception:
                volumes[sym] = 0.0
            tail = closes.tail(60).tolist()
            sparks[sym] = [float(x) for x in tail]
        except Exception:
            return

    # Multi-symbol response has a two-level column MultiIndex keyed by ticker
    if len(symbols) > 1:
        for sym in symbols:
            try:
                sub = data[sym]
                _extract(sub, sym)
            except Exception:
                continue
    else:
        _extract(data, symbols[0])

    _price_cache["ts"] = time.time()
    _price_cache["prices"].update(prices)
    _price_cache["change24h"].update(change)
    _price_cache["volumes"].update(volumes)
    _price_cache["sparks"].update(sparks)


def _get_live_prices(symbols: List[str]) -> Dict[str, Any]:
    """Returns {'prices':{sym:float}, 'change24h':{}, 'volumes':{}, 'sparks':{}}."""
    now = time.time()
    cached = _price_cache["prices"]
    fresh = now - _price_cache["ts"] < _PRICE_CACHE_TTL
    missing = [s for s in symbols if s not in cached]
    if not fresh or missing:
        _fetch_bulk_prices(symbols)
    return {
        "prices": {s: _price_cache["prices"].get(s) for s in symbols},
        "change24h": {s: _price_cache["change24h"].get(s, 0.0) for s in symbols},
        "volumes": {s: _price_cache["volumes"].get(s, 0.0) for s in symbols},
        "sparks": {s: _price_cache["sparks"].get(s, []) for s in symbols},
    }


def _fetch_candles(symbol: str, interval: str = "4h", lookback_days: int = 60) -> List[Dict[str, Any]]:
    key = f"{symbol}:{interval}:{lookback_days}"
    now = time.time()
    hit = _candle_cache.get(key)
    if hit and now - hit["ts"] < _CANDLE_CACHE_TTL:
        return hit["data"]

    period_map = {
        "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
        "1h": "730d", "4h": "730d", "1d": "3y", "1wk": "5y", "1mo": "10y",
    }
    period = period_map.get(interval, "60d")
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        logger.warning("yfinance candle fetch failed for %s: %s", symbol, e)
        return []
    if df is None or df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else "Date"
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            ts = row[tcol]
            if hasattr(ts, "timestamp"):
                t = int(ts.timestamp())
            else:
                t = int(pd.Timestamp(ts).timestamp())
            out.append({
                "time": t,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume") or 0.0),
            })
        except Exception:
            continue
    _candle_cache[key] = {"ts": now, "data": out}
    return out


# ───────────────────────────── Unrealized P&L ─────────────────────────────

def _unrealized_pnl(trade: Dict[str, Any], current_price: Optional[float]) -> Dict[str, float]:
    entry = float(trade.get("entry_price") or 0.0)
    qty = float(trade.get("quantity") or 0.0)
    size = float(trade.get("position_size") or 0.0)
    if current_price is None or entry == 0 or qty == 0:
        return {"pnl": 0.0, "pnl_pct": 0.0, "current_price": current_price or entry}
    direction = (trade.get("direction") or "LONG").upper()
    if direction == "LONG":
        pnl = (current_price - entry) * qty
    else:
        pnl = (entry - current_price) * qty
    pnl_pct = (pnl / size * 100.0) if size else 0.0
    return {"pnl": pnl, "pnl_pct": pnl_pct, "current_price": current_price}


# ───────────────────────────── Shared CSS + nav ─────────────────────────────

_BASE_CSS = """
<style>
:root {
  --bg: #0d1117;
  --card: #161b22;
  --card-hi: #1c232c;
  --border: #30363d;
  --text: #c9d1d9;
  --muted: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --green-dim: #1a4730;
  --red: #f85149;
  --red-dim: #6e1010;
  --amber: #d29922;
  --blue: #1f6feb;
  --purple: #a371f7;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { background: var(--bg); color: var(--text); }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  min-height: 100vh;
  padding: 0 0 48px;
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header.topbar {
  display: flex; align-items: center; gap: 16px;
  padding: 12px 20px; border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #0d1117, #0b0f14);
  position: sticky; top: 0; z-index: 10;
  flex-wrap: wrap;
}
.brand { display: flex; align-items: center; gap: 10px; font-weight: 600; letter-spacing: 0.2px; }
.brand-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--green);
             box-shadow: 0 0 10px var(--green); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.brand-text { color: #f0f6fc; }
.brand-sub { color: var(--muted); font-weight: 400; margin-left: 4px; font-size: 0.85em; }
nav.mainnav { display: flex; gap: 2px; flex-wrap: wrap; margin-left: 12px; }
nav.mainnav a {
  color: var(--muted); padding: 7px 12px; border-radius: 6px;
  font-size: 0.9em; font-weight: 500; transition: all .15s;
  text-decoration: none;
}
nav.mainnav a:hover { color: var(--text); background: var(--card); }
nav.mainnav a.active { color: #fff; background: var(--blue); }
.scanner-pill {
  margin-left: auto; padding: 5px 10px; border-radius: 14px;
  font-size: 0.8em; background: var(--card); border: 1px solid var(--border);
  color: var(--muted); white-space: nowrap;
}
.scanner-pill.running { color: var(--green); border-color: var(--green-dim); }
.scanner-pill.idle { color: var(--muted); }
.scanner-controls { display: flex; gap: 6px; }
.scanner-controls button {
  background: var(--card); color: var(--text); border: 1px solid var(--border);
  padding: 6px 12px; border-radius: 6px; font-size: 0.85em; cursor: pointer;
  font-family: inherit; transition: all .15s;
}
.scanner-controls button:hover { background: var(--card-hi); border-color: var(--accent); }
.scanner-controls button:disabled { opacity: 0.5; cursor: not-allowed; }
main { padding: 20px 24px; max-width: 1680px; margin: 0 auto; }
h1.page-title { font-size: 1.4em; margin-bottom: 4px; color: #f0f6fc; font-weight: 600; }
.page-sub { color: var(--muted); margin-bottom: 20px; font-size: 0.9em; }
.hero { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 22px; }
.hero-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; transition: all .2s ease;
}
.hero-card:hover { border-color: #3d454f; transform: translateY(-1px); }
.hero-card h3 {
  color: var(--muted); font-size: 0.75em; text-transform: uppercase;
  letter-spacing: 0.7px; margin-bottom: 8px; font-weight: 500;
}
.hero-card .value { font-size: 1.9em; font-weight: 700; line-height: 1.1; letter-spacing: -0.5px; }
.hero-card .sub { color: var(--muted); font-size: 0.8em; margin-top: 4px; }
.positive { color: var(--green); }
.negative { color: var(--red); }
.neutral { color: var(--text); }
.muted { color: var(--muted); }
.panel {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px; margin-bottom: 18px;
}
.panel h2 {
  color: #f0f6fc; font-size: 1.05em; font-weight: 600; margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px;
}
.panel h2 .count { color: var(--muted); font-weight: 400; font-size: 0.85em; }
.two-col { display: grid; grid-template-columns: 2fr 1fr; gap: 18px; }
@media (max-width: 1000px) { .two-col { grid-template-columns: 1fr; } }
table { width: 100%; border-collapse: collapse; }
th, td {
  padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border);
  font-size: 0.9em;
}
th {
  color: var(--muted); font-size: 0.72em; text-transform: uppercase;
  letter-spacing: 0.6px; font-weight: 500; background: rgba(0,0,0,0.15);
}
tr:last-child td { border-bottom: none; }
tr.clickable { cursor: pointer; }
tr.clickable:hover { background: var(--card-hi); }
tr.expanded-row td { background: #0f1419; padding: 14px 16px; font-size: 0.85em; color: var(--muted); }
.badge {
  display: inline-block; padding: 2px 9px; border-radius: 12px;
  font-size: 0.75em; font-weight: 600; letter-spacing: 0.3px;
}
.badge-long { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.3); }
.badge-short { background: rgba(248,81,73,0.15); color: var(--red); border: 1px solid rgba(248,81,73,0.3); }
.badge-open { background: rgba(88,166,255,0.15); color: var(--accent); border: 1px solid rgba(88,166,255,0.3); }
.badge-closed { background: rgba(139,148,158,0.15); color: var(--muted); border: 1px solid var(--border); }
.badge-stopped { background: rgba(210,153,34,0.15); color: var(--amber); border: 1px solid rgba(210,153,34,0.3); }
input, select, button.action {
  background: var(--card); color: var(--text); border: 1px solid var(--border);
  padding: 7px 11px; border-radius: 6px; font-family: inherit; font-size: 0.88em;
}
input:focus, select:focus { outline: none; border-color: var(--accent); }
button.action { cursor: pointer; transition: all .15s; }
button.action:hover { background: var(--card-hi); border-color: var(--accent); }
button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
button.primary:hover { background: #388bfd; }
.filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; align-items: center; }
.filters label { color: var(--muted); font-size: 0.85em; margin-right: 4px; }
.chart-wrap { width: 100%; height: 520px; background: var(--bg); border-radius: 8px; position: relative; }
.market-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.market-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px; cursor: pointer; transition: all .2s ease; position: relative; overflow: hidden;
}
.market-card:hover { border-color: var(--accent); transform: translateY(-1px); }
.market-card.profitable::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--green);
}
.market-card.losing::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--red);
}
.market-card .sym { font-weight: 700; font-size: 1.05em; color: #f0f6fc; }
.market-card .name { color: var(--muted); font-size: 0.8em; margin-bottom: 8px; }
.market-card .price { font-size: 1.3em; font-weight: 600; margin: 6px 0; letter-spacing: -0.3px; }
.market-card .change { font-size: 0.9em; font-weight: 600; }
.market-card .spark { margin-top: 6px; height: 40px; }
.strategy-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.strategy-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px;
}
.strategy-card h3 {
  font-size: 1em; color: #f0f6fc; margin-bottom: 10px; text-transform: capitalize;
}
.strategy-card .row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.88em; }
.strategy-card .row .k { color: var(--muted); }
.pie-legend { display: flex; flex-direction: column; gap: 6px; font-size: 0.85em; }
.pie-legend .item { display: flex; align-items: center; gap: 8px; }
.pie-legend .dot { width: 10px; height: 10px; border-radius: 2px; }
.kv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
.kv { background: rgba(0,0,0,0.2); padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border); }
.kv .k { color: var(--muted); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.4px; }
.kv .v { font-size: 1.25em; font-weight: 600; margin-top: 3px; }
.loading { color: var(--muted); padding: 20px; text-align: center; font-size: 0.9em; }
.empty { color: var(--muted); padding: 32px; text-align: center; font-size: 0.9em; }
.flash { animation: flash .5s ease; }
@keyframes flash { 0% { background: rgba(88,166,255,0.15); } 100% { background: transparent; } }
.num { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }
.reasoning-box {
  background: rgba(0,0,0,0.3); border-left: 3px solid var(--accent);
  padding: 10px 14px; border-radius: 4px; margin: 8px 0; white-space: pre-wrap;
  font-size: 0.85em; line-height: 1.5; color: var(--text);
}
</style>
"""

_NAV_HTML = """
<header class="topbar">
  <div class="brand">
    <span class="brand-dot"></span>
    <span class="brand-text">QuantAgent <span class="brand-sub">Terminal</span></span>
  </div>
  <nav class="mainnav">
    <a href="/dashboard" data-route="/dashboard">Overview</a>
    <a href="/dashboard/charts" data-route="/dashboard/charts">Charts</a>
    <a href="/dashboard/markets" data-route="/dashboard/markets">Markets</a>
    <a href="/dashboard/strategies" data-route="/dashboard/strategies">Strategies</a>
    <a href="/dashboard/equity" data-route="/dashboard/equity">Equity</a>
    <a href="/dashboard/journal" data-route="/dashboard/journal">Journal</a>
    <a href="/" target="_blank" rel="noopener">Analysis ↗</a>
  </nav>
  <span id="scanner-status" class="scanner-pill">…</span>
  <div class="scanner-controls">
    <button id="btn-scan-once" title="Run a single scan cycle">Scan Once</button>
    <button id="btn-start" title="Start background scanner">Start</button>
    <button id="btn-stop" title="Stop background scanner">Stop</button>
  </div>
</header>
<script>
(function(){
  const path = (location.pathname.replace(/\\/$/, '') || '/dashboard');
  document.querySelectorAll('nav.mainnav a[data-route]').forEach(a => {
    if (a.getAttribute('data-route') === path) a.classList.add('active');
  });
  async function refreshStatus() {
    try {
      const r = await fetch('/api/status');
      const j = await r.json();
      const pill = document.getElementById('scanner-status');
      if (!pill) return;
      if (j.scanner && j.scanner.running) {
        pill.textContent = '● Scanner: Running';
        pill.className = 'scanner-pill running';
      } else {
        pill.textContent = '○ Scanner: Idle';
        pill.className = 'scanner-pill idle';
      }
    } catch(e){}
  }
  async function postJson(url, body){
    return fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body||{})});
  }
  document.getElementById('btn-start')?.addEventListener('click', async () => {
    await postJson('/api/start-scanner', {interval_seconds: 14400, use_agents: false});
    refreshStatus();
  });
  document.getElementById('btn-stop')?.addEventListener('click', async () => {
    await fetch('/api/stop-scanner', {method:'POST'});
    refreshStatus();
  });
  document.getElementById('btn-scan-once')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-scan-once');
    btn.disabled = true; btn.textContent = 'Scanning…';
    try { await fetch('/api/scan-once', {method:'POST'}); } catch(e){}
    btn.disabled = false; btn.textContent = 'Scan Once';
    refreshStatus();
  });
  refreshStatus();
  setInterval(refreshStatus, 10000);
})();
</script>
"""

_SHARED_JS = """
<script>
window.QA = window.QA || {};
QA.fmt = {
  money: (v, d=2) => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    const sign = v < 0 ? '-' : '';
    const abs = Math.abs(v);
    return sign + '$' + abs.toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
  },
  pct: (v, d=2) => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return (v>=0?'+':'') + v.toFixed(d) + '%';
  },
  num: (v, d=2) => {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return Number(v).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
  },
  cls: v => (v >= 0 ? 'positive' : 'negative'),
  ago: iso => {
    if (!iso) return '—';
    const d = new Date(iso);
    const s = Math.floor((Date.now() - d.getTime())/1000);
    if (s < 60) return s + 's ago';
    if (s < 3600) return Math.floor(s/60) + 'm ago';
    if (s < 86400) return Math.floor(s/3600) + 'h ago';
    return Math.floor(s/86400) + 'd ago';
  },
  shortTime: iso => {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'});
  },
};
QA.flash = el => {
  if (!el) return;
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
};
</script>
"""


# ───────────────────────────── Overview (/dashboard) ─────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard · QuantAgent Terminal</title>
  {{ base_css|safe }}
</head>
<body>
  {{ nav_html|safe }}
  {{ shared_js|safe }}
  <main>
    <h1 class="page-title">Dashboard</h1>
    <p class="page-sub">Real-time portfolio state · live P&amp;L updates every 15 s</p>

    <div class="hero" id="hero">
      <div class="hero-card"><h3>Total P&amp;L</h3><div class="value num" id="h-pnl">—</div><div class="sub" id="h-pnl-sub">realized + unrealized</div></div>
      <div class="hero-card"><h3>Total Balance</h3><div class="value num" id="h-bal">—</div><div class="sub" id="h-bal-sub">across all markets</div></div>
      <div class="hero-card"><h3>Win Rate</h3><div class="value num" id="h-wr">—</div><div class="sub" id="h-wr-sub">closed trades</div></div>
      <div class="hero-card"><h3>Open Positions</h3><div class="value num" id="h-op">—</div><div class="sub" id="h-op-sub">live</div></div>
      <div class="hero-card"><h3>API Cost</h3><div class="value num" id="h-cost">—</div><div class="sub">agent LLM spend</div></div>
    </div>

    <div class="two-col">
      <div class="panel">
        <h2>Open Positions <span class="count" id="pos-count"></span></h2>
        <div id="positions-container"><div class="loading">Loading positions…</div></div>
      </div>
      <div class="panel">
        <h2>Portfolio Allocation</h2>
        <div id="alloc-container"><div class="loading">Loading allocation…</div></div>
      </div>
    </div>

    <div class="panel">
      <h2>Market Portfolios <span class="count" id="port-count"></span></h2>
      <div id="portfolios-container"><div class="loading">Loading portfolios…</div></div>
    </div>
  </main>

  <script>
  const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#a371f7', '#f778ba', '#56d4dd', '#f85149', '#8b949e', '#ff7b72', '#79c0ff', '#bc8cff', '#7ee787'];

  function dirBadge(d){ return '<span class="badge ' + (d==='LONG'?'badge-long':'badge-short') + '">' + d + '</span>'; }

  function renderPositions(positions){
    const box = document.getElementById('positions-container');
    document.getElementById('pos-count').textContent = positions.length ? '· ' + positions.length : '';
    if (!positions.length){
      box.innerHTML = '<div class="empty">No open positions. Start the scanner to generate paper trades.</div>';
      return;
    }
    let h = '<table><thead><tr><th>Sym</th><th>Dir</th><th>Strat</th><th class="num">Entry</th><th class="num">Current</th><th class="num">P&amp;L</th><th class="num">%</th><th>Held</th></tr></thead><tbody>';
    for (const p of positions){
      const cls = QA.fmt.cls(p.unrealized_pnl);
      h += '<tr data-pnl="' + p.unrealized_pnl + '"><td><strong>' + p.symbol + '</strong></td>';
      h += '<td>' + dirBadge(p.direction) + '</td>';
      h += '<td class="muted">' + p.strategy + '</td>';
      h += '<td class="num">' + QA.fmt.money(p.entry_price) + '</td>';
      h += '<td class="num">' + QA.fmt.money(p.current_price) + '</td>';
      h += '<td class="num ' + cls + '">' + QA.fmt.money(p.unrealized_pnl) + '</td>';
      h += '<td class="num ' + cls + '">' + QA.fmt.pct(p.unrealized_pnl_pct) + '</td>';
      h += '<td class="muted">' + QA.fmt.ago(p.entry_time) + '</td></tr>';
    }
    h += '</tbody></table>';
    box.innerHTML = h;
  }

  function donutChart(slices){
    // slices: [{label, value, color}]
    const total = slices.reduce((s, x) => s + x.value, 0);
    if (total <= 0) return '<div class="empty">No open exposure</div>';
    const size = 220, cx = size/2, cy = size/2, r = 90, ir = 55;
    let a = -Math.PI/2, paths = '';
    slices.forEach((s, i) => {
      const frac = s.value / total;
      const a2 = a + frac * Math.PI * 2;
      const large = frac > 0.5 ? 1 : 0;
      const x1 = cx + r*Math.cos(a), y1 = cy + r*Math.sin(a);
      const x2 = cx + r*Math.cos(a2), y2 = cy + r*Math.sin(a2);
      const x3 = cx + ir*Math.cos(a2), y3 = cy + ir*Math.sin(a2);
      const x4 = cx + ir*Math.cos(a), y4 = cy + ir*Math.sin(a);
      paths += '<path d="M' + x1 + ',' + y1 + ' A' + r + ',' + r + ' 0 ' + large + ' 1 ' + x2 + ',' + y2 +
               ' L' + x3 + ',' + y3 + ' A' + ir + ',' + ir + ' 0 ' + large + ' 0 ' + x4 + ',' + y4 + ' Z" fill="' + s.color + '"></path>';
      a = a2;
    });
    let legend = '<div class="pie-legend">';
    slices.forEach(s => {
      const pct = (s.value/total*100).toFixed(1);
      legend += '<div class="item"><span class="dot" style="background:' + s.color + '"></span>' +
                '<strong>' + s.label + '</strong><span class="muted" style="margin-left:auto">' + pct + '%</span></div>';
    });
    legend += '</div>';
    return '<div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;justify-content:center;">' +
           '<svg viewBox="0 0 ' + size + ' ' + size + '" width="220" height="220">' + paths +
           '<text x="' + cx + '" y="' + (cy-4) + '" text-anchor="middle" fill="#8b949e" font-size="11">Exposure</text>' +
           '<text x="' + cx + '" y="' + (cy+14) + '" text-anchor="middle" fill="#c9d1d9" font-size="16" font-weight="600">' + QA.fmt.money(total, 0) + '</text>' +
           '</svg>' + legend + '</div>';
  }

  function renderAllocation(positions){
    const box = document.getElementById('alloc-container');
    if (!positions.length){ box.innerHTML = '<div class="empty">No open positions</div>'; return; }
    const grouped = {};
    positions.forEach(p => { grouped[p.symbol] = (grouped[p.symbol] || 0) + (p.position_size || 0); });
    const slices = Object.entries(grouped).map(([label, value], i) => ({label, value, color: COLORS[i % COLORS.length]}));
    slices.sort((a,b) => b.value - a.value);
    box.innerHTML = donutChart(slices);
  }

  function renderPortfolios(portfolios){
    const box = document.getElementById('portfolios-container');
    document.getElementById('port-count').textContent = portfolios.length ? '· ' + portfolios.length : '';
    if (!portfolios.length){ box.innerHTML = '<div class="empty">No portfolios configured</div>'; return; }
    let h = '<table><thead><tr><th>Market</th><th class="num">Balance</th><th class="num">P&amp;L</th><th class="num">Trades</th><th class="num">Win Rate</th><th class="num">Max DD</th><th>Status</th></tr></thead><tbody>';
    for (const p of portfolios){
      const wr = p.total_trades > 0 ? (p.winning_trades / p.total_trades * 100).toFixed(1) + '%' : '—';
      const pnlCls = QA.fmt.cls(p.total_pnl);
      const status = p.is_circuit_breaker_active ? '<span class="badge badge-stopped">Halted</span>' : '<span class="badge badge-open">Active</span>';
      h += '<tr><td><strong>' + p.symbol + '</strong></td>';
      h += '<td class="num">' + QA.fmt.money(p.current_balance) + '</td>';
      h += '<td class="num ' + pnlCls + '">' + QA.fmt.money(p.total_pnl) + '</td>';
      h += '<td class="num">' + p.total_trades + '</td><td class="num">' + wr + '</td>';
      h += '<td class="num muted">' + (p.max_drawdown * 100).toFixed(1) + '%</td>';
      h += '<td>' + status + '</td></tr>';
    }
    h += '</tbody></table>';
    box.innerHTML = h;
  }

  async function refresh(){
    try {
      const [positions, summary, portfolios] = await Promise.all([
        fetch('/api/positions').then(r => r.json()),
        fetch('/api/dashboard/summary').then(r => r.json()),
        fetch('/api/dashboard/portfolios').then(r => r.json()),
      ]);
      // Summary enriched with unrealized
      const unreal = positions.reduce((s, p) => s + (p.unrealized_pnl || 0), 0);
      const totalBal = portfolios.reduce((s, p) => s + p.current_balance, 0);
      const initBal = portfolios.reduce((s, p) => s + p.initial_balance, 0);

      const pnlTotal = summary.total_pnl + unreal;
      const pnlEl = document.getElementById('h-pnl');
      pnlEl.textContent = QA.fmt.money(pnlTotal);
      pnlEl.className = 'value num ' + QA.fmt.cls(pnlTotal);
      QA.flash(pnlEl);
      document.getElementById('h-pnl-sub').innerHTML = 'realized ' + QA.fmt.money(summary.total_pnl) + ' · unrealized <span class="' + QA.fmt.cls(unreal) + '">' + QA.fmt.money(unreal) + '</span>';

      document.getElementById('h-bal').textContent = QA.fmt.money(totalBal, 0);
      const rtn = initBal > 0 ? ((totalBal - initBal) / initBal * 100) : 0;
      document.getElementById('h-bal-sub').innerHTML = '<span class="' + QA.fmt.cls(rtn) + '">' + QA.fmt.pct(rtn) + '</span> vs start';

      const wr = summary.win_rate * 100;
      const wrEl = document.getElementById('h-wr');
      wrEl.textContent = wr.toFixed(1) + '%';
      wrEl.className = 'value num ' + (wr >= 50 ? 'positive' : (summary.total_trades === 0 ? 'neutral' : 'negative'));
      document.getElementById('h-wr-sub').textContent = summary.total_trades + ' total';

      document.getElementById('h-op').textContent = positions.length;
      document.getElementById('h-op-sub').textContent = positions.length > 0 ? 'live tracking' : 'none';
      document.getElementById('h-cost').textContent = QA.fmt.money(summary.api_cost, 4);

      renderPositions(positions);
      renderAllocation(positions);
      renderPortfolios(portfolios);
    } catch(e) {
      console.error(e);
    }
  }
  refresh();
  setInterval(refresh, 15000);
  </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard")
def dashboard_page():
    return render_template_string(
        DASHBOARD_HTML, base_css=_BASE_CSS, nav_html=_NAV_HTML, shared_js=_SHARED_JS
    )


# ───────────────────────────── Charts (/dashboard/charts) ─────────────────────────────

CHARTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Charts · QuantAgent Terminal</title>
  {{ base_css|safe }}
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
  {{ nav_html|safe }}
  {{ shared_js|safe }}
  <main>
    <h1 class="page-title">Charts</h1>
    <p class="page-sub">TradingView lightweight candlesticks · trade markers overlay · auto-refresh 30 s</p>

    <div class="panel">
      <div class="filters">
        <label>Market</label>
        <select id="sym-sel"></select>
        <label>Timeframe</label>
        <select id="tf-sel"></select>
        <span id="price-badge" class="muted" style="margin-left:12px; font-size:0.95em;"></span>
        <span id="change-badge" class="muted" style="margin-left:8px; font-weight:600;"></span>
        <button class="action" id="reload" style="margin-left:auto">Refresh now</button>
      </div>
      <div class="chart-wrap" id="chart"></div>
      <div id="markers-legend" class="muted" style="margin-top: 10px; font-size: 0.85em;"></div>
    </div>
  </main>

  <script>
  const MARKETS = {{ markets_json|safe }};
  const syms = MARKETS.map(m => m.symbol);
  const symSel = document.getElementById('sym-sel');
  const tfSel = document.getElementById('tf-sel');
  MARKETS.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.symbol;
    opt.textContent = m.symbol + ' — ' + m.display_name;
    symSel.appendChild(opt);
  });

  function setTfOptions(symbol){
    const m = MARKETS.find(x => x.symbol === symbol);
    const tfs = (m && m.timeframes.length) ? m.timeframes : ['1d', '4h'];
    tfSel.innerHTML = '';
    tfs.forEach(tf => {
      const o = document.createElement('option');
      o.value = tf; o.textContent = tf;
      tfSel.appendChild(o);
    });
  }

  const chartEl = document.getElementById('chart');
  const chart = LightweightCharts.createChart(chartEl, {
    layout: { background: { color: '#0d1117' }, textColor: '#c9d1d9' },
    grid: { vertLines: { color: '#161b22' }, horzLines: { color: '#161b22' } },
    crosshair: { mode: 1 },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
    autoSize: true,
  });
  const series = chart.addCandlestickSeries({
    upColor: '#3fb950', downColor: '#f85149', borderVisible: false,
    wickUpColor: '#3fb950', wickDownColor: '#f85149',
  });

  async function load(){
    const symbol = symSel.value;
    const tf = tfSel.value || '4h';
    const badge = document.getElementById('price-badge');
    badge.textContent = 'Loading ' + symbol + ' @ ' + tf + '…';
    document.getElementById('change-badge').textContent = '';
    try {
      const [candleRes, tradesRes] = await Promise.all([
        fetch('/api/candles/' + encodeURIComponent(symbol) + '?interval=' + encodeURIComponent(tf)).then(r => r.json()),
        fetch('/api/trades?symbol=' + encodeURIComponent(symbol) + '&limit=200').then(r => r.json()),
      ]);
      const candles = (candleRes.candles || candleRes || []);
      if (!candles.length){
        badge.innerHTML = '<span class="negative">No candle data returned for ' + symbol + '</span>';
        series.setData([]);
        return;
      }
      series.setData(candles.map(c => ({time: c.time, open: c.open, high: c.high, low: c.low, close: c.close})));

      const markers = [];
      (tradesRes || []).forEach(t => {
        if (t.entry_time){
          const ts = Math.floor(new Date(t.entry_time).getTime() / 1000);
          markers.push({
            time: ts,
            position: t.direction === 'LONG' ? 'belowBar' : 'aboveBar',
            color: t.direction === 'LONG' ? '#3fb950' : '#f85149',
            shape: t.direction === 'LONG' ? 'arrowUp' : 'arrowDown',
            text: (t.direction === 'LONG' ? 'L ' : 'S ') + t.strategy,
          });
        }
        if (t.exit_time){
          const ts = Math.floor(new Date(t.exit_time).getTime() / 1000);
          markers.push({
            time: ts, position: 'inBar', color: '#8b949e', shape: 'circle',
            text: 'Exit ' + (t.pnl >= 0 ? '+' : '') + t.pnl.toFixed(2),
          });
        }
      });
      markers.sort((a,b) => a.time - b.time);
      series.setMarkers(markers);

      const last = candles[candles.length - 1].close;
      const first = candles[0].close;
      const chg = ((last - first) / first) * 100;
      badge.textContent = symbol + ' · ' + QA.fmt.money(last);
      const chgEl = document.getElementById('change-badge');
      chgEl.textContent = QA.fmt.pct(chg) + ' (range)';
      chgEl.className = QA.fmt.cls(chg);

      document.getElementById('markers-legend').innerHTML =
        markers.length ? (markers.length + ' trade markers plotted — ▲ long entry · ▼ short entry · ● exit') :
                         'No trades recorded for this market';
      chart.timeScale().fitContent();
    } catch(e){
      console.error(e);
      badge.innerHTML = '<span class="negative">Failed to load: ' + e.message + '</span>';
    }
  }

  symSel.addEventListener('change', () => { setTfOptions(symSel.value); load(); });
  tfSel.addEventListener('change', load);
  document.getElementById('reload').addEventListener('click', load);
  setTfOptions(symSel.value);
  load();
  setInterval(load, 30000);
  </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard/charts")
def charts_page():
    markets = [
        {
            "symbol": s,
            "display_name": cfg.display_name,
            "category": cfg.category.value,
            "timeframes": cfg.timeframes,
        }
        for s, cfg in MARKETS.items()
    ]
    return render_template_string(
        CHARTS_HTML,
        base_css=_BASE_CSS,
        nav_html=_NAV_HTML,
        shared_js=_SHARED_JS,
        markets_json=json.dumps(markets),
    )


# ───────────────────────────── Markets (/dashboard/markets) ─────────────────────────────

MARKETS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Markets · QuantAgent Terminal</title>
  {{ base_css|safe }}
</head>
<body>
  {{ nav_html|safe }}
  {{ shared_js|safe }}
  <main>
    <h1 class="page-title">Markets</h1>
    <p class="page-sub">All tracked markets · 24 h change · profitability tint · click to open charts</p>

    <div class="panel">
      <div id="markets-grid"><div class="loading">Fetching live prices…</div></div>
    </div>
  </main>
  <script>
  function sparkSVG(points){
    if (!points || points.length < 2) return '';
    const w = 200, h = 40;
    const min = Math.min.apply(null, points);
    const max = Math.max.apply(null, points);
    const span = Math.max(max - min, 1e-9);
    const dx = w / (points.length - 1);
    const path = points.map((p, i) => {
      const x = i * dx;
      const y = h - ((p - min) / span) * h;
      return (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
    }).join(' ');
    const up = points[points.length - 1] >= points[0];
    const color = up ? '#3fb950' : '#f85149';
    return '<svg viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" width="100%" height="40">' +
           '<path d="' + path + '" fill="none" stroke="' + color + '" stroke-width="1.5"/></svg>';
  }

  async function refresh(){
    try {
      const data = await fetch('/api/market-prices').then(r => r.json());
      const box = document.getElementById('markets-grid');
      if (!data.markets || !data.markets.length){
        box.innerHTML = '<div class="empty">No market data</div>';
        return;
      }
      let h = '<div class="market-grid">';
      for (const m of data.markets){
        const pnlCls = m.realized_pnl > 0 ? 'profitable' : (m.realized_pnl < 0 ? 'losing' : '');
        const chgCls = QA.fmt.cls(m.change_24h);
        const priceStr = m.price !== null ? QA.fmt.money(m.price, m.price > 100 ? 2 : 4) : '—';
        h += '<div class="market-card ' + pnlCls + '" onclick="location.href=\\'/dashboard/charts?symbol=' + encodeURIComponent(m.symbol) + '\\'">';
        h += '<div class="sym">' + m.symbol + '</div>';
        h += '<div class="name">' + m.display_name + ' · ' + m.category + '</div>';
        h += '<div class="price num">' + priceStr + '</div>';
        h += '<div class="change num ' + chgCls + '">' + QA.fmt.pct(m.change_24h) + ' 24h</div>';
        h += '<div class="spark">' + sparkSVG(m.spark) + '</div>';
        h += '<div class="muted" style="font-size:0.8em; margin-top:4px;">';
        h += 'Realized P&L <span class="' + QA.fmt.cls(m.realized_pnl) + '">' + QA.fmt.money(m.realized_pnl) + '</span>';
        if (m.open_positions) h += ' · ' + m.open_positions + ' open';
        h += '</div>';
        h += '</div>';
      }
      h += '</div>';
      box.innerHTML = h;
    } catch(e){
      console.error(e);
      document.getElementById('markets-grid').innerHTML = '<div class="empty negative">Failed to fetch prices</div>';
    }
  }
  refresh();
  setInterval(refresh, 30000);
  </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard/markets")
def markets_page():
    return render_template_string(
        MARKETS_HTML, base_css=_BASE_CSS, nav_html=_NAV_HTML, shared_js=_SHARED_JS
    )


# ───────────────────────────── Strategies (/dashboard/strategies) ─────────────────────────────

STRATEGIES_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strategies · QuantAgent Terminal</title>
  {{ base_css|safe }}
</head>
<body>
  {{ nav_html|safe }}
  {{ shared_js|safe }}
  <main>
    <h1 class="page-title">Strategy Performance</h1>
    <p class="page-sub">Comparative P&amp;L, win rate, and trade count across strategies</p>

    <div class="panel">
      <h2>Strategy Comparison</h2>
      <div id="bars-container"><div class="loading">Loading…</div></div>
    </div>

    <div class="panel">
      <h2>Strategy Cards</h2>
      <div id="cards-container" class="strategy-grid"></div>
    </div>

    <div class="panel">
      <h2>Per-Strategy / Market Breakdown</h2>
      <div id="table-container"><div class="loading">Loading…</div></div>
    </div>
  </main>
  <script>
  function barChart(data, keyFn, labelFn, valFn, fmtFn, color){
    if (!data.length) return '<div class="empty">No data</div>';
    const maxVal = Math.max.apply(null, data.map(valFn).map(Math.abs));
    const h = data.length * 34 + 20;
    let svg = '<svg viewBox="0 0 500 ' + h + '" width="100%" height="' + h + '" preserveAspectRatio="none">';
    data.forEach((d, i) => {
      const y = 10 + i * 34;
      const v = valFn(d);
      const w = maxVal > 0 ? (Math.abs(v) / maxVal) * 320 : 0;
      const fill = typeof color === 'function' ? color(d) : color;
      svg += '<rect x="140" y="' + y + '" width="' + w + '" height="22" fill="' + fill + '" rx="3"/>';
      svg += '<text x="135" y="' + (y + 16) + '" fill="#c9d1d9" font-size="12" text-anchor="end">' + labelFn(d) + '</text>';
      svg += '<text x="' + (140 + w + 6) + '" y="' + (y + 16) + '" fill="#8b949e" font-size="12">' + fmtFn(v) + '</text>';
    });
    svg += '</svg>';
    return svg;
  }

  function groupByStrategy(rows){
    const agg = {};
    for (const r of rows){
      const s = r.strategy;
      if (!agg[s]){
        agg[s] = {strategy: s, total_trades: 0, winning_trades: 0, total_pnl: 0, avg_win: 0, avg_loss: 0, profit_factor: 0};
      }
      agg[s].total_trades += r.total_trades;
      agg[s].winning_trades += r.winning_trades;
      agg[s].total_pnl += r.total_pnl;
      agg[s].avg_win = Math.max(agg[s].avg_win, r.avg_win);
      agg[s].avg_loss = Math.max(agg[s].avg_loss, r.avg_loss);
      agg[s].profit_factor = Math.max(agg[s].profit_factor, r.profit_factor);
    }
    Object.values(agg).forEach(s => { s.win_rate = s.total_trades > 0 ? s.winning_trades / s.total_trades : 0; });
    return Object.values(agg).sort((a,b) => b.total_pnl - a.total_pnl);
  }

  async function load(){
    const rows = await fetch('/api/dashboard/strategy-performance').then(r => r.json());
    const bars = document.getElementById('bars-container');
    const cards = document.getElementById('cards-container');
    const tab = document.getElementById('table-container');

    if (!rows.length){
      const empty = '<div class="empty">No strategy performance recorded yet — run the scanner to generate trades.</div>';
      bars.innerHTML = empty; cards.innerHTML = ''; tab.innerHTML = '';
      return;
    }

    const grouped = groupByStrategy(rows);
    const pnlChart = '<h3 style="color:#8b949e;font-size:0.85em;margin-bottom:8px;">Total P&L by Strategy</h3>' +
      barChart(grouped, g => g.strategy, g => g.strategy, g => g.total_pnl, v => QA.fmt.money(v), g => g.total_pnl >= 0 ? '#3fb950' : '#f85149');
    const wrChart = '<h3 style="color:#8b949e;font-size:0.85em;margin:14px 0 8px;">Win Rate</h3>' +
      barChart(grouped, g => g.strategy, g => g.strategy, g => g.win_rate * 100, v => v.toFixed(1) + '%', '#58a6ff');
    const tradesChart = '<h3 style="color:#8b949e;font-size:0.85em;margin:14px 0 8px;">Trade Count</h3>' +
      barChart(grouped, g => g.strategy, g => g.strategy, g => g.total_trades, v => String(v), '#a371f7');
    bars.innerHTML = pnlChart + wrChart + tradesChart;

    let ch = '';
    for (const g of grouped){
      ch += '<div class="strategy-card">';
      ch += '<h3>' + g.strategy.replace(/_/g, ' ') + '</h3>';
      ch += '<div class="row"><span class="k">Total P&L</span><span class="' + QA.fmt.cls(g.total_pnl) + ' num">' + QA.fmt.money(g.total_pnl) + '</span></div>';
      ch += '<div class="row"><span class="k">Win rate</span><span class="num">' + (g.win_rate * 100).toFixed(1) + '%</span></div>';
      ch += '<div class="row"><span class="k">Trades</span><span class="num">' + g.total_trades + '</span></div>';
      ch += '<div class="row"><span class="k">Avg win</span><span class="positive num">' + QA.fmt.money(g.avg_win) + '</span></div>';
      ch += '<div class="row"><span class="k">Avg loss</span><span class="negative num">' + QA.fmt.money(g.avg_loss) + '</span></div>';
      ch += '<div class="row"><span class="k">Profit factor</span><span class="num">' + g.profit_factor.toFixed(2) + '</span></div>';
      ch += '</div>';
    }
    cards.innerHTML = ch;

    let th = '<table><thead><tr><th>Strategy</th><th>Market</th><th>TF</th><th class="num">Trades</th><th class="num">Win Rate</th><th class="num">Total P&L</th><th class="num">Avg Win</th><th class="num">Avg Loss</th><th class="num">PF</th></tr></thead><tbody>';
    for (const r of rows){
      const cls = QA.fmt.cls(r.total_pnl);
      th += '<tr><td><strong>' + r.strategy + '</strong></td><td>' + r.symbol + '</td><td>' + r.timeframe + '</td>';
      th += '<td class="num">' + r.total_trades + '</td><td class="num">' + (r.win_rate * 100).toFixed(1) + '%</td>';
      th += '<td class="num ' + cls + '">' + QA.fmt.money(r.total_pnl) + '</td>';
      th += '<td class="num positive">' + QA.fmt.money(r.avg_win) + '</td>';
      th += '<td class="num negative">' + QA.fmt.money(r.avg_loss) + '</td>';
      th += '<td class="num">' + r.profit_factor.toFixed(2) + '</td></tr>';
    }
    th += '</tbody></table>';
    tab.innerHTML = th;
  }
  load();
  setInterval(load, 30000);
  </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard/strategies")
def strategies_page():
    return render_template_string(
        STRATEGIES_HTML, base_css=_BASE_CSS, nav_html=_NAV_HTML, shared_js=_SHARED_JS
    )


# ───────────────────────────── Equity (/dashboard/equity) ─────────────────────────────

EQUITY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Equity · QuantAgent Terminal</title>
  {{ base_css|safe }}
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
  {{ nav_html|safe }}
  {{ shared_js|safe }}
  <main>
    <h1 class="page-title">Equity Curve</h1>
    <p class="page-sub">Portfolio value over time · drawdown visualization · risk ratios</p>

    <div class="panel">
      <div class="kv-grid" id="stats"></div>
    </div>

    <div class="panel">
      <h2>Portfolio Value</h2>
      <div id="eq-chart" style="height: 320px;"></div>
    </div>
    <div class="panel">
      <h2>Drawdown</h2>
      <div id="dd-chart" style="height: 220px;"></div>
    </div>
  </main>
  <script>
  function buildChart(el, fillColor, lineColor){
    return LightweightCharts.createChart(el, {
      layout: { background: { color: '#0d1117' }, textColor: '#c9d1d9' },
      grid: { vertLines: { color: '#161b22' }, horzLines: { color: '#161b22' } },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
      autoSize: true,
    });
  }

  const eqChart = buildChart(document.getElementById('eq-chart'));
  const ddChart = buildChart(document.getElementById('dd-chart'));
  const eqSeries = eqChart.addAreaSeries({
    topColor: 'rgba(88,166,255,0.4)', bottomColor: 'rgba(88,166,255,0.0)',
    lineColor: '#58a6ff', lineWidth: 2,
  });
  const ddSeries = ddChart.addAreaSeries({
    topColor: 'rgba(248,81,73,0.0)', bottomColor: 'rgba(248,81,73,0.4)',
    lineColor: '#f85149', lineWidth: 2,
  });

  function kv(label, value, cls){
    return '<div class="kv"><div class="k">' + label + '</div><div class="v ' + (cls||'') + '">' + value + '</div></div>';
  }

  async function load(){
    const data = await fetch('/api/equity-curve').then(r => r.json());
    const series = data.series || [];
    if (series.length < 1){
      document.getElementById('stats').innerHTML = '<div class="empty">No equity data yet. Snapshots are recorded as trades execute.</div>';
      eqSeries.setData([]); ddSeries.setData([]);
      return;
    }
    const eq = series.map(p => ({time: p.time, value: p.balance}));
    const dd = series.map(p => ({time: p.time, value: -Math.abs(p.drawdown_pct)}));
    eqSeries.setData(eq); ddSeries.setData(dd);
    eqChart.timeScale().fitContent(); ddChart.timeScale().fitContent();

    const s = data.stats || {};
    let html = '';
    html += kv('Current Value', QA.fmt.money(s.current_value));
    html += kv('Peak Value', QA.fmt.money(s.peak_value));
    html += kv('Total Return', QA.fmt.pct(s.total_return_pct), QA.fmt.cls(s.total_return_pct));
    html += kv('Max Drawdown', QA.fmt.pct(-Math.abs(s.max_drawdown_pct || 0)), 'negative');
    html += kv('Sharpe Ratio', (s.sharpe_ratio ?? 0).toFixed(2), (s.sharpe_ratio||0) > 0 ? 'positive' : 'negative');
    html += kv('Sortino Ratio', (s.sortino_ratio ?? 0).toFixed(2), (s.sortino_ratio||0) > 0 ? 'positive' : 'negative');
    html += kv('Profit Factor', (s.profit_factor ?? 0).toFixed(2));
    html += kv('Total Trades', s.total_trades ?? 0);
    document.getElementById('stats').innerHTML = html;
  }
  load();
  setInterval(load, 30000);
  </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard/equity")
def equity_page():
    return render_template_string(
        EQUITY_HTML, base_css=_BASE_CSS, nav_html=_NAV_HTML, shared_js=_SHARED_JS
    )


# ───────────────────────────── Journal (/dashboard/journal) ─────────────────────────────

JOURNAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Journal · QuantAgent Terminal — Trade Journal</title>
  {{ base_css|safe }}
</head>
<body>
  {{ nav_html|safe }}
  {{ shared_js|safe }}
  <main>
    <h1 class="page-title">Trade Journal</h1>
    <p class="page-sub">Full trade history · searchable · expandable reasoning · CSV export</p>

    <div class="panel">
      <div class="filters">
        <input id="q" placeholder="Search symbol / strategy / reasoning…" style="min-width:280px;">
        <label>Symbol</label>
        <select id="f-sym"><option value="">All</option></select>
        <label>Strategy</label>
        <select id="f-strat"><option value="">All</option></select>
        <label>Status</label>
        <select id="f-status">
          <option value="">All</option>
          <option>OPEN</option><option>CLOSED</option><option>STOPPED</option>
        </select>
        <button class="action" id="reload">Refresh</button>
        <button class="action primary" id="export">Export CSV</button>
      </div>
      <div id="trades-container"><div class="loading">Loading trades…</div></div>
    </div>
  </main>

  <script>
  let ALL = [];

  function filtered(){
    const q = document.getElementById('q').value.toLowerCase().trim();
    const fs = document.getElementById('f-sym').value;
    const fst = document.getElementById('f-strat').value;
    const fstat = document.getElementById('f-status').value;
    return ALL.filter(t => {
      if (fs && t.symbol !== fs) return false;
      if (fst && t.strategy !== fst) return false;
      if (fstat && t.status !== fstat) return false;
      if (q){
        const hay = (t.symbol + ' ' + t.strategy + ' ' + (t.agent_reasoning || '')).toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  function render(){
    const rows = filtered();
    const box = document.getElementById('trades-container');
    if (!rows.length){
      box.innerHTML = '<div class="empty">No trades match filters</div>';
      return;
    }
    let h = '<table><thead><tr><th>#</th><th>Time</th><th>Symbol</th><th>TF</th><th>Dir</th><th>Strategy</th>';
    h += '<th class="num">Entry</th><th class="num">Exit</th><th class="num">Size</th><th class="num">P&L</th><th class="num">%</th><th>Status</th></tr></thead><tbody>';
    for (const t of rows){
      const dirCls = t.direction === 'LONG' ? 'badge-long' : 'badge-short';
      const pnlCls = QA.fmt.cls(t.pnl || 0);
      const statCls = t.status === 'OPEN' ? 'badge-open' : (t.status === 'STOPPED' ? 'badge-stopped' : 'badge-closed');
      h += '<tr class="clickable" data-id="' + t.id + '"><td class="muted">' + t.id + '</td>';
      h += '<td class="muted">' + QA.fmt.shortTime(t.entry_time) + '</td>';
      h += '<td><strong>' + t.symbol + '</strong></td><td class="muted">' + t.timeframe + '</td>';
      h += '<td><span class="badge ' + dirCls + '">' + t.direction + '</span></td>';
      h += '<td class="muted">' + t.strategy + '</td>';
      h += '<td class="num">' + QA.fmt.money(t.entry_price) + '</td>';
      h += '<td class="num">' + (t.exit_price ? QA.fmt.money(t.exit_price) : '—') + '</td>';
      h += '<td class="num muted">' + QA.fmt.money(t.position_size, 0) + '</td>';
      h += '<td class="num ' + pnlCls + '">' + QA.fmt.money(t.pnl || 0) + '</td>';
      h += '<td class="num ' + pnlCls + '">' + QA.fmt.pct(t.pnl_pct || 0) + '</td>';
      h += '<td><span class="badge ' + statCls + '">' + t.status + '</span></td></tr>';
      h += '<tr class="expanded-row" id="exp-' + t.id + '" style="display:none"><td colspan="12">' + buildDetail(t) + '</td></tr>';
    }
    h += '</tbody></table>';
    box.innerHTML = h;

    box.querySelectorAll('tr.clickable').forEach(tr => {
      tr.addEventListener('click', () => {
        const id = tr.dataset.id;
        const exp = document.getElementById('exp-' + id);
        exp.style.display = exp.style.display === 'none' ? '' : 'none';
      });
    });
  }

  function buildDetail(t){
    const r = t.agent_reasoning ? '<div class="reasoning-box">' + (t.agent_reasoning) + '</div>' : '<div class="muted">No agent reasoning recorded.</div>';
    const tp = (t.take_profit ? QA.fmt.money(t.take_profit) : '—');
    const sl = (t.stop_loss ? QA.fmt.money(t.stop_loss) : '—');
    let detail = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:8px;">';
    detail += '<div><span class="k muted">Entry</span><div>' + QA.fmt.shortTime(t.entry_time) + '</div></div>';
    detail += '<div><span class="k muted">Exit</span><div>' + (t.exit_time ? QA.fmt.shortTime(t.exit_time) : '—') + '</div></div>';
    detail += '<div><span class="k muted">Stop</span><div>' + sl + '</div></div>';
    detail += '<div><span class="k muted">Take Profit</span><div>' + tp + '</div></div>';
    detail += '<div><span class="k muted">Quantity</span><div>' + (t.quantity || '—') + '</div></div>';
    detail += '</div>';
    detail += '<div class="k muted" style="text-transform:uppercase;font-size:0.7em;letter-spacing:0.5px;margin-top:6px;">Agent reasoning</div>';
    detail += r;
    return detail;
  }

  function toCSV(rows){
    const cols = ['id','entry_time','exit_time','symbol','timeframe','direction','strategy','entry_price','exit_price','position_size','quantity','stop_loss','take_profit','pnl','pnl_pct','status','agent_reasoning'];
    const esc = v => {
      if (v === null || v === undefined) return '';
      const s = String(v).replace(/"/g, '""').replace(/\\n/g, ' ');
      return /[,"]/.test(s) ? '"' + s + '"' : s;
    };
    const lines = [cols.join(',')];
    for (const r of rows) lines.push(cols.map(c => esc(r[c])).join(','));
    return lines.join('\\n');
  }

  async function load(){
    ALL = await fetch('/api/trades?limit=1000').then(r => r.json());
    const symSet = new Set(), stratSet = new Set();
    ALL.forEach(t => { symSet.add(t.symbol); stratSet.add(t.strategy); });
    const fs = document.getElementById('f-sym'); const fst = document.getElementById('f-strat');
    fs.innerHTML = '<option value="">All</option>' + Array.from(symSet).sort().map(s => '<option>' + s + '</option>').join('');
    fst.innerHTML = '<option value="">All</option>' + Array.from(stratSet).sort().map(s => '<option>' + s + '</option>').join('');
    render();
  }

  ['q','f-sym','f-strat','f-status'].forEach(id => document.getElementById(id).addEventListener('input', render));
  document.getElementById('reload').addEventListener('click', load);
  document.getElementById('export').addEventListener('click', () => {
    const rows = filtered();
    const blob = new Blob([toCSV(rows)], {type: 'text/csv'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'quantagent_trades_' + Date.now() + '.csv';
    a.click(); URL.revokeObjectURL(url);
  });
  load();
  </script>
</body>
</html>
"""


@dashboard_bp.route("/dashboard/journal")
def journal_page():
    return render_template_string(
        JOURNAL_HTML, base_css=_BASE_CSS, nav_html=_NAV_HTML, shared_js=_SHARED_JS
    )


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
def api_trades_dashboard():
    """Backwards-compatible alias for /api/trades."""
    return api_trades_list()


@dashboard_bp.route("/api/trades")
def api_trades_list():
    symbol = request.args.get("symbol")
    strategy = request.args.get("strategy")
    status = request.args.get("status")
    limit = request.args.get("limit", 200, type=int)

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
        if status:
            conditions.append("status = ?")
            params.append(status)
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


@dashboard_bp.route("/api/positions")
def api_positions_live():
    """Open positions enriched with live current price and unrealized P&L."""
    engine = _get_engine()
    positions = engine.get_open_positions()
    if not positions:
        return jsonify([])

    symbols = list({p["symbol"] for p in positions})
    live = _get_live_prices(symbols)

    out = []
    for p in positions:
        price = live["prices"].get(p["symbol"])
        u = _unrealized_pnl(p, price)
        enriched = dict(p)
        enriched["current_price"] = u["current_price"]
        enriched["unrealized_pnl"] = u["pnl"]
        enriched["unrealized_pnl_pct"] = u["pnl_pct"]
        out.append(enriched)
    return jsonify(out)


@dashboard_bp.route("/api/market-prices")
def api_market_prices():
    """Current price, 24h change, sparkline, and realized P&L for every tracked market."""
    engine = _get_engine()
    portfolios = {p["symbol"]: p for p in engine.get_all_portfolios()}
    open_counts: Dict[str, int] = {}
    for pos in engine.get_open_positions():
        open_counts[pos["symbol"]] = open_counts.get(pos["symbol"], 0) + 1

    symbols = list(MARKETS.keys())
    live = _get_live_prices(symbols)

    markets = []
    for sym, cfg in MARKETS.items():
        port = portfolios.get(sym, {})
        markets.append({
            "symbol": sym,
            "display_name": cfg.display_name,
            "category": cfg.category.value,
            "price": live["prices"].get(sym),
            "change_24h": live["change24h"].get(sym, 0.0),
            "volume_24h": live["volumes"].get(sym, 0.0),
            "spark": live["sparks"].get(sym, []),
            "realized_pnl": float(port.get("total_pnl") or 0.0),
            "current_balance": float(port.get("current_balance") or 0.0),
            "open_positions": open_counts.get(sym, 0),
        })

    return jsonify({"markets": markets, "fetched_at": time.time()})


@dashboard_bp.route("/api/candles/<path:symbol>")
def api_candles(symbol):
    """OHLC candles for a symbol, formatted for lightweight-charts."""
    interval = request.args.get("interval", "4h")
    lookback = request.args.get("lookback_days", 60, type=int)
    candles = _fetch_candles(symbol, interval=interval, lookback_days=lookback)
    return jsonify({"symbol": symbol, "interval": interval, "candles": candles})


def _compute_equity_stats(series: List[Dict[str, Any]], trades: List[Dict[str, Any]],
                         initial_balance: float) -> Dict[str, Any]:
    if not series:
        current = initial_balance
        peak = initial_balance
    else:
        balances = [p["balance"] for p in series]
        current = balances[-1]
        peak = max(balances)

    total_return_pct = ((current - initial_balance) / initial_balance * 100.0) if initial_balance else 0.0

    metrics = calculate_performance(trades, initial_balance=initial_balance)

    # Sortino ratio from closed-trade P&L
    pnls = [t.get("pnl", 0.0) for t in trades if t.get("pnl") is not None]
    sortino = 0.0
    if pnls:
        mean = sum(pnls) / len(pnls)
        downside = [p for p in pnls if p < 0]
        if downside:
            dd_std = math.sqrt(sum(p * p for p in downside) / len(downside))
            sortino = (mean / dd_std * math.sqrt(len(pnls))) if dd_std > 0 else 0.0

    max_dd_pct = 0.0
    if series:
        running_peak = series[0]["balance"]
        for s in series:
            running_peak = max(running_peak, s["balance"])
            if running_peak > 0:
                dd = (running_peak - s["balance"]) / running_peak * 100.0
                max_dd_pct = max(max_dd_pct, dd)

    return {
        "current_value": current,
        "peak_value": peak,
        "initial_value": initial_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct or (metrics.max_drawdown_pct * 100.0),
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": sortino,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
    }


@dashboard_bp.route("/api/equity-curve")
def api_equity_curve():
    """Aggregate portfolio value over time across every market."""
    engine = _get_engine()
    portfolios = engine.get_all_portfolios()
    initial_total = sum(p["initial_balance"] for p in portfolios)
    current_total = sum(p["current_balance"] for p in portfolios)

    # Pull snapshots and aggregate by snapshot_time
    snapshots = get_portfolio_snapshots(limit=5000)
    buckets: Dict[str, Dict[str, float]] = {}
    for s in snapshots:
        key = s["snapshot_time"]
        b = buckets.setdefault(key, {"balance": 0.0, "total_pnl": 0.0, "open": 0, "drawdown_pct": 0.0, "count": 0})
        b["balance"] += float(s.get("balance") or 0.0)
        b["total_pnl"] += float(s.get("total_pnl") or 0.0)
        b["open"] += int(s.get("open_positions") or 0)
        b["drawdown_pct"] = max(b["drawdown_pct"], float(s.get("drawdown_pct") or 0.0))
        b["count"] += 1

    series: List[Dict[str, Any]] = []
    for ts, b in sorted(buckets.items()):
        try:
            t = int(pd.Timestamp(ts).timestamp())
        except Exception:
            continue
        series.append({
            "time": t,
            "balance": b["balance"],
            "total_pnl": b["total_pnl"],
            "drawdown_pct": b["drawdown_pct"],
            "open_positions": b["open"],
        })

    # Anchor with initial balance if series empty / partial
    if not series:
        series = [{"time": int(time.time()) - 60, "balance": initial_total, "total_pnl": 0.0, "drawdown_pct": 0.0, "open_positions": 0}]
    # Always append the current aggregate as the rightmost point
    series.append({
        "time": int(time.time()),
        "balance": current_total,
        "total_pnl": current_total - initial_total,
        "drawdown_pct": series[-1]["drawdown_pct"] if series else 0.0,
        "open_positions": len(engine.get_open_positions()),
    })

    # Closed-trade performance for ratios
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status IN ('CLOSED', 'STOPPED') ORDER BY entry_time"
        ).fetchall()
        trades = [dict(r) for r in rows]

    stats = _compute_equity_stats(series, trades, initial_total)
    return jsonify({"series": series, "stats": stats, "initial_balance": initial_total})


@dashboard_bp.route("/api/trades.csv")
def api_trades_csv():
    """Download trades as CSV (server-rendered fallback)."""
    symbol = request.args.get("symbol")
    with get_connection() as conn:
        query = "SELECT * FROM trades"
        params: list = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY entry_time DESC"
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    buf = io.StringIO()
    cols = [
        "id", "entry_time", "exit_time", "symbol", "timeframe", "direction",
        "strategy", "entry_price", "exit_price", "position_size", "quantity",
        "stop_loss", "take_profit", "pnl", "pnl_pct", "status", "agent_reasoning",
    ]
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=quantagent_trades.csv"},
    )
