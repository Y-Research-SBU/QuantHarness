/* QuantAgent Dashboard — auto-refresh + chart rendering */
(function () {
  'use strict';

  const REFRESH_MS = 30_000;

  // ──────────────── helpers ────────────────
  const $ = (id) => document.getElementById(id);

  const fmtUSD = (v, decimals = 2) =>
    (v == null || Number.isNaN(v))
      ? '—'
      : '$' + Number(v).toLocaleString('en-US', {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        });

  const fmtPct = (v, decimals = 2) =>
    (v == null || Number.isNaN(v))
      ? '—'
      : `${Number(v).toFixed(decimals)}%`;

  const fmtNum = (v) => (v == null ? '—' : Number(v).toLocaleString('en-US'));

  const fmtTime = (iso) => {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString('en-US', { hour12: false });
  };

  const fmtDuration = (seconds) => {
    if (seconds == null || !isFinite(seconds)) return '—';
    const s = Math.max(0, Math.floor(seconds));
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (d) return `${d}d ${h}h`;
    if (h) return `${h}h ${m}m`;
    return `${m}m`;
  };

  const pnlClass = (v) => (Number(v) >= 0 ? 'pos' : 'neg');

  const signFmtUSD = (v) => {
    if (v == null) return '—';
    const n = Number(v);
    return (n >= 0 ? '+' : '') + fmtUSD(n);
  };

  const signFmtPct = (v, decimals = 2) => {
    if (v == null) return '—';
    const n = Number(v);
    return (n >= 0 ? '+' : '') + fmtPct(n, decimals);
  };

  async function fetchJSON(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }

  // ──────────────── chart cache (per symbol sparkline) ────────────────
  const sparkCharts = new Map();

  function renderSparkline(canvas, data, isPositive) {
    const color = isPositive ? '#10b981' : '#ef4444';
    const ctx = canvas.getContext('2d');

    const existing = sparkCharts.get(canvas.id);
    if (existing) existing.destroy();

    const labels = data.map((_, i) => i);
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data,
          borderColor: color,
          backgroundColor: color + '22',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { display: false },
          y: { display: false, beginAtZero: false },
        },
        elements: { line: { borderJoinStyle: 'round' } },
      },
    });
    sparkCharts.set(canvas.id, chart);
  }

  // ──────────────── overview ────────────────
  async function loadOverview() {
    try {
      const d = await fetchJSON('/api/overview');
      $('ov-balance').textContent = fmtUSD(d.total_balance);
      $('ov-pnl').textContent = signFmtUSD(d.total_pnl);
      $('ov-pnl').className = `text-2xl font-bold monospace ${pnlClass(d.total_pnl)}`;
      $('ov-pnl-pct').textContent = signFmtPct(d.total_pnl_pct);
      $('ov-pnl-pct').className = `text-xs monospace ${pnlClass(d.total_pnl)}`;
      $('ov-open').textContent = fmtNum(d.open_positions);
      $('ov-markets').textContent = fmtNum(d.markets_tracked);
      $('ov-signals').textContent = fmtNum(d.signals_last_hour);
      $('ov-uptime').textContent = fmtDuration(d.uptime_seconds);
      $('ov-last-scan').textContent = `last scan: ${fmtTime(d.last_scan_time)}`;
      $('overview-timestamp').textContent = fmtTime(d.current_time);

      // indicator bar: scale to pct, clamp to +/-20%
      const pct = Math.max(-20, Math.min(20, d.total_pnl_pct || 0));
      const width = Math.abs(pct) / 20 * 100;
      const bar = $('ov-indicator-bar');
      bar.style.width = `${width}%`;
      bar.className = `h-full transition-all ${d.is_profitable ? 'bg-emerald-500' : 'bg-red-500'}`;
      $('last-update').textContent = `updated ${fmtTime(d.current_time)}`;
    } catch (e) {
      console.error('overview load failed', e);
    }
  }

  // ──────────────── market grid ────────────────
  function makeMarketCard(m) {
    const profit = (m.total_pnl || 0) >= 0;
    const borderClass = m.circuit_breaker
      ? 'border-yellow-600/60'
      : (profit ? 'border-emerald-700/50' : 'border-red-800/50');
    const bgClass = m.circuit_breaker
      ? 'bg-yellow-950/20'
      : (profit ? 'bg-emerald-950/20' : 'bg-red-950/20');

    const signal = m.latest_signal;
    const sigHtml = signal
      ? `<div class="text-xs muted mt-1">
           Latest:
           <span class="${signal.direction === 'LONG' ? 'pos' : 'neg'} font-semibold">${signal.direction}</span>
           · ${signal.strategy}
           · <span class="monospace">${fmtTime(signal.entry_time)}</span>
         </div>`
      : `<div class="text-xs muted mt-1">No signals yet</div>`;

    const breakerBadge = m.circuit_breaker
      ? `<span class="text-xs px-2 py-0.5 rounded bg-yellow-900/60 text-yellow-300">🔴 BREAKER</span>`
      : '';

    const canvasId = `spark-${m.symbol.replace(/[^A-Za-z0-9]/g, '_')}`;

    return `
      <div class="rounded-lg p-3 border ${borderClass} ${bgClass}">
        <div class="flex items-start justify-between">
          <div>
            <div class="flex items-center gap-2">
              <span class="font-semibold">${m.symbol}</span>
              ${breakerBadge}
            </div>
            <div class="text-xs muted">${m.display_name} · ${m.category}</div>
          </div>
          <div class="text-right">
            <div class="monospace font-semibold ${pnlClass(m.total_pnl)}">${signFmtPct(m.pnl_pct)}</div>
            <div class="text-xs muted monospace">${fmtUSD(m.current_balance)}</div>
          </div>
        </div>
        <div class="mt-2">
          <canvas id="${canvasId}" class="sparkline" data-spark='${JSON.stringify(m.sparkline)}' data-profit="${profit}"></canvas>
        </div>
        <div class="flex justify-between text-xs mt-2">
          <div class="muted">${m.total_trades} trades · DD ${fmtPct(m.max_drawdown_pct, 1)}</div>
          <div class="${pnlClass(m.total_pnl)} monospace">${signFmtUSD(m.total_pnl)}</div>
        </div>
        ${sigHtml}
      </div>
    `;
  }

  async function loadMarkets() {
    try {
      const rows = await fetchJSON('/api/markets');
      $('grid-count').textContent = rows.length;
      const host = $('market-grid');
      host.innerHTML = rows.map(makeMarketCard).join('');
      // render sparklines
      rows.forEach((m) => {
        const canvasId = `spark-${m.symbol.replace(/[^A-Za-z0-9]/g, '_')}`;
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        renderSparkline(canvas, m.sparkline || [], (m.total_pnl || 0) >= 0);
      });
    } catch (e) {
      console.error('markets load failed', e);
    }
  }

  // ──────────────── strategy table ────────────────
  let strategyData = [];
  let sortCol = 'total_pnl';
  let sortDir = 'desc';

  function sortRows(rows) {
    const dir = sortDir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
  }

  function renderStrategyTable() {
    const tbody = $('strategy-tbody');
    const rows = sortRows(strategyData);
    if (rows.length === 0) {
      tbody.innerHTML = '';
      $('strategy-empty').classList.remove('hidden');
      return;
    }
    $('strategy-empty').classList.add('hidden');
    tbody.innerHTML = rows.map((r) => {
      let rowCls = 'border-b border-gray-800/50';
      if (r.is_best) rowCls += ' bg-emerald-950/30';
      else if (r.is_worst) rowCls += ' bg-red-950/30';
      return `
        <tr class="${rowCls}">
          <td class="py-2 px-3 font-semibold">${r.strategy}${r.is_best ? ' 🏆' : ''}${r.is_worst ? ' ⚠️' : ''}</td>
          <td class="py-2 px-3 text-right monospace">${fmtNum(r.total_trades)}</td>
          <td class="py-2 px-3 text-right monospace">${fmtPct(r.win_rate * 100, 1)}</td>
          <td class="py-2 px-3 text-right monospace ${pnlClass(r.total_pnl)}">${signFmtUSD(r.total_pnl)}</td>
          <td class="py-2 px-3 text-right monospace pos">${fmtUSD(r.avg_win)}</td>
          <td class="py-2 px-3 text-right monospace neg">${fmtUSD(r.avg_loss)}</td>
        </tr>
      `;
    }).join('');
  }

  async function loadStrategies() {
    try {
      strategyData = await fetchJSON('/api/strategies');
      renderStrategyTable();
    } catch (e) {
      console.error('strategies load failed', e);
    }
  }

  function wireSorting() {
    document.querySelectorAll('#strategy-table th.sortable').forEach((th) => {
      th.addEventListener('click', () => {
        const col = th.dataset.col;
        if (sortCol === col) {
          sortDir = sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          sortCol = col;
          sortDir = 'desc';
        }
        renderStrategyTable();
      });
    });
  }

  // ──────────────── trade log ────────────────
  async function loadTrades() {
    try {
      const rows = await fetchJSON('/api/trades?limit=50');
      const tbody = $('trade-tbody');
      if (rows.length === 0) {
        tbody.innerHTML = '';
        $('trade-empty').classList.remove('hidden');
        return;
      }
      $('trade-empty').classList.add('hidden');
      tbody.innerHTML = rows.map((t) => {
        const isPos = (t.pnl || 0) >= 0;
        const statusCls = {
          OPEN: 'text-sky-400',
          CLOSED: 'text-emerald-400',
          STOPPED: 'text-red-400',
        }[t.status] || 'muted';

        let rowBg = '';
        if (t.status === 'CLOSED') rowBg = isPos ? 'bg-emerald-950/20' : 'bg-red-950/20';
        else if (t.status === 'STOPPED') rowBg = 'bg-red-950/30';
        else if (t.status === 'OPEN') rowBg = 'bg-sky-950/10';

        return `
          <tr class="border-b border-gray-800/50 ${rowBg}">
            <td class="py-2 px-3 font-semibold">${t.symbol}</td>
            <td class="py-2 px-3 ${t.direction === 'LONG' ? 'pos' : 'neg'} font-semibold">${t.direction}</td>
            <td class="py-2 px-3 muted">${t.strategy}</td>
            <td class="py-2 px-3 text-right monospace">${fmtUSD(t.entry_price, 2)}</td>
            <td class="py-2 px-3 text-right monospace">${t.exit_price ? fmtUSD(t.exit_price, 2) : '—'}</td>
            <td class="py-2 px-3 text-right monospace ${pnlClass(t.pnl)}">${signFmtUSD(t.pnl)}</td>
            <td class="py-2 px-3 ${statusCls} text-xs font-semibold">${t.status}</td>
            <td class="py-2 px-3 text-xs muted monospace">${fmtTime(t.entry_time)}</td>
          </tr>
        `;
      }).join('');
    } catch (e) {
      console.error('trades load failed', e);
    }
  }

  // ──────────────── scanner status ────────────────
  async function loadScanner() {
    try {
      const d = await fetchJSON('/api/scanner');
      $('sc-last-scan').textContent = fmtTime(d.last_scan_time);
      $('sc-snap-hour').textContent = fmtNum(d.snapshots_last_hour);
      $('sc-sig-hour').textContent = fmtNum(d.signals_last_hour);
      $('sc-markets-hour').textContent = (d.markets_scanned_last_hour || []).length;
      $('sc-total-snap').textContent = fmtNum(d.total_snapshots);

      const host = $('sc-recent');
      if (!d.recent_scans || d.recent_scans.length === 0) {
        host.innerHTML = '<div class="muted">No recent scans.</div>';
        return;
      }
      host.innerHTML = d.recent_scans.slice(0, 12).map((s) => `
        <div class="flex justify-between">
          <span>${s.symbol}</span>
          <span class="muted">${fmtTime(s.last_scan)}</span>
        </div>
      `).join('');
    } catch (e) {
      console.error('scanner load failed', e);
    }
  }

  // ──────────────── backtests ────────────────
  async function loadBacktests() {
    try {
      const rows = await fetchJSON('/api/backtests');
      const tbody = $('backtest-tbody');
      if (!rows || rows.length === 0) {
        tbody.innerHTML = '';
        $('backtest-empty').classList.remove('hidden');
        $('bt-file').textContent = '—';
        return;
      }
      $('backtest-empty').classList.add('hidden');
      $('bt-file').textContent = rows[0].file + ' · ' + fmtTime(rows[0].generated_at);

      tbody.innerHTML = rows.map((b) => `
        <tr class="border-b border-gray-800/50">
          <td class="py-2 px-3 font-semibold">${b.symbol || '—'}</td>
          <td class="py-2 px-3 muted">${b.strategy || '—'}</td>
          <td class="py-2 px-3 muted">${b.timeframe || '—'}</td>
          <td class="py-2 px-3 text-right monospace ${pnlClass(b.total_return_pct)}">${signFmtPct(b.total_return_pct, 2)}</td>
          <td class="py-2 px-3 text-right monospace">${Number(b.sharpe_ratio || 0).toFixed(2)}</td>
          <td class="py-2 px-3 text-right monospace neg">${fmtPct(b.max_drawdown_pct, 2)}</td>
          <td class="py-2 px-3 text-right monospace">${fmtPct((b.win_rate || 0) * 100, 1)}</td>
          <td class="py-2 px-3 text-right monospace">${fmtNum(b.total_trades)}</td>
        </tr>
      `).join('');
    } catch (e) {
      console.error('backtests load failed', e);
    }
  }

  // ──────────────── orchestration ────────────────
  async function loadPositions() {
    try {
      const rows = await fetchJSON('/api/positions');
      const tbody = $('positions-tbody');
      const empty = $('positions-empty');
      const count = $('positions-count');
      if (!tbody) return;

      if (!rows || rows.length === 0) {
        tbody.innerHTML = '';
        if (empty) empty.classList.remove('hidden');
        if (count) count.textContent = '0 positions';
        return;
      }
      if (empty) empty.classList.add('hidden');
      if (count) count.textContent = `${rows.length} positions`;

      tbody.innerHTML = rows.map(p => {
        const dirCls = p.direction === 'LONG' ? 'text-green-400' : 'text-red-400';
        const pnlCls = p.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400';
        const rrCls = (p.risk_reward && p.risk_reward >= 1.5) ? 'text-green-400' : 
                      (p.risk_reward && p.risk_reward < 1.0) ? 'text-red-400' : 'text-yellow-400';

        const fmtPrice = (v) => v != null ? Number(v).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 6}) : '—';

        const slLabel = p.stop_loss_pct != null ? `${fmtPrice(p.stop_loss)} (${p.stop_loss_pct > 0 ? '+' : ''}${p.stop_loss_pct}%)` : '—';
        const tpLabel = p.take_profit_pct != null ? `${fmtPrice(p.take_profit)} (${p.take_profit_pct > 0 ? '+' : ''}${p.take_profit_pct}%)` : '—';

        return `<tr class="border-b border-gray-800/50 hover:bg-white/5">
          <td class="py-2 px-2 font-medium">${p.display_name || p.symbol}</td>
          <td class="py-2 px-2 ${dirCls} font-bold">${p.direction}</td>
          <td class="py-2 px-2 text-xs muted">${p.strategy}</td>
          <td class="py-2 px-2 text-right">${fmtPrice(p.entry_price)}</td>
          <td class="py-2 px-2 text-right font-medium">${fmtPrice(p.current_price)}</td>
          <td class="py-2 px-2 text-right text-red-400 text-xs">${slLabel}</td>
          <td class="py-2 px-2 text-right text-green-400 text-xs">${tpLabel}</td>
          <td class="py-2 px-2 text-right ${rrCls} font-bold">${p.risk_reward != null ? '1:' + p.risk_reward : '—'}</td>
          <td class="py-2 px-2 text-right ${pnlCls} font-medium">${fmtUSD(p.unrealized_pnl)} (${p.unrealized_pct > 0 ? '+' : ''}${p.unrealized_pct}%)</td>
          <td class="py-2 px-2 text-right text-xs">${fmtUSD(p.position_size)}</td>
          <td class="py-2 px-2 text-xs muted max-w-[200px] truncate" title="${(p.rationale || '').replace(/"/g, '&quot;')}">${p.rationale || '—'}</td>
        </tr>`;
      }).join('');
    } catch (e) {
      console.warn('positions load failed', e);
    }
  }

  async function refreshAll() {
    await Promise.all([
      loadOverview(),
      loadMarkets(),
      loadStrategies(),
      loadPositions(),
      loadTrades(),
      loadScanner(),
      loadBacktests(),
    ]);
  }

  function init() {
    wireSorting();
    refreshAll();
    setInterval(refreshAll, REFRESH_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
