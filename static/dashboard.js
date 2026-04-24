/* QuantAgent Dashboard — auto-refresh + chart rendering + live WS prices */
(function () {
  'use strict';

  // Snapshot refresh (DB-backed data: trades, overview, strategies, …).
  // Prices are pushed live via socket.io when available.
  const SNAPSHOT_REFRESH_MS = 30_000;
  // Fallback REST price poll when the socket is down.
  const PRICE_FALLBACK_MS = 10_000;

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

  const fmtPrice = (v) => v == null || Number.isNaN(Number(v))
    ? '—'
    : Number(v).toLocaleString('en-US', {
        minimumFractionDigits: 2, maximumFractionDigits: 6,
      });

  const safeId = (s) => String(s).replace(/[^A-Za-z0-9]/g, '_');

  async function fetchJSON(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }

  function flashElement(el, isUp) {
    if (!el) return;
    const cls = isUp ? 'flash-up' : 'flash-down';
    el.classList.remove('flash-up', 'flash-down');
    // Force reflow to restart animation even if class is re-added quickly.
    void el.offsetWidth;
    el.classList.add(cls);
  }

  // ──────────────── live-price state ────────────────
  // Authoritative latest-price cache, keyed by symbol. Seeded from the REST
  // snapshot at load, updated by every `price_update` WS message.
  const livePrices = {};
  // Mutable per-position/per-card state used to refresh dependent cells.
  const positionState = new Map();  // trade_id → { symbol, direction, entry, qty, size, el refs }
  // Equity = initial + realized + unrealized (positionState); we keep the
  // server values around and rederive the derived cells whenever a price
  // changes.
  let realizedPnl = 0;
  let totalInitial = 0;

  function computePositionPnl(entry, current, qty, direction) {
    if (entry == null || current == null || qty == null) return 0;
    return direction === 'SHORT'
      ? (Number(entry) - Number(current)) * Number(qty)
      : (Number(current) - Number(entry)) * Number(qty);
  }

  function applyLivePrice(symbol, price) {
    const prev = livePrices[symbol];
    livePrices[symbol] = price;
    const isUp = prev == null ? true : price >= prev;

    // Update Market Grid cards for this symbol (both in "My Positions" and
    // inside category sections share element IDs that include the symbol).
    document.querySelectorAll(`[data-live-price="${symbol}"]`).forEach((el) => {
      el.textContent = fmtPrice(price);
      if (prev != null && prev !== price) flashElement(el, isUp);
    });

    // Update positions (open-positions table + My Positions cards).
    let positionsPnlChanged = false;
    positionState.forEach((state) => {
      if (state.symbol !== symbol) return;
      const pnl = computePositionPnl(state.entry, price, state.qty, state.direction);
      const pct = state.size > 0 ? (pnl / state.size) * 100 : 0;
      state.currentPrice = price;
      state.unrealizedPnl = pnl;

      if (state.priceEls) state.priceEls.forEach((el) => {
        el.textContent = fmtPrice(price);
        if (prev != null && prev !== price) flashElement(el, isUp);
      });
      if (state.pnlEls) state.pnlEls.forEach((el) => {
        el.textContent = signFmtUSD(pnl);
        el.className = state.pnlBaseClass + ' ' + pnlClass(pnl);
      });
      if (state.pnlPctEls) state.pnlPctEls.forEach((el) => {
        el.textContent = signFmtPct(pct);
        el.className = state.pnlPctBaseClass + ' ' + pnlClass(pnl);
      });
      if (state.borderEl) {
        state.borderEl.classList.remove('pos-border-profit', 'pos-border-loss');
        state.borderEl.classList.add(pnl >= 0 ? 'pos-border-profit' : 'pos-border-loss');
      }
      positionsPnlChanged = true;
    });

    if (positionsPnlChanged) {
      refreshPositionsTotals();
      refreshOverviewEquity();
    }
  }

  function refreshPositionsTotals() {
    let totalPnl = 0;
    let totalExposure = 0;
    positionState.forEach((s) => {
      totalPnl += Number(s.unrealizedPnl || 0);
      totalExposure += Number(s.size || 0);
    });
    const pnlEl = $('my-positions-pnl');
    const expEl = $('my-positions-exposure');
    if (pnlEl) {
      pnlEl.textContent = signFmtUSD(totalPnl);
      pnlEl.className = `monospace ${pnlClass(totalPnl)}`;
    }
    if (expEl) expEl.textContent = fmtUSD(totalExposure);
  }

  function refreshOverviewEquity() {
    let unrealized = 0;
    positionState.forEach((s) => { unrealized += Number(s.unrealizedPnl || 0); });
    const totalPnl = realizedPnl + unrealized;
    const equity = totalInitial + totalPnl;
    const pnlPct = totalInitial > 0 ? (totalPnl / totalInitial) * 100 : 0;

    if ($('ov-balance')) $('ov-balance').textContent = fmtUSD(equity);
    if ($('ov-pnl')) {
      $('ov-pnl').textContent = signFmtUSD(totalPnl);
      $('ov-pnl').className = `text-2xl font-bold monospace ${pnlClass(totalPnl)}`;
    }
    if ($('ov-pnl-pct')) {
      $('ov-pnl-pct').textContent = signFmtPct(pnlPct);
      $('ov-pnl-pct').className = `text-xs monospace ${pnlClass(totalPnl)}`;
    }

    const bar = $('ov-indicator-bar');
    if (bar) {
      const pct = Math.max(-20, Math.min(20, pnlPct || 0));
      bar.style.width = `${Math.abs(pct) / 20 * 100}%`;
      bar.className = `h-full transition-all ${totalPnl >= 0 ? 'bg-emerald-500' : 'bg-red-500'}`;
    }
  }

  // ──────────────── overview ────────────────
  async function loadOverview() {
    try {
      const d = await fetchJSON('/api/overview');
      realizedPnl = Number(d.realized_pnl || 0);
      totalInitial = Number(d.total_initial || 0);

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

  // ──────────────── market categories (grid) ────────────────
  function marketCardHTML(m) {
    const hasPos = (m.open_positions || 0) > 0;
    const profit = (m.total_pnl || 0) >= 0;
    const bgClass = m.circuit_breaker
      ? 'bg-yellow-950/20 border-yellow-600/60'
      : (profit ? 'border-emerald-700/50 bg-emerald-950/10' : 'border-red-800/40 bg-red-950/10');
    const posBorder = hasPos ? (profit ? 'pos-border-profit' : 'pos-border-loss') : '';
    const posDot = hasPos ? '<span class="pos-dot" title="open position"></span>' : '';

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

    const tvSymbol = m.tv_symbol || m.symbol;
    const sid = safeId(m.symbol);
    const miniChartUrl = `https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(tvSymbol)}&interval=60&timezone=America%2FNew_York&theme=dark&style=3&locale=en&hide_top_toolbar=1&hide_legend=1&save_image=0&hide_volume=1&backgroundColor=rgba(0,0,0,0)`;

    const seedPrice = livePrices[m.symbol];
    const priceLabel = seedPrice != null ? fmtPrice(seedPrice) : '—';

    return `
      <div id="card-${sid}" class="rounded-lg p-3 border ${bgClass} ${posBorder} cursor-pointer hover:border-gray-500 transition-colors"
           onclick="openChart('${m.symbol}', '${tvSymbol}', '${(m.display_name || '').replace(/'/g, '&#39;')}', '${signal ? signal.direction : ''}', '${signal ? signal.entry_price : 0}', '0', '0')">
        <div class="flex items-start justify-between">
          <div>
            <div class="flex items-center gap-2">
              <span class="font-semibold">${m.symbol}</span>
              ${posDot}
              ${breakerBadge}
            </div>
            <div class="text-xs muted">${m.display_name} · ${m.category}</div>
          </div>
          <div class="text-right">
            <div class="text-xs muted">Price</div>
            <div class="monospace font-semibold" data-live-price="${m.symbol}">${priceLabel}</div>
          </div>
        </div>
        <div class="mt-2 rounded overflow-hidden" style="height:120px;">
          <iframe src="${miniChartUrl}" style="width:100%;height:100%;border:none;pointer-events:none;" allowtransparency="true" frameborder="0" loading="lazy"></iframe>
        </div>
        <div class="flex justify-between text-xs mt-2">
          <div id="card-info-${sid}" class="muted">${m.total_trades} trades · ${m.open_positions || 0} open</div>
          <div id="card-pnl-${sid}" class="${pnlClass(m.total_pnl)} monospace">${signFmtUSD(m.total_pnl)}</div>
        </div>
        ${sigHtml}
      </div>
    `;
  }

  function positionCardHTML(p) {
    const profit = (p.unrealized_pnl || 0) >= 0;
    const border = profit ? 'pos-border-profit' : 'pos-border-loss';
    const dirCls = p.direction === 'LONG' ? 'text-green-400' : 'text-red-400';
    const dirArrow = p.direction === 'LONG' ? '↑' : '↓';
    const tvSymbol = p.tv_symbol || p.symbol;
    const tid = `t${p.trade_id || safeId(p.symbol)}_${safeId(p.direction)}`;
    const miniChartUrl = `https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(tvSymbol)}&interval=60&timezone=America%2FNew_York&theme=dark&style=3&locale=en&hide_top_toolbar=1&hide_legend=1&save_image=0&hide_volume=1&backgroundColor=rgba(0,0,0,0)`;

    return `
      <div id="pos-card-${tid}" class="rounded-lg p-3 border bg-[#0f172a] border-gray-700 ${border} cursor-pointer hover:border-gray-500 transition-colors"
           onclick="openChart('${p.symbol}', '${tvSymbol}', '${(p.display_name || '').replace(/'/g, '&#39;')}', '${p.direction}', '${p.entry_price || 0}', '${p.stop_loss || 0}', '${p.take_profit || 0}')">
        <div class="flex items-start justify-between">
          <div>
            <div class="flex items-center gap-2">
              <span class="font-semibold">${p.symbol}</span>
              <span class="${dirCls} font-bold text-sm">${dirArrow} ${p.direction}</span>
            </div>
            <div class="text-xs muted">${p.display_name} · ${p.strategy || ''}</div>
          </div>
          <div class="text-right">
            <div class="text-xs muted">Current</div>
            <div class="monospace font-semibold" id="pos-price-${tid}" data-live-price="${p.symbol}">${fmtPrice(p.current_price)}</div>
          </div>
        </div>
        <div class="mt-2 rounded overflow-hidden" style="height:110px;">
          <iframe src="${miniChartUrl}" style="width:100%;height:100%;border:none;pointer-events:none;" allowtransparency="true" frameborder="0" loading="lazy"></iframe>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-2 text-xs">
          <div>
            <div class="muted">Entry</div>
            <div class="monospace">${fmtPrice(p.entry_price)}</div>
          </div>
          <div>
            <div class="muted">Size</div>
            <div class="monospace">${fmtUSD(p.position_size)}</div>
          </div>
          <div class="text-right">
            <div class="muted">Unrealized</div>
            <div id="pos-pnl-${tid}" class="monospace font-bold ${pnlClass(p.unrealized_pnl)}">${signFmtUSD(p.unrealized_pnl)}</div>
            <div id="pos-pct-${tid}" class="monospace ${pnlClass(p.unrealized_pnl)}">${signFmtPct(p.unrealized_pct)}</div>
          </div>
        </div>
      </div>
    `;
  }

  function sectionHTML(section, collapsed) {
    const sid = safeId(section.id);
    const expanded = !collapsed;
    return `
      <div id="sec-${sid}" class="rounded-lg border border-gray-800 bg-[#0e1524]" data-section="${section.id}">
        <button type="button" class="w-full flex items-center justify-between px-4 py-3 hover:bg-white/5" data-toggle="${sid}">
          <div class="flex items-center gap-2">
            <span class="chev ${expanded ? 'open' : ''}" id="chev-${sid}"></span>
            <span class="text-sm font-semibold">${section.display_name}</span>
            <span class="text-xs muted">${section.count} markets</span>
          </div>
          <div class="text-xs muted flex items-center gap-4">
            <span>${section.with_signals} with signals</span>
            <span>${section.with_positions} positions open</span>
            <span class="monospace ${pnlClass(section.unrealized_pnl)}">${signFmtUSD(section.unrealized_pnl)}</span>
          </div>
        </button>
        <div id="sec-body-${sid}" class="${expanded ? '' : 'hidden'} p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3"></div>
      </div>
    `;
  }

  // Persist collapsed/expanded state across refreshes and page loads.
  const SECTION_KEY = 'qa.sections.collapsed';
  function loadCollapsedSet() {
    try {
      const raw = localStorage.getItem(SECTION_KEY);
      if (!raw) return new Set();
      return new Set(JSON.parse(raw));
    } catch (e) { return new Set(); }
  }
  function saveCollapsedSet(set) {
    try { localStorage.setItem(SECTION_KEY, JSON.stringify([...set])); } catch (e) {}
  }

  let _sectionsRendered = false;
  let _lastSectionsFingerprint = '';
  let _lastPositionsFingerprint = '';

  async function loadMarkets() {
    try {
      const data = await fetchJSON('/api/market_categories');
      // ── My Positions ──
      const posSection = $('my-positions-section');
      const posGrid = $('my-positions-grid');
      const posCount = $('my-positions-count');

      // Rebuild positionState to match the server's truth. We always rebuild
      // here (incremental edits are fragile when trades open/close).
      positionState.clear();

      const cards = data.positions?.cards || [];
      const posFingerprint = cards.map(p => `${p.trade_id}:${p.direction}:${p.entry_price}`).join('|');
      if (cards.length === 0) {
        posSection.classList.add('hidden');
        posGrid.innerHTML = '';
        posCount.textContent = '0 positions';
      } else {
        posSection.classList.remove('hidden');
        posCount.textContent = `${cards.length} position${cards.length === 1 ? '' : 's'}`;
        if (posFingerprint !== _lastPositionsFingerprint) {
          posGrid.innerHTML = cards.map(positionCardHTML).join('');
          _lastPositionsFingerprint = posFingerprint;
        }
      }

      // Record state refs for each position so WS updates can mutate.
      cards.forEach((p) => {
        const tid = `t${p.trade_id || safeId(p.symbol)}_${safeId(p.direction)}`;
        const borderEl = document.getElementById(`pos-card-${tid}`);
        const priceEls = Array.from(
          document.querySelectorAll(`#pos-price-${tid}`)
        );
        const pnlEl = document.getElementById(`pos-pnl-${tid}`);
        const pctEl = document.getElementById(`pos-pct-${tid}`);
        // Also include the corresponding open-positions table cells so they
        // tick in sync with the cards.
        const rowPriceEls = Array.from(
          document.querySelectorAll(`[data-row-price="${p.trade_id}"]`)
        );
        const rowPnlEls = Array.from(
          document.querySelectorAll(`[data-row-pnl="${p.trade_id}"]`)
        );
        positionState.set(p.trade_id, {
          symbol: p.symbol,
          direction: p.direction,
          entry: Number(p.entry_price),
          qty: Number(p.quantity),
          size: Number(p.position_size),
          unrealizedPnl: Number(p.unrealized_pnl || 0),
          currentPrice: Number(p.current_price),
          priceEls: [...priceEls, ...rowPriceEls],
          pnlEls: pnlEl ? [pnlEl, ...rowPnlEls] : rowPnlEls,
          pnlBaseClass: 'monospace font-bold',
          pnlPctEls: pctEl ? [pctEl] : [],
          pnlPctBaseClass: 'monospace',
          borderEl,
        });
        if (p.current_price != null && !(p.symbol in livePrices)) {
          livePrices[p.symbol] = Number(p.current_price);
        }
      });

      // ── My Positions totals ──
      $('my-positions-exposure').textContent = fmtUSD(data.positions?.total_exposure || 0);
      const totalUnreal = Number(data.positions?.unrealized_pnl || 0);
      const pnlEl = $('my-positions-pnl');
      pnlEl.textContent = signFmtUSD(totalUnreal);
      pnlEl.className = `monospace ${pnlClass(totalUnreal)}`;

      // ── Category sections ──
      const sections = data.sections || [];
      const totalMarkets = sections.reduce((acc, s) => acc + (s.count || 0), 0);
      $('grid-count').textContent = totalMarkets;

      const host = $('market-sections');
      const fingerprint = sections.map(
        (s) => `${s.id}:${s.markets.map(m => m.symbol).join(',')}`
      ).join('|');

      const collapsed = loadCollapsedSet();

      if (!_sectionsRendered || fingerprint !== _lastSectionsFingerprint) {
        // On first render default every section to collapsed per spec.
        if (!_sectionsRendered && collapsed.size === 0) {
          sections.forEach((s) => collapsed.add(s.id));
          saveCollapsedSet(collapsed);
        }
        host.innerHTML = sections.map((s) => sectionHTML(s, collapsed.has(s.id))).join('');
        // Populate bodies of expanded sections with the market cards.
        sections.forEach((s) => {
          const sid = safeId(s.id);
          const body = document.getElementById(`sec-body-${sid}`);
          if (body && !collapsed.has(s.id)) {
            body.innerHTML = s.markets.map(marketCardHTML).join('');
          } else if (body) {
            // Stash HTML as a data-src so we can render on expand lazily.
            body.dataset.html = s.markets.map(marketCardHTML).join('');
          }
        });
        wireSectionToggles(sections);
        _sectionsRendered = true;
        _lastSectionsFingerprint = fingerprint;
      } else {
        // Incremental update — refresh only the summary stats + pnl deltas.
        sections.forEach((s) => {
          const sid = safeId(s.id);
          const body = document.getElementById(`sec-body-${sid}`);
          if (!body) return;
          if (!collapsed.has(s.id) && body.children.length) {
            // Update only card text children (prices flow through WS).
            s.markets.forEach((m) => {
              const infoEl = document.getElementById(`card-info-${safeId(m.symbol)}`);
              const pnlEl = document.getElementById(`card-pnl-${safeId(m.symbol)}`);
              if (infoEl) infoEl.textContent = `${m.total_trades} trades · ${m.open_positions || 0} open`;
              if (pnlEl) {
                pnlEl.textContent = signFmtUSD(m.total_pnl);
                pnlEl.className = `${pnlClass(m.total_pnl)} monospace`;
              }
            });
          }
        });
      }

      // Now that positionState reflects the server, recompute live overview.
      refreshOverviewEquity();
    } catch (e) {
      console.error('market categories load failed', e);
    }
  }

  function wireSectionToggles(sections) {
    document.querySelectorAll('[data-toggle]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const sid = btn.dataset.toggle;
        const body = document.getElementById(`sec-body-${sid}`);
        const chev = document.getElementById(`chev-${sid}`);
        if (!body) return;
        const isHidden = body.classList.contains('hidden');
        if (isHidden) {
          // Expanding — lazy-render the cards if needed.
          if (body.dataset.html && !body.children.length) {
            body.innerHTML = body.dataset.html;
          }
          body.classList.remove('hidden');
          chev && chev.classList.add('open');
        } else {
          body.classList.add('hidden');
          chev && chev.classList.remove('open');
        }

        // Persist collapsed state by section id (NOT safeId).
        const sectionId = btn.closest('[data-section]')?.dataset.section || sid;
        const set = loadCollapsedSet();
        if (isHidden) set.delete(sectionId); else set.add(sectionId);
        saveCollapsedSet(set);
      });
    });
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

  // ──────────────── open positions table ────────────────
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

        const slLabel = p.stop_loss_pct != null ? `${fmtPrice(p.stop_loss)} (${p.stop_loss_pct > 0 ? '+' : ''}${p.stop_loss_pct}%)` : '—';
        const tpLabel = p.take_profit_pct != null ? `${fmtPrice(p.take_profit)} (${p.take_profit_pct > 0 ? '+' : ''}${p.take_profit_pct}%)` : '—';

        return `<tr class="border-b border-gray-800/50 hover:bg-white/5 cursor-pointer" onclick="openChart('${p.symbol}', '${p.tv_symbol || p.symbol}', '${p.display_name || p.symbol}', '${p.direction}', '${p.entry_price}', '${p.stop_loss || 0}', '${p.take_profit || 0}')">
          <td class="py-2 px-2 font-medium">
            <span class="underline decoration-dotted">${p.display_name || p.symbol}</span>
            <span class="text-xs muted ml-1">📈</span>
          </td>
          <td class="py-2 px-2 ${dirCls} font-bold">${p.direction}</td>
          <td class="py-2 px-2 text-xs muted">${p.strategy}</td>
          <td class="py-2 px-2 text-right">${fmtPrice(p.entry_price)}</td>
          <td class="py-2 px-2 text-right font-medium" data-row-price="${p.id}">${fmtPrice(p.current_price)}</td>
          <td class="py-2 px-2 text-right text-red-400 text-xs">${slLabel}</td>
          <td class="py-2 px-2 text-right text-green-400 text-xs">${tpLabel}</td>
          <td class="py-2 px-2 text-right ${rrCls} font-bold">${p.risk_reward != null ? '1:' + p.risk_reward : '—'}</td>
          <td class="py-2 px-2 text-right ${pnlCls} font-medium" data-row-pnl="${p.id}">${fmtUSD(p.unrealized_pnl)} (${p.unrealized_pct > 0 ? '+' : ''}${p.unrealized_pct}%)</td>
          <td class="py-2 px-2 text-right text-xs">${fmtUSD(p.position_size)}</td>
          <td class="py-2 px-2 text-xs muted max-w-[200px] truncate" title="${(p.rationale || '').replace(/"/g, '&quot;')}">${p.rationale || '—'}</td>
        </tr>`;
      }).join('');
    } catch (e) {
      console.warn('positions load failed', e);
    }
  }

  // ──────────────── live indicator ────────────────
  function setLiveIndicator(connected) {
    const dot = $('live-dot');
    const label = $('live-label');
    if (!dot || !label) return;
    if (connected) {
      dot.classList.remove('offline');
      label.textContent = 'LIVE';
      label.className = 'text-emerald-400 font-semibold';
    } else {
      dot.classList.add('offline');
      label.textContent = 'OFFLINE';
      label.className = 'muted';
    }
  }

  // ──────────────── WS wiring (with REST fallback) ────────────────
  let socket = null;
  let fallbackTimer = null;

  function startRESTFallback() {
    if (fallbackTimer) return;
    fallbackTimer = setInterval(async () => {
      try {
        const data = await fetchJSON('/api/prices');
        if (data && data.prices) {
          Object.entries(data.prices).forEach(([s, p]) => applyLivePrice(s, Number(p)));
        }
      } catch (e) { /* ignore */ }
    }, PRICE_FALLBACK_MS);
  }
  function stopRESTFallback() {
    if (fallbackTimer) {
      clearInterval(fallbackTimer);
      fallbackTimer = null;
    }
  }

  function initSocket() {
    if (typeof io === 'undefined') {
      // socket.io client not loaded — rely on REST fallback.
      setLiveIndicator(false);
      startRESTFallback();
      return;
    }
    socket = io({
      reconnection: true,
      reconnectionDelay: 500,
      reconnectionDelayMax: 5_000,
      transports: ['websocket', 'polling'],
    });

    socket.on('connect', () => {
      setLiveIndicator(true);
      stopRESTFallback();
    });
    socket.on('disconnect', () => {
      setLiveIndicator(false);
      startRESTFallback();
    });
    socket.on('connect_error', () => {
      setLiveIndicator(false);
      startRESTFallback();
    });
    socket.on('price_snapshot', (data) => {
      if (!data || !data.prices) return;
      Object.entries(data.prices).forEach(([s, p]) => applyLivePrice(s, Number(p)));
    });
    socket.on('price_update', (data) => {
      if (!data || !data.symbol || data.price == null) return;
      applyLivePrice(data.symbol, Number(data.price));
    });
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
    // Seed prices from REST before sockets connect so the grid isn't blank.
    fetchJSON('/api/prices').then((data) => {
      if (data && data.prices) {
        Object.entries(data.prices).forEach(([s, p]) => { livePrices[s] = Number(p); });
        if (data.ws_connected) setLiveIndicator(true);
      }
    }).catch(() => {});
    refreshAll();
    setInterval(refreshAll, SNAPSHOT_REFRESH_MS);
    initSocket();
  }

  // Expose test hooks (no-ops in production UI, used by unit tests).
  window.__qa = {
    applyLivePrice,
    livePrices,
    positionState,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
