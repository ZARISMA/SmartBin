/* HexaBin Analytics view
 * Every widget reads real data from /api/analytics?period=… — KPIs with
 * previous-period deltas, per-bucket category series, material mix, bin
 * leaderboard, LLM backend mix — plus /api/entries for the recent table.
 */
(function () {
    'use strict';

    const CAT = ['Plastic', 'Paper', 'Glass', 'Organic', 'Aluminum', 'Other'];
    const COLORS = {
        Plastic:  '#87CEEB',
        Paper:    '#D2B48C',
        Glass:    '#40E0D0',
        Organic:  '#1E4D2B',
        Aluminum: '#A9A9A9',
        Other:    '#9370DB',
        Empty:    '#8C8C8C',
    };

    let period = '7d';

    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    }

    // ── Fetch ─────────────────────────────────────────────────────────
    async function fetchJSON(url) {
        try {
            const r = await fetch(url, { cache: 'no-store' });
            if (!r.ok) {
                if (r.status === 401) { window.location.href = '/login'; }
                return null;
            }
            return await r.json();
        } catch (e) { return null; }
    }

    // ── KPI strip ─────────────────────────────────────────────────────
    const KPI_DEFS = [
        {
            key: 'total', el: 'kpi-total',
            fmt: (v) => (v == null ? '—' : v.toLocaleString()),
            deltaFmt: (d) => `${d > 0 ? '+' : ''}${d}%`,
        },
        {
            key: 'diversion_rate', el: 'kpi-diversion',
            fmt: (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`),
            deltaFmt: (d) => `${d > 0 ? '+' : ''}${d}pt`,
        },
        {
            key: 'avg_confidence', el: 'kpi-confidence',
            fmt: (v) => (v == null ? '—' : v.toFixed(2)),
            deltaFmt: (d) => `${d > 0 ? '+' : ''}${d.toFixed(2)}`,
        },
        {
            key: 'active_bins', el: 'kpi-bins',
            fmt: (v) => (v == null ? '—' : v.toLocaleString()),
            deltaFmt: (d) => `${d > 0 ? '+' : ''}${d}`,
        },
    ];

    function renderKpis(kpis, periodLabel) {
        KPI_DEFS.forEach((def) => {
            const k = (kpis && kpis[def.key]) || {};
            const numEl = document.getElementById(def.el);
            const deltaEl = document.getElementById(def.el + '-delta');
            const noteEl = document.getElementById(def.el + '-note');
            if (numEl) numEl.textContent = def.fmt(k.value);
            if (deltaEl) {
                deltaEl.classList.remove('pos', 'neg');
                if (k.delta == null) {
                    deltaEl.textContent = '—';
                } else if (k.delta === 0) {
                    deltaEl.textContent = '· ±0';
                } else {
                    deltaEl.textContent = `${k.delta > 0 ? '↗' : '↘'} ${def.deltaFmt(k.delta)}`;
                    deltaEl.classList.add(k.delta > 0 ? 'pos' : 'neg');
                }
            }
            if (noteEl) noteEl.textContent = `vs. previous ${periodLabel}`;
        });
    }

    // ── Period-aware card titles ──────────────────────────────────────
    function retitle(periodLabel) {
        const titles = {
            'area-title': `Classifications by category · ${periodLabel}`,
            'donut-title': `Material mix · ${periodLabel}`,
            'leaderboard-title': `Bins by throughput · ${periodLabel}`,
        };
        Object.entries(titles).forEach(([id, text]) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        });
    }

    // ── Stacked area chart ───────────────────────────────────────────
    function renderAreaLegend() {
        const host = document.getElementById('area-legend');
        host.innerHTML = CAT.map((c) =>
            `<span class="legend-pill"><span class="sw" style="background:${COLORS[c]};"></span>${c}</span>`
        ).join('');
    }

    function renderStackedArea(series) {
        const host = document.getElementById('stacked-area');
        const buckets = (series && series.buckets) || [];
        const data = (series && series.data) || {};
        const order = (series && series.categories) || CAT;
        const n = buckets.length;
        if (!n) {
            host.innerHTML = '<div class="empty-line">No data in this period.</div>';
            return;
        }
        const W = 740, H = 240, P = 12;
        const totals = Array.from({ length: n }, (_, i) =>
            order.reduce((a, c) => a + ((data[c] || [])[i] || 0), 0)
        );
        const maxT = Math.max(1, ...totals);
        const xs = (i) => (n > 1 ? P + (i / (n - 1)) * (W - 2 * P) : W / 2);
        const ys = (v) => H - P - (v / maxT) * (H - 2 * P);

        let cumul = Array(n).fill(0);
        let marks;
        if (n === 1) {
            // Single bucket (e.g. YTD on Jan 1) — a centered stacked bar.
            marks = order.map((c) => {
                const v = (data[c] || [])[0] || 0;
                if (!v) return '';
                const y0 = ys(cumul[0]), y1 = ys(cumul[0] + v);
                cumul[0] += v;
                return `<rect x="${W / 2 - 30}" y="${y1}" width="60" height="${Math.max(1, y0 - y1)}"
                              fill="${COLORS[c]}"><title>${c}: ${v}</title></rect>`;
            });
        } else {
            marks = order.map((c) => {
                const s = buckets.map((_, i) => (data[c] || [])[i] || 0);
                const top = s.map((v, i) => cumul[i] + v);
                const bottom = [...cumul];
                const path = [
                    `M ${xs(0)} ${ys(top[0])}`,
                    ...top.map((v, i) => `L ${xs(i)} ${ys(v)}`),
                    ...bottom.reverse().map((v, i) => `L ${xs(n - 1 - i)} ${ys(v)}`),
                    'Z',
                ].join(' ');
                cumul = top;
                const catTotal = s.reduce((a, v) => a + v, 0);
                return `<path d="${path}" fill="${COLORS[c]}" fill-opacity="0.92"
                              stroke="${COLORS[c]}" stroke-width="0.5"><title>${c}: ${catTotal}</title></path>`;
            });
        }
        const grid = [0.25, 0.5, 0.75, 1].map((t) =>
            `<line x1="${P}" x2="${W - P}" y1="${H - P - t * (H - 2 * P)}" y2="${H - P - t * (H - 2 * P)}" stroke="rgba(29,39,34,0.08)" stroke-width="1"/>`
        ).join('');
        const labelStep = Math.max(1, Math.ceil(n / 10));
        const labels = buckets.map((b, i) => {
            if (i % labelStep !== 0 && i !== n - 1) return '';
            return `<text x="${xs(i)}" y="${H - 1}" font-size="10" fill="#8C8C8C" text-anchor="middle" font-family="Manrope">${escapeHtml(b.label)}</text>`;
        }).join('');

        host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="100%" height="${H}">${grid}${marks.join('')}${labels}</svg>`;
    }

    // ── Donut chart ───────────────────────────────────────────────────
    function foldCategories(byCategory) {
        const out = Object.fromEntries(CAT.map((c) => [c, 0]));
        Object.entries(byCategory || {}).forEach(([label, v]) => {
            if (label === 'Empty') return; // mix excludes empty checks
            out[CAT.includes(label) ? label : 'Other'] += v;
        });
        return out;
    }

    function renderDonut(byCategory, byCategoryPrev) {
        const host = document.getElementById('donut-chart');
        const cur = foldCategories(byCategory);
        const prev = foldCategories(byCategoryPrev);
        const cats = CAT.map((c) => ({ name: c, value: cur[c] || 0 }));
        const total = cats.reduce((a, c) => a + c.value, 0);
        const sum = Math.max(1, total);
        const R = 80, r = 50, cx = 110, cy = 110;
        let angle = -Math.PI / 2;
        const slicesSvg = cats.map((c) => {
            const portion = c.value / sum;
            if (portion === 0) return '';
            const a0 = angle, a1 = angle + portion * 2 * Math.PI;
            angle = a1;
            const large = portion > 0.5 ? 1 : 0;
            const x0 = cx + R * Math.cos(a0), y0 = cy + R * Math.sin(a0);
            const x1 = cx + R * Math.cos(a1), y1 = cy + R * Math.sin(a1);
            const xi0 = cx + r * Math.cos(a0), yi0 = cy + r * Math.sin(a0);
            const xi1 = cx + r * Math.cos(a1), yi1 = cy + r * Math.sin(a1);
            return `<path d="M ${x0} ${y0} A ${R} ${R} 0 ${large} 1 ${x1} ${y1} L ${xi1} ${yi1} A ${r} ${r} 0 ${large} 0 ${xi0} ${yi0} Z" fill="${COLORS[c.name]}"><title>${c.name}: ${c.value}</title></path>`;
        }).join('');
        const legend = cats.map((c) => {
            const p = prev[c.name] || 0;
            let deltaHtml = '<span class="delta">—</span>';
            if (p > 0) {
                const d = Math.round(((c.value - p) / p) * 100);
                deltaHtml = `<span class="delta ${d < 0 ? 'neg' : 'pos'}">${d > 0 ? '+' : ''}${d}%</span>`;
            }
            return `<div class="donut-legend-row">
                <span class="sw" style="background:${COLORS[c.name]};"></span>
                <span class="name">${c.name}</span>
                <span class="val cc-mono">${c.value}</span>
                ${deltaHtml}
            </div>`;
        }).join('');

        host.innerHTML = `
            <svg viewBox="0 0 220 220" width="220" height="220">${slicesSvg}
                <text x="110" y="105" text-anchor="middle" font-size="11" fill="#8C8C8C" font-family="Manrope" font-weight="700" letter-spacing="2">TOTAL</text>
                <text x="110" y="130" text-anchor="middle" font-size="28" fill="#1d2722" font-family="Chakra Petch" font-weight="600">${total.toLocaleString()}</text>
            </svg>
            <div class="donut-legend">${legend}</div>
        `;
    }

    // ── Leaderboard ──────────────────────────────────────────────────
    function renderLeaderboard(rows) {
        const host = document.getElementById('leaderboard');
        const sorted = (rows || []).slice(0, 6);
        if (!sorted.length) {
            host.innerHTML = `<div class="empty-line">No classifications in this period.</div>`;
            return;
        }
        const max = Math.max(1, ...sorted.map((b) => b.count || 0));
        host.innerHTML = sorted.map((b, i) => `
            <div class="leader-row">
                <div class="leader-rank cc-mono">${i + 1}</div>
                <div class="leader-body">
                    <div class="leader-head">
                        <span class="leader-name">${escapeHtml(b.bin_id)}</span>
                        <span class="leader-num cc-mono">${(b.count || 0).toLocaleString()}</span>
                    </div>
                    <div class="leader-track">
                        <div class="leader-bar" style="width:${(b.count || 0) / max * 100}%;"></div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    // ── Recent classifications ───────────────────────────────────────
    function renderRecent(entries) {
        const host = document.getElementById('recent-classifications');
        if (!entries || !entries.length) {
            host.innerHTML = `<div class="empty-line">No classifications yet — start a bin to fill this table.</div>`;
            return;
        }
        host.innerHTML = entries.slice(0, 6).map((e) => {
            let timeLabel = '—';
            try {
                const d = new Date(e.timestamp);
                if (!isNaN(d.getTime())) {
                    const hh = String(d.getHours()).padStart(2, '0');
                    const mm = String(d.getMinutes()).padStart(2, '0');
                    const ss = String(d.getSeconds()).padStart(2, '0');
                    timeLabel = `${hh}:${mm}:${ss}`;
                }
            } catch (_) {}
            const cat = CAT.includes(e.label) ? e.label : (e.label || 'Other');
            const color = COLORS[cat] || COLORS.Other;
            const desc = e.description || e.brand_product || cat;
            return `
                <div class="recent-row">
                    <span class="recent-time cc-mono">${timeLabel}</span>
                    <span class="recent-cat"><span class="sw" style="background:${color};"></span>${escapeHtml(cat)}</span>
                    <span class="recent-desc">${escapeHtml(desc)}</span>
                    <span class="recent-bin cc-mono">${escapeHtml(e.bin_id || '—')}</span>
                </div>
            `;
        }).join('');
    }

    // ── LLM backend mix ──────────────────────────────────────────────
    function renderBackends(backends, total, periodLabel) {
        const host = document.getElementById('backend-stats');
        const meta = document.getElementById('backend-meta');
        if (meta) meta.textContent = `n = ${(total || 0).toLocaleString()} · ${periodLabel}`;
        if (!backends || !backends.length) {
            host.innerHTML = `<div class="empty-line">No classifications in this period.</div>`;
            return;
        }
        const max = Math.max(1, ...backends.map((b) => b.count || 0));
        host.innerHTML = backends.map((b) => `
            <div class="backend-row">
                <span class="backend-name cc-mono">${escapeHtml(b.backend)}</span>
                <div class="leader-track">
                    <div class="leader-bar" style="width:${(b.count || 0) / max * 100}%;"></div>
                </div>
                <span class="backend-count cc-mono">${(b.count || 0).toLocaleString()}</span>
                <span class="backend-conf cc-mono">${b.avg_confidence == null ? 'conf —' : 'conf ' + b.avg_confidence.toFixed(2)}</span>
            </div>
        `).join('');
    }

    // ── Boot ──────────────────────────────────────────────────────────
    async function boot() {
        renderAreaLegend();

        const [payload, entries] = await Promise.all([
            fetchJSON('/api/analytics?period=' + encodeURIComponent(period)),
            fetchJSON('/api/entries?limit=6'),
        ]);

        if (payload) {
            retitle(payload.period_label);
            renderKpis(payload.kpis, payload.period_label);
            renderStackedArea(payload.series);
            renderDonut(payload.by_category, payload.by_category_prev);
            renderLeaderboard(payload.leaderboard);
            const total = payload.kpis && payload.kpis.total ? payload.kpis.total.value : 0;
            renderBackends(payload.backends, total, payload.period_label);
        }

        renderRecent(entries || []);

        const hostEl = document.getElementById('cc-server-host');
        if (hostEl) hostEl.textContent = window.location.host || '—';
    }

    document.querySelectorAll('.period-btn').forEach((b) => {
        b.addEventListener('click', () => {
            document.querySelectorAll('.period-btn').forEach((x) => x.classList.remove('active'));
            b.classList.add('active');
            period = b.dataset.period || '7d';
            boot();
        });
    });

    document.getElementById('export-csv').addEventListener('click', () => {
        window.location.href = '/api/analytics/export?period=' + encodeURIComponent(period);
    });

    boot();
    setInterval(boot, 15000);
})();
