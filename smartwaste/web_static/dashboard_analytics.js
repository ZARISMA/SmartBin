/* SmartBin Analytics view
 * Wires real data where we have it (/api/stats + /api/entries + /api/dashboard)
 * and uses static placeholders for metrics the backend doesn't yet compute.
 * Period selector (24h/7d/30d/90d/YTD) is a visual stub until period filtering exists.
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
    const DELTAS = {
        Plastic: '+12%', Paper: '+8%', Glass: '+18%',
        Organic: '+24%', Aluminum: '+3%', Other: '−4%',
    };
    const NEG = new Set(['Other']);

    // Deterministic fallback when there isn't enough real data for a 7-day series.
    const FALLBACK_SERIES = {
        Plastic:  [40, 52, 64, 48, 70, 92, 116],
        Paper:    [38, 42, 48, 56, 60, 72, 80],
        Glass:    [22, 24, 30, 28, 36, 38, 33],
        Organic:  [16, 18, 22, 26, 28, 32, 36],
        Aluminum: [12, 14, 14, 16, 16, 18, 14],
        Other:    [4,  5,  6,  8,  7,  9,  8],
    };

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
    function renderKpis(stats) {
        const t = (stats && stats.total) || 0;
        document.getElementById('kpi-total').textContent = t.toLocaleString();
    }

    // ── Stacked area chart ───────────────────────────────────────────
    function renderAreaLegend() {
        const host = document.getElementById('area-legend');
        host.innerHTML = CAT.map((c) =>
            `<span class="legend-pill"><span class="sw" style="background:${COLORS[c]};"></span>${c}</span>`
        ).join('');
    }

    function buildSeriesFromEntries(entries) {
        if (!entries || !entries.length) return null;
        const day = 24 * 60 * 60 * 1000;
        const now = Date.now();
        const buckets = Array.from({ length: 7 }, () => Object.fromEntries(CAT.map((c) => [c, 0])));
        let any = false;
        entries.forEach((e) => {
            const ts = e.timestamp ? new Date(e.timestamp).getTime() : 0;
            if (!ts) return;
            const ageDays = Math.floor((now - ts) / day);
            if (ageDays < 0 || ageDays > 6) return;
            const idx = 6 - ageDays;
            const cat = CAT.includes(e.label) ? e.label : 'Other';
            buckets[idx][cat] += 1;
            any = true;
        });
        if (!any) return null;
        const out = {};
        CAT.forEach((c) => { out[c] = buckets.map((b) => b[c]); });
        return out;
    }

    function renderStackedArea(series) {
        const host = document.getElementById('stacked-area');
        const days = 7;
        const W = 740, H = 240, P = 12;
        const order = CAT;
        const totals = Array.from({ length: days }, (_, d) =>
            order.reduce((a, c) => a + series[c][d], 0)
        );
        const maxT = Math.max(1, ...totals);
        const xs = (i) => P + (i / (days - 1)) * (W - 2 * P);
        const ys = (v) => H - P - (v / maxT) * (H - 2 * P);

        let cumul = Array(days).fill(0);
        const paths = order.map((c) => {
            const s = series[c];
            const top = s.map((v, d) => cumul[d] + v);
            const bottom = [...cumul];
            const path = [
                `M ${xs(0)} ${ys(top[0])}`,
                ...top.map((v, d) => `L ${xs(d)} ${ys(v)}`),
                ...bottom.reverse().map((v, d) => `L ${xs(days - 1 - d)} ${ys(v)}`),
                'Z',
            ].join(' ');
            cumul = top;
            return `<path d="${path}" fill="${COLORS[c]}" fill-opacity="0.92" stroke="${COLORS[c]}" stroke-width="0.5"/>`;
        });
        const grid = [0.25, 0.5, 0.75, 1].map((t) =>
            `<line x1="${P}" x2="${W - P}" y1="${H - P - t * (H - 2 * P)}" y2="${H - P - t * (H - 2 * P)}" stroke="rgba(29,39,34,0.08)" stroke-width="1"/>`
        ).join('');
        const days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const labels = days_labels.map((d, i) =>
            `<text x="${xs(i)}" y="${H - 1}" font-size="10" fill="#8C8C8C" text-anchor="middle" font-family="Manrope">${d}</text>`
        ).join('');

        host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="100%" height="${H}">${grid}${paths.join('')}${labels}</svg>`;
    }

    // ── Donut chart ───────────────────────────────────────────────────
    function renderDonut(by_category, total) {
        const host = document.getElementById('donut-chart');
        const cats = CAT.map((c) => ({ name: c, value: (by_category && by_category[c]) || 0 }));
        const sum = Math.max(1, cats.reduce((a, c) => a + c.value, 0));
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
            return `<path d="M ${x0} ${y0} A ${R} ${R} 0 ${large} 1 ${x1} ${y1} L ${xi1} ${yi1} A ${r} ${r} 0 ${large} 0 ${xi0} ${yi0} Z" fill="${COLORS[c.name]}"/>`;
        }).join('');
        const totalNum = (total || sum).toLocaleString();
        const legend = cats.map((c) =>
            `<div class="donut-legend-row">
                <span class="sw" style="background:${COLORS[c.name]};"></span>
                <span class="name">${c.name}</span>
                <span class="val cc-mono">${c.value}</span>
                <span class="delta ${NEG.has(c.name) ? 'neg' : 'pos'}">${DELTAS[c.name] || ''}</span>
            </div>`
        ).join('');

        host.innerHTML = `
            <svg viewBox="0 0 220 220" width="220" height="220">${slicesSvg}
                <text x="110" y="105" text-anchor="middle" font-size="11" fill="#8C8C8C" font-family="Manrope" font-weight="700" letter-spacing="2">TOTAL</text>
                <text x="110" y="130" text-anchor="middle" font-size="28" fill="#1d2722" font-family="Instrument Serif">${totalNum}</text>
            </svg>
            <div class="donut-legend">${legend}</div>
        `;
    }

    // ── Leaderboard ──────────────────────────────────────────────────
    function renderLeaderboard(bins) {
        const host = document.getElementById('leaderboard');
        const sorted = bins.slice().sort((a, b) => (b.total_entries || 0) - (a.total_entries || 0)).slice(0, 6);
        if (!sorted.length) {
            host.innerHTML = `<div class="empty-line">No bin throughput yet.</div>`;
            return;
        }
        const max = Math.max(1, ...sorted.map((b) => b.total_entries || 0));
        host.innerHTML = sorted.map((b, i) => `
            <div class="leader-row">
                <div class="leader-rank cc-mono">${i + 1}</div>
                <div class="leader-body">
                    <div class="leader-head">
                        <span class="leader-name">${b.location || b.bin_id}</span>
                        <span class="leader-num cc-mono">${(b.total_entries || 0).toLocaleString()}</span>
                    </div>
                    <div class="leader-track">
                        <div class="leader-bar" style="width:${(b.total_entries || 0) / max * 100}%;"></div>
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
                    <span class="recent-cat"><span class="sw" style="background:${color};"></span>${cat}</span>
                    <span class="recent-desc">${desc}</span>
                    <span class="recent-bin cc-mono">${e.bin_id || '—'}</span>
                </div>
            `;
        }).join('');
    }

    // ── Confusion matrix (static placeholder) ────────────────────────
    function renderConfusionMatrix() {
        const labels = ['Plastic', 'Paper', 'Glass', 'Organic', 'Aluminum', 'Other', 'Empty'];
        const m = [
            [0.96, 0.01, 0.01, 0.00, 0.00, 0.02, 0.00],
            [0.02, 0.94, 0.00, 0.02, 0.00, 0.01, 0.01],
            [0.01, 0.00, 0.93, 0.00, 0.04, 0.02, 0.00],
            [0.00, 0.03, 0.00, 0.91, 0.00, 0.05, 0.01],
            [0.00, 0.00, 0.05, 0.00, 0.92, 0.03, 0.00],
            [0.04, 0.02, 0.03, 0.06, 0.03, 0.81, 0.01],
            [0.01, 0.00, 0.00, 0.01, 0.00, 0.01, 0.97],
        ];
        const host = document.getElementById('confusion-matrix');
        let html = `<div class="cm-cell cm-corner"></div>`;
        labels.forEach((l) => { html += `<div class="cm-cell cm-h">${l}</div>`; });
        m.forEach((row, i) => {
            html += `<div class="cm-cell cm-r">${labels[i]}</div>`;
            row.forEach((v, j) => {
                const txtLight = v > 0.5 ? 'light' : '';
                const bold = i === j ? 'bold' : '';
                html += `<div class="cm-cell cm-v ${txtLight} ${bold}" style="background:rgba(45,90,66,${v.toFixed(2)});">${Math.round(v * 100)}</div>`;
            });
        });
        host.innerHTML = html;
    }

    // ── Boot ──────────────────────────────────────────────────────────
    async function boot() {
        renderAreaLegend();
        renderConfusionMatrix();

        const [stats, dash, entries] = await Promise.all([
            fetchJSON('/api/stats'),
            fetchJSON('/api/dashboard'),
            fetchJSON('/api/entries?limit=100'),
        ]);

        renderKpis(stats || {});

        const total = (stats && stats.total) || 0;
        const byCat = (stats && stats.by_category) || {};
        renderDonut(byCat, total);

        const series = buildSeriesFromEntries(entries) || FALLBACK_SERIES;
        renderStackedArea(series);

        const bins = (dash && dash.bins) || [];
        renderLeaderboard(bins);

        renderRecent(entries || []);

        document.getElementById('cm-meta').textContent = `n = ${(total || 0).toLocaleString()} · 7 days`;

        const hostEl = document.getElementById('cc-server-host');
        if (hostEl) hostEl.textContent = window.location.host || '—';
    }

    // Period selector — visual stub. Re-rendering with real period filtering would
    // require backend support (period query param on /api/stats and /api/entries).
    document.querySelectorAll('.period-btn').forEach((b) => {
        b.addEventListener('click', () => {
            document.querySelectorAll('.period-btn').forEach((x) => x.classList.remove('active'));
            b.classList.add('active');
        });
    });

    document.getElementById('export-csv').addEventListener('click', async () => {
        const entries = await fetchJSON('/api/entries?limit=1000');
        if (!entries || !entries.length) return;
        const cols = ['timestamp', 'bin_id', 'label', 'description', 'brand_product', 'location'];
        const csv = [cols.join(',')]
            .concat(entries.map((e) => cols.map((c) => JSON.stringify(e[c] ?? '')).join(',')))
            .join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'smartbin-classifications.csv';
        a.click();
    });

    boot();
    setInterval(boot, 15000);
})();
