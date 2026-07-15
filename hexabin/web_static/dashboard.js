/* HexaBin Fleet Control dashboard
 * Polls /api/dashboard, renders bin cards in the new editorial style,
 * and dispatches admin commands to POST /api/bin/{id}/command.
 */
(function () {
    'use strict';

    const POLL_INTERVAL_MS = 5000;
    const grid = document.getElementById('bins-grid');
    const filterInput = document.getElementById('filter-input');
    const refreshBtn = document.getElementById('refresh-btn');
    const toastHost = document.getElementById('toast-host');
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    const modalOk = document.getElementById('modal-ok');
    const modalCancel = document.getElementById('modal-cancel');
    const lastRefresh = document.getElementById('last-refresh');
    const filterChips = document.querySelectorAll('.filter-chips .chip');

    let lastBins = [];
    let textFilter = '';
    let statusFilter = 'all';

    function toast(message, kind) {
        const el = document.createElement('div');
        el.className = 'toast ' + (kind || '');
        el.textContent = message;
        toastHost.appendChild(el);
        setTimeout(() => {
            el.classList.add('leave');
            setTimeout(() => el.remove(), 260);
        }, 3200);
    }

    function confirmModal(title, body) {
        return new Promise((resolve) => {
            modalTitle.textContent = title;
            modalBody.textContent = body;
            modal.hidden = false;
            const cleanup = (ans) => {
                modal.hidden = true;
                modalOk.onclick = null;
                modalCancel.onclick = null;
                resolve(ans);
            };
            modalOk.onclick = () => cleanup(true);
            modalCancel.onclick = () => cleanup(false);
        });
    }

    async function sendCommand(binId, action, value) {
        try {
            const res = await fetch(`/api/bin/${encodeURIComponent(binId)}/command`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, value: value ?? null }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                toast(data.error || data.message || `Command failed (${res.status})`, 'error');
                return false;
            }
            toast(data.message || 'Command sent', 'success');
            refresh();
            return true;
        } catch (e) {
            toast('Network error: ' + e.message, 'error');
            return false;
        }
    }

    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    }

    function lastSeenLabel(iso) {
        if (!iso) return 'never';
        try {
            const t = new Date(iso);
            const diffSec = Math.max(0, Math.round((Date.now() - t.getTime()) / 1000));
            if (diffSec < 60) return `${diffSec}s ago`;
            if (diffSec < 3600) return `${Math.round(diffSec / 60)}m ago`;
            return `${Math.round(diffSec / 3600)}h ago`;
        } catch (e) { return iso; }
    }

    function renderWarnings(warnings) {
        if (!warnings || !warnings.length) return '';
        const icons = { info: 'i', warning: '⚠', error: '×' };
        return `<div class="bin-warnings">${warnings.map((w) => `
            <div class="bin-warning sev-${escapeHtml(w.severity || 'warning')}">
                <span class="sev">${icons[w.severity] || '⚠'}</span>
                <span>${escapeHtml(w.message || w.code)}</span>
            </div>`).join('')}</div>`;
    }

    function pipelineLabel(p) {
        if (p === 'oak') return 'OAK Dual';
        if (p === 'oak-native') return 'OAK Native';
        if (p === 'raspberry') return 'Raspberry';
        return p || '—';
    }

    function renderCard(bin) {
        const isOnline = bin.status === 'online' || bin.status === 'degraded';
        const hasHost = !!bin.has_host;
        const running = !!bin.running;
        const pipeline = bin.pipeline || bin.camera_mode || 'oak';
        const strategy = bin.strategy || 'manual';
        const pipelineOptions = ['oak', 'oak-native']
            .map((p) => `<option value="${p}" ${p === pipeline ? 'selected' : ''}>${p === 'oak' ? 'OAK Dual' : 'OAK Native'}</option>`)
            .join('');
        const strategyOptions = ['manual', 'auto']
            .map((s) => `<option value="${s}" ${s === strategy ? 'selected' : ''}>${s === 'manual' ? 'Manual' : 'Auto Gate'}</option>`)
            .join('');
        const strategyDisabled = pipeline !== 'oak';

        const thumbInner = isOnline && hasHost
            ? `<img alt="live stream for ${escapeHtml(bin.bin_id)}"
                    src="/api/bin/${encodeURIComponent(bin.bin_id)}/stream" loading="lazy">
               <div class="bin-thumb-stripes"></div>`
            : (bin.status === 'offline'
                ? `<div class="bin-thumb-placeholder"><div class="glyph">⏻</div>NO SIGNAL</div>`
                : `<div class="bin-thumb-placeholder">STREAM UNAVAILABLE</div>`);

        // Lifecycle button mirrors the design: Start (when stopped), Stop (when running+online), Restart otherwise.
        const lifecycleBtn = running
            ? (bin.status === 'offline'
                ? `<button class="sb-btn sb-btn-secondary" data-action="restart">↻ Restart</button>`
                : `<button class="sb-btn sb-btn-secondary" data-action="stop">⏸ Stop</button>`)
            : `<button class="sb-btn sb-btn-primary" data-action="start">▶ Start</button>`;

        // Fill bar is hidden until backend exposes a real fill metric. We keep the markup but render a muted placeholder.
        const fillRow = `
            <div class="bin-fill">
                <div class="bin-fill-head">
                    <span>Fill level</span>
                    <span class="bin-fill-unknown">—</span>
                </div>
                <div class="bin-fill-track"><div class="bin-fill-bar" style="width:0%;"></div></div>
            </div>`;

        return `
            <article class="bin-card" data-bin="${escapeHtml(bin.bin_id)}" data-status="${escapeHtml(bin.status)}">
                <div class="bin-thumb ${bin.status === 'offline' ? 'offline' : ''}">
                    ${thumbInner}
                    <div class="bin-thumb-status"><span class="sb-pill ${escapeHtml(bin.status)}"><span class="dot"></span>${escapeHtml(bin.status)}</span></div>
                    <div class="bin-thumb-foot">
                        <span>● REC · ${escapeHtml(bin.bin_id.toUpperCase())}</span>
                        <span>${escapeHtml(lastSeenLabel(bin.last_seen))}</span>
                    </div>
                </div>

                <div class="bin-body">
                    <div class="bin-head">
                        <div class="bin-head-text">
                            <div class="bin-loc">${escapeHtml(bin.location || 'Location not set')}</div>
                            <div class="bin-id">${escapeHtml(bin.bin_id)}${bin.lat && bin.lng ? ' · ' + bin.lat.toFixed(4) + ', ' + bin.lng.toFixed(4) : ''}</div>
                        </div>
                        <div class="bin-overflow">
                            <button class="bin-overflow-btn" data-action="toggle-menu" aria-label="More options">⋯</button>
                            <div class="bin-overflow-menu" data-menu>
                                <div class="row">
                                    <label>Strategy</label>
                                    <select data-control="strategy" ${strategyDisabled ? 'disabled title="Only on OAK Dual"' : ''}>${strategyOptions}</select>
                                </div>
                                <div class="row">
                                    <label>Pipeline</label>
                                    <select data-control="pipeline">${pipelineOptions}</select>
                                </div>
                            </div>
                        </div>
                    </div>

                    ${fillRow}

                    <div class="bin-meta">
                        <div class="bin-meta-cell">
                            <div class="key">Today</div>
                            <div class="val">${bin.total_entries ?? 0}</div>
                        </div>
                        <div class="bin-meta-cell">
                            <div class="key">Mode</div>
                            <div class="val ${bin.auto_classify ? '' : 'muted'}">${bin.auto_classify ? 'AUTO' : 'MANUAL'}</div>
                        </div>
                        <div class="bin-meta-cell">
                            <div class="key">Pipeline</div>
                            <div class="val">${escapeHtml(pipelineLabel(pipeline))}</div>
                        </div>
                    </div>

                    ${renderWarnings(bin.warnings)}

                    <div class="bin-controls">
                        ${lifecycleBtn}
                        <button class="sb-btn sb-btn-secondary" data-action="classify" ${(!running || !hasHost) ? 'disabled' : ''}>⊙ Classify</button>
                        <a class="sb-btn sb-btn-outline" href="/bin/${encodeURIComponent(bin.bin_id)}">Open →</a>
                    </div>
                </div>
            </article>
        `;
    }

    function applyFilters(bins) {
        let out = bins;
        if (statusFilter !== 'all') {
            out = out.filter((b) => (b.status || '') === statusFilter);
        }
        if (textFilter) {
            const q = textFilter.toLowerCase();
            out = out.filter((b) =>
                (b.bin_id || '').toLowerCase().includes(q) ||
                (b.location || '').toLowerCase().includes(q)
            );
        }
        return out;
    }

    function render() {
        const bins = applyFilters(lastBins);
        if (!bins.length) {
            const msg = lastBins.length ? 'No matches' : 'No bins registered';
            const sub = lastBins.length ? 'Adjust the filters above.' : 'Start an edge device to see it here.';
            grid.innerHTML = `<div class="empty-state"><h2>${msg}</h2><p>${sub}</p></div>`;
            return;
        }
        grid.innerHTML = bins.map(renderCard).join('');
        wireCardHandlers();
    }

    async function onAction(card, action) {
        const binId = card.dataset.bin;
        if (!binId) return;
        if (action === 'restart' || action === 'stop') {
            const title = action === 'restart' ? 'Restart bin?' : 'Stop bin?';
            const body = action === 'restart'
                ? `Process will exit and the container will restart. 5–15 seconds of downtime on ${binId}.`
                : `${binId} will stop accepting classifications until you press Start again.`;
            const ok = await confirmModal(title, body);
            if (!ok) return;
        }
        sendCommand(binId, action === 'toggle_auto' ? 'toggle_auto' : action);
    }

    async function onSelectChange(card, control, value) {
        const binId = card.dataset.bin;
        if (!binId) return;
        if (control === 'pipeline') {
            const ok = await confirmModal(
                'Change pipeline?',
                `Switching to ${value} requires a container restart on ${binId}. ~10s of downtime.`
            );
            if (!ok) { refresh(); return; }
            sendCommand(binId, 'set_pipeline', value);
        } else if (control === 'strategy') {
            sendCommand(binId, 'set_strategy', value);
        }
    }

    function wireCardHandlers() {
        grid.querySelectorAll('.bin-card').forEach((card) => {
            const menu = card.querySelector('[data-menu]');
            card.querySelectorAll('button[data-action]').forEach((btn) => {
                btn.addEventListener('click', (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                    const action = btn.dataset.action;
                    if (action === 'toggle-menu') {
                        menu.classList.toggle('open');
                        return;
                    }
                    btn.disabled = true;
                    onAction(card, action).finally(() => { btn.disabled = false; });
                });
            });
            card.querySelectorAll('select[data-control]').forEach((sel) => {
                sel.addEventListener('change', () => {
                    onSelectChange(card, sel.dataset.control, sel.value);
                });
            });
        });
    }

    // Close overflow menus when clicking outside
    document.addEventListener('click', (ev) => {
        document.querySelectorAll('.bin-overflow-menu.open').forEach((m) => {
            if (!m.parentElement.contains(ev.target)) m.classList.remove('open');
        });
    });

    function updateStats(d) {
        const total = d.total_bins ?? 0;
        document.getElementById('stat-online').textContent = d.online ?? 0;
        document.getElementById('stat-degraded').textContent = d.degraded ?? 0;
        document.getElementById('stat-offline').textContent = d.offline ?? 0;
        document.getElementById('stat-total').textContent = (d.total_entries ?? 0).toLocaleString();
        document.getElementById('stat-online-sub').textContent = `of ${total} total`;
        document.getElementById('stat-degraded-sub').textContent = (d.degraded ?? 0) === 1 ? '1 warning' : `${d.degraded ?? 0} warnings`;
    }

    async function refresh() {
        try {
            const res = await fetch('/api/dashboard', { cache: 'no-store' });
            if (!res.ok) {
                if (res.status === 401) { window.location.href = '/login'; return; }
                return;
            }
            const data = await res.json();
            lastBins = data.bins || [];
            updateStats(data);
            const now = new Date();
            const hh = String(now.getHours()).padStart(2, '0');
            const mm = String(now.getMinutes()).padStart(2, '0');
            const ss = String(now.getSeconds()).padStart(2, '0');
            if (lastRefresh) lastRefresh.textContent = `updated ${hh}:${mm}:${ss}`;
            // Server host (sidebar foot) — best-effort from window.location
            const hostEl = document.getElementById('cc-server-host');
            if (hostEl) hostEl.textContent = window.location.host || '—';
            render();
        } catch (e) { /* transient errors are non-fatal */ }
    }

    filterInput && filterInput.addEventListener('input', () => {
        textFilter = filterInput.value.trim();
        render();
    });
    refreshBtn && refreshBtn.addEventListener('click', refresh);
    filterChips.forEach((c) => {
        c.addEventListener('click', () => {
            filterChips.forEach((x) => x.classList.remove('active'));
            c.classList.add('active');
            statusFilter = c.dataset.filter || 'all';
            render();
        });
    });

    refresh();
    setInterval(refresh, POLL_INTERVAL_MS);
})();
