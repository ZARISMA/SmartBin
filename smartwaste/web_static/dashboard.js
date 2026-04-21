/* Smart Waste AI — Fleet Control dashboard logic
 * Polls /api/dashboard, renders bin cards, and dispatches admin commands
 * to POST /api/bin/{id}/command with optimistic UI feedback.
 */
(function () {
    'use strict';

    const POLL_INTERVAL_MS = 4000;
    const grid = document.getElementById('bins-grid');
    const filterInput = document.getElementById('filter-input');
    const refreshBtn = document.getElementById('refresh-btn');
    const toastHost = document.getElementById('toast-host');

    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    const modalOk = document.getElementById('modal-ok');
    const modalCancel = document.getElementById('modal-cancel');

    let lastBins = [];
    let filter = '';

    // Preserve control state across re-renders so dropdowns don't reset while
    // the user is interacting with them.
    const pendingSelects = new Map();

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

    function statusClass(status) {
        switch (status) {
            case 'online': return 'status-online';
            case 'degraded': return 'status-degraded';
            case 'stopped': return 'status-stopped';
            default: return 'status-offline';
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
        const icons = { info: 'i', warning: '!', error: '×' };
        return `<div class="bin-warnings">${warnings.map((w) => `
            <div class="warning-item sev-${escapeHtml(w.severity || 'warning')}">
                <span class="sev-icon">${icons[w.severity] || '!'}</span>
                <span>${escapeHtml(w.message || w.code)}</span>
            </div>`).join('')}</div>`;
    }

    function renderCard(bin) {
        const isOnline = bin.status === 'online' || bin.status === 'degraded';
        const hasHost = !!bin.has_host;
        const running = !!bin.running;
        const thumb = isOnline && hasHost
            ? `<img alt="live stream for ${escapeHtml(bin.bin_id)}"
                    src="/api/bin/${encodeURIComponent(bin.bin_id)}/stream" loading="lazy">
               <span class="thumb-badge">LIVE</span>`
            : `<div class="thumb-placeholder">${bin.status === 'offline' ? 'Offline — no stream' : 'Stream unavailable'}</div>`;

        const cameraCount = bin.camera_count || 0;
        const pipeline = bin.pipeline || bin.camera_mode || '—';
        const strategy = bin.strategy || '—';
        const pipelineOptions = ['oak', 'oak-native']
            .map((p) => `<option value="${p}" ${p === pipeline ? 'selected' : ''}>${p === 'oak' ? 'OAK Dual' : 'OAK Native'}</option>`)
            .join('');
        const strategyOptions = ['manual', 'auto']
            .map((s) => `<option value="${s}" ${s === strategy ? 'selected' : ''}>${s === 'manual' ? 'Manual' : 'Auto Gate'}</option>`)
            .join('');
        const strategyDisabled = pipeline !== 'oak';

        const lifecycleBtn = running
            ? `<button class="btn-ghost" data-action="stop">⏸ Stop</button>`
            : `<button class="btn-primary" data-action="start">▶ Start</button>`;

        return `
            <article class="bin-card ${statusClass(bin.status)}" data-bin="${escapeHtml(bin.bin_id)}">
                <header class="bin-card-head">
                    <div>
                        <div class="bin-title">${escapeHtml(bin.bin_id)}</div>
                        <div class="bin-subtitle">${escapeHtml(bin.location || 'Location not set')} · last seen ${escapeHtml(lastSeenLabel(bin.last_seen))}</div>
                    </div>
                    <span class="bin-status-pill ${statusClass(bin.status)}">${escapeHtml(bin.status)}</span>
                </header>

                <div class="bin-thumb">${thumb}</div>

                <div class="bin-meta">
                    <div class="meta-cell">
                        <div class="meta-key">Pipeline</div>
                        <div class="meta-val">${escapeHtml(pipeline)}</div>
                    </div>
                    <div class="meta-cell">
                        <div class="meta-key">Strategy</div>
                        <div class="meta-val">${escapeHtml(strategy)}</div>
                    </div>
                    <div class="meta-cell">
                        <div class="meta-key">Cameras</div>
                        <div class="meta-val">${cameraCount}</div>
                    </div>
                    <div class="meta-cell">
                        <div class="meta-key">Classifications</div>
                        <div class="meta-val">${bin.total_entries ?? 0}</div>
                    </div>
                    <div class="meta-cell">
                        <div class="meta-key">Auto</div>
                        <div class="meta-val">${bin.auto_classify ? 'ON' : 'OFF'}</div>
                    </div>
                    <div class="meta-cell">
                        <div class="meta-key">Running</div>
                        <div class="meta-val">${running ? 'YES' : 'NO'}</div>
                    </div>
                </div>

                ${renderWarnings(bin.warnings)}

                <div class="bin-controls">
                    <label class="select-wrap">
                        <span class="lbl">Strategy</span>
                        <select data-control="strategy" ${strategyDisabled ? 'disabled title="Only available on OAK-Dual pipeline"' : ''}>${strategyOptions}</select>
                    </label>
                    <label class="select-wrap">
                        <span class="lbl">Pipeline</span>
                        <select data-control="pipeline">${pipelineOptions}</select>
                    </label>
                    <div class="bin-actions">
                        ${lifecycleBtn}
                        <button class="btn-secondary" data-action="restart">↻ Restart</button>
                    </div>
                    <div class="bin-actions">
                        <button class="btn-primary" data-action="classify" ${!running || !hasHost ? 'disabled' : ''}>📸 Classify</button>
                        <button class="btn-ghost" data-action="toggle_auto" ${!running || !hasHost ? 'disabled' : ''}>${bin.auto_classify ? 'Auto: ON' : 'Auto: OFF'}</button>
                    </div>
                    <a class="open-detail" href="/bin/${encodeURIComponent(bin.bin_id)}">Open detail view →</a>
                </div>
            </article>
        `;
    }

    function applyFilter(bins) {
        if (!filter) return bins;
        const q = filter.toLowerCase();
        return bins.filter((b) =>
            (b.bin_id || '').toLowerCase().includes(q) ||
            (b.location || '').toLowerCase().includes(q)
        );
    }

    function render() {
        const bins = applyFilter(lastBins);
        if (!bins.length) {
            grid.innerHTML = `<div class="empty-state"><h2>${lastBins.length ? 'No matches' : 'No bins registered'}</h2>
                <p>${lastBins.length ? 'Clear the filter to see all bins.' : 'Start an edge device to see it here.'}</p></div>`;
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
                ? `Process will exit and the container will restart. There will be 5–15 seconds of downtime on ${binId}.`
                : `Bin ${binId} will stop accepting classifications until you press Start again.`;
            const ok = await confirmModal(title, body);
            if (!ok) return;
        }
        sendCommand(binId, action);
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
            card.querySelectorAll('button[data-action]').forEach((btn) => {
                btn.addEventListener('click', (ev) => {
                    ev.preventDefault();
                    const action = btn.dataset.action;
                    btn.disabled = true;
                    onAction(card, action).finally(() => { btn.disabled = false; });
                });
            });
            card.querySelectorAll('select[data-control]').forEach((sel) => {
                sel.addEventListener('change', (ev) => {
                    const control = sel.dataset.control;
                    onSelectChange(card, control, sel.value);
                });
            });
        });
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
            document.getElementById('online-bins').textContent = data.online ?? 0;
            document.getElementById('degraded-bins').textContent = data.degraded ?? 0;
            document.getElementById('offline-bins').textContent = data.offline ?? 0;
            document.getElementById('total-classifications').textContent = data.total_entries ?? 0;
            render();
        } catch (e) { /* ignore transient errors */ }
    }

    filterInput.addEventListener('input', () => {
        filter = filterInput.value.trim();
        render();
    });
    refreshBtn.addEventListener('click', refresh);

    refresh();
    setInterval(refresh, POLL_INTERVAL_MS);
})();
