/* SmartBin Alerts view
 * Polls /api/alerts and renders camera-availability alerts per reporting bin.
 */
(function () {
    'use strict';

    const POLL_INTERVAL_MS = 5000;
    const list = document.getElementById('alerts-list');

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

    function pipelineLabel(p) {
        if (p === 'oak') return 'OAK Dual';
        if (p === 'oak-native') return 'OAK Native';
        if (p === 'raspberry') return 'Raspberry';
        return p || '—';
    }

    function renderAlert(a) {
        const icon = a.severity === 'error' ? '×' : '⚠';
        return `
            <article class="alert-row sev-${escapeHtml(a.severity)}">
                <div class="alert-icon">${icon}</div>
                <div class="alert-body">
                    <div class="alert-message">${escapeHtml(a.message)}</div>
                    <div class="alert-meta cc-mono">
                        <a href="/bin/${encodeURIComponent(a.bin_id)}">${escapeHtml(a.bin_id)}</a>
                        · ${escapeHtml(pipelineLabel(a.pipeline))}
                        · cameras: ${a.camera_count ?? 0}
                        ${a.host ? '· ' + escapeHtml(a.host) : ''}
                    </div>
                </div>
                <div class="alert-time cc-mono">${escapeHtml(lastSeenLabel(a.last_seen))}</div>
            </article>
        `;
    }

    function render(data) {
        const alerts = data.alerts || [];
        const counts = data.counts || {};
        document.getElementById('alert-stat-error').textContent = counts.error ?? 0;
        document.getElementById('alert-stat-warning').textContent = counts.warning ?? 0;
        document.getElementById('alert-stat-monitored').textContent = counts.monitored ?? 0;
        const badge = document.getElementById('cc-alerts-badge');
        if (badge) {
            badge.textContent = counts.total ?? 0;
            badge.hidden = !(counts.total > 0);
        }
        if (!alerts.length) {
            list.innerHTML = `<div class="empty-state">
                <h2>All clear</h2>
                <p>Every reporting bin has healthy cameras. Bins without a recent heartbeat
                   are not monitored here — see the Fleet page for offline detection.</p>
            </div>`;
            return;
        }
        list.innerHTML = alerts.map(renderAlert).join('');
    }

    async function refresh() {
        try {
            const res = await fetch('/api/alerts', { cache: 'no-store' });
            if (!res.ok) {
                if (res.status === 401) { window.location.href = '/login'; }
                return;
            }
            render(await res.json());
            const now = new Date();
            const pad = (n) => String(n).padStart(2, '0');
            const el = document.getElementById('alerts-refresh');
            if (el) el.textContent = `updated ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
            const hostEl = document.getElementById('cc-server-host');
            if (hostEl) hostEl.textContent = window.location.host || '—';
        } catch (e) { /* transient errors are non-fatal */ }
    }

    refresh();
    setInterval(refresh, POLL_INTERVAL_MS);
})();
