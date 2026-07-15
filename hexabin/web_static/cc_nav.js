/* HexaBin Control Center shared nav helpers.
 * Keeps the sidebar alerts badge current on every page that extends
 * _cc_base.html. Failures are silent — a background badge poll must never
 * redirect to /login or surface an error.
 */
(function () {
    'use strict';

    const badge = document.getElementById('cc-alerts-badge');
    if (!badge) return;

    async function refreshBadge() {
        try {
            const res = await fetch('/api/alerts', { cache: 'no-store' });
            if (!res.ok) return;
            const data = await res.json();
            const total = (data.counts && data.counts.total) || 0;
            badge.textContent = total;
            badge.hidden = total === 0;
        } catch (e) { /* silent */ }
    }

    refreshBadge();
    setInterval(refreshBadge, 15000);
})();
