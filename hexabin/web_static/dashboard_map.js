/* HexaBin Map view — operator deployment map
 * Plots one pin per bin from /api/dashboard, colored by status.
 * Online pins get a pulse halo. Clicking opens the floating detail panel.
 */
(function () {
    'use strict';

    if (typeof L === 'undefined') return;

    const YEREVAN_CENTER = [40.1792, 44.5152];
    const STATUS_COLOR = {
        online:   '#4CAF50',
        degraded: '#FF9800',
        offline:  '#C62828',
        stopped:  '#8C8C8C',
    };

    // Fallback coordinates if /api/dashboard doesn't yet carry lat/lng.
    const FALLBACK_COORDS = {
        'bin-01': { lat: 40.1776, lng: 44.5126, loc: 'Republic Square' },
        'bin-02': { lat: 40.1894, lng: 44.5183, loc: 'Cascade Complex' },
        'bin-03': { lat: 40.1781, lng: 44.5142, loc: 'Vernissage Market' },
        'bin-04': { lat: 40.1561, lng: 44.4837, loc: 'Yerevan Mall' },
        'bin-05': { lat: 40.1820, lng: 44.5147, loc: 'Northern Avenue' },
        'bin-06': { lat: 40.1858, lng: 44.5142, loc: 'Opera House' },
        'bin-07': { lat: 40.1612, lng: 44.5067, loc: 'Yerevan Station' },
        'bin-08': { lat: 40.1738, lng: 44.5236, loc: "Children's Park" },
    };

    const BASEMAPS = {
        dark:      { url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',  attr: '&copy; OpenStreetMap &copy; CARTO', subdomains: 'abcd' },
        streets:   { url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',             attr: '&copy; OpenStreetMap contributors' },
        satellite: { url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr: 'Tiles &copy; Esri' },
    };

    const map = L.map('cc-map', {
        center: YEREVAN_CENTER,
        zoom: 13,
        zoomControl: false,
        attributionControl: true,
    });
    let currentTiles = L.tileLayer(BASEMAPS.dark.url, { attribution: BASEMAPS.dark.attr, subdomains: BASEMAPS.dark.subdomains }).addTo(map);

    let markers = [];
    let lastBins = [];
    let selectedId = null;
    let mapFilter = 'all';

    function pinDivIcon(color, isOnline) {
        const halo = isOnline ? `<div class="cc-pin-halo" style="border-color:${color};"></div>` : '';
        return L.divIcon({
            className: 'cc-pin',
            html: `<div class="cc-pin-wrap" style="background:${color};">${halo}</div>`,
            iconSize: [18, 18],
            iconAnchor: [9, 9],
        });
    }

    function clearMarkers() {
        markers.forEach((m) => map.removeLayer(m));
        markers = [];
    }

    function updateLegendCounts(bins) {
        const counts = { online: 0, degraded: 0, offline: 0, stopped: 0 };
        bins.forEach((b) => { if (counts[b.status] !== undefined) counts[b.status]++; });
        document.getElementById('legend-online').textContent   = counts.online;
        document.getElementById('legend-degraded').textContent = counts.degraded;
        document.getElementById('legend-offline').textContent  = counts.offline;
        document.getElementById('legend-stopped').textContent  = counts.stopped;
    }

    function pinsFor(bins) {
        if (mapFilter === 'all') return bins;
        if (mapFilter === 'alerts') return bins.filter((b) => (b.warnings && b.warnings.length) || b.status === 'offline' || b.status === 'degraded');
        if (mapFilter === 'needs') return bins.filter((b) => b.status === 'online' || b.status === 'degraded');
        return bins;
    }

    function plot(bins) {
        clearMarkers();
        const toShow = pinsFor(bins);
        toShow.forEach((b) => {
            const fc = FALLBACK_COORDS[b.bin_id];
            const lat = b.lat || (fc && fc.lat);
            const lng = b.lng || (fc && fc.lng);
            if (lat == null || lng == null) return;
            const color = STATUS_COLOR[b.status] || STATUS_COLOR.stopped;
            const marker = L.marker([lat, lng], {
                icon: pinDivIcon(color, b.status === 'online'),
                riseOnHover: true,
            });
            marker.on('click', () => selectBin(b));
            marker.addTo(map);
            markers.push(marker);
        });
        updateLegendCounts(bins);
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
        return p || '—';
    }

    function selectBin(bin) {
        selectedId = bin.bin_id;
        const panel = document.getElementById('map-detail');
        const fc = FALLBACK_COORDS[bin.bin_id];
        const lat = bin.lat || (fc && fc.lat);
        const lng = bin.lng || (fc && fc.lng);
        const loc = bin.location || (fc && fc.loc) || 'Location not set';

        panel.hidden = false;
        const statusPill = document.getElementById('detail-status');
        statusPill.className = 'sb-pill ' + bin.status;
        statusPill.querySelector('[data-field="status"]').textContent = bin.status;

        document.getElementById('detail-loc').textContent = loc;
        document.getElementById('detail-id').textContent = `${bin.bin_id}${lat != null ? ' · ' + lat.toFixed(4) + '°N ' + lng.toFixed(4) + '°E' : ''}`;
        document.getElementById('detail-today').textContent = bin.total_entries ?? 0;
        document.getElementById('detail-pipeline').textContent = pipelineLabel(bin.pipeline || bin.camera_mode);
        document.getElementById('detail-last').textContent = lastSeenLabel(bin.last_seen);

        const warnEl = document.getElementById('detail-warn');
        if (bin.warnings && bin.warnings.length) {
            warnEl.hidden = false;
            warnEl.innerHTML = bin.warnings.map((w) =>
                `<div class="bin-warning sev-${w.severity || 'warning'}"><span class="sev">⚠</span><span>${w.message || w.code}</span></div>`
            ).join('');
        } else {
            warnEl.hidden = true;
        }

        document.getElementById('detail-open').href = `/bin/${encodeURIComponent(bin.bin_id)}`;

        if (lat != null && lng != null) map.panTo([lat, lng], { animate: true });
    }

    function switchBasemap(key) {
        const b = BASEMAPS[key];
        if (!b) return;
        map.removeLayer(currentTiles);
        const opts = { attribution: b.attr };
        if (b.subdomains) opts.subdomains = b.subdomains;
        currentTiles = L.tileLayer(b.url, opts).addTo(map);
    }

    // ── Controls ──────────────────────────────────────────────────────
    document.querySelectorAll('.bm-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.bm-btn').forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            switchBasemap(btn.dataset.basemap);
        });
    });

    document.getElementById('zoom-in').addEventListener('click', () => map.zoomIn());
    document.getElementById('zoom-out').addEventListener('click', () => map.zoomOut());
    document.getElementById('zoom-home').addEventListener('click', () => map.setView(YEREVAN_CENTER, 13));

    document.getElementById('detail-close').addEventListener('click', () => {
        document.getElementById('map-detail').hidden = true;
        selectedId = null;
    });

    document.querySelectorAll('[data-mapfilter]').forEach((b) => {
        b.addEventListener('click', () => {
            document.querySelectorAll('[data-mapfilter]').forEach((x) => x.classList.remove('active'));
            b.classList.add('active');
            mapFilter = b.dataset.mapfilter;
            plot(lastBins);
        });
    });

    document.getElementById('export-geojson').addEventListener('click', () => {
        const features = lastBins.map((b) => {
            const fc = FALLBACK_COORDS[b.bin_id];
            const lat = b.lat || (fc && fc.lat);
            const lng = b.lng || (fc && fc.lng);
            if (lat == null || lng == null) return null;
            return {
                type: 'Feature',
                properties: { bin_id: b.bin_id, location: b.location, status: b.status, total_entries: b.total_entries },
                geometry: { type: 'Point', coordinates: [lng, lat] },
            };
        }).filter(Boolean);
        const blob = new Blob([JSON.stringify({ type: 'FeatureCollection', features }, null, 2)], { type: 'application/geo+json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'hexabin-fleet.geojson';
        a.click();
    });

    // ── Poll /api/dashboard ────────────────────────────────────────────
    async function refresh() {
        try {
            const res = await fetch('/api/dashboard', { cache: 'no-store' });
            if (!res.ok) {
                if (res.status === 401) { window.location.href = '/login'; return; }
                return;
            }
            const data = await res.json();
            lastBins = data.bins || [];
            plot(lastBins);
            // If a bin was selected, refresh its detail.
            if (selectedId) {
                const b = lastBins.find((x) => x.bin_id === selectedId);
                if (b) selectBin(b);
            }
            const hostEl = document.getElementById('cc-server-host');
            if (hostEl) hostEl.textContent = window.location.host || '—';
        } catch (e) { /* transient */ }
    }

    refresh();
    setInterval(refresh, 5000);
})();
