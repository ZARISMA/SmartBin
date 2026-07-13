/* SmartBin marketing site — Leaflet map + live stats + animated counter */
(function () {
    'use strict';

    // ── Palette (mirrors brand.css) ──────────────────────────────────
    const COLORS = {
        Plastic:  '#87CEEB',
        Paper:    '#D2B48C',
        Glass:    '#40E0D0',
        Organic:  '#1E4D2B',
        Aluminum: '#A9A9A9',
        Other:    '#9370DB',
        Empty:    '#8C8C8C',
    };
    const LIGHT_TEXT = new Set(['Organic']); // bars where the serif number should be white

    // ── Yerevan landmark pins ─────────────────────────────────────────
    const PINS = [
        { lat: 40.1776, lng: 44.5126, name: 'Republic Square',     wave: 1 },
        { lat: 40.1894, lng: 44.5183, name: 'Cascade Complex',     wave: 1 },
        { lat: 40.1781, lng: 44.5142, name: 'Vernissage Market',   wave: 1 },
        { lat: 40.1561, lng: 44.4837, name: 'Yerevan Mall',        wave: 1 },
        { lat: 40.1820, lng: 44.5147, name: 'Northern Avenue',     wave: 1 },
        { lat: 40.1858, lng: 44.5142, name: 'Opera House',         wave: 2 },
        { lat: 40.1612, lng: 44.5067, name: 'Yerevan Station',     wave: 2 },
        { lat: 40.1738, lng: 44.5236, name: "Children's Park",     wave: 2 },
    ];

    // ── Leaflet map ───────────────────────────────────────────────────
    function initMap() {
        const el = document.getElementById('smartbin-map');
        if (!el || typeof L === 'undefined') return;
        const map = L.map(el, {
            center: [40.1792, 44.5152],
            zoom: 13,
            zoomControl: true,
            scrollWheelZoom: false,
        });
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            subdomains: 'abcd',
            maxZoom: 19,
        }).addTo(map);

        PINS.forEach((p) => {
            const color = p.wave === 1 ? '#2D5A42' : '#BDB76B';
            const icon = L.divIcon({
                className: 'sb-leaflet-pin',
                html: `<div class="sb-pin-wrap">
                         <div class="sb-pin-label">${p.name}</div>
                         <div class="sb-pin-dot" style="background:${color};"></div>
                       </div>`,
                iconSize: [120, 44],
                iconAnchor: [60, 44],
            });
            L.marker([p.lat, p.lng], { icon }).addTo(map);
        });
    }

    // ── Stats ─────────────────────────────────────────────────────────
    async function fetchStats() {
        // Public aggregate endpoint (no auth) — fall back to the preview demo on failure
        try {
            const res = await fetch('/api/public/stats', { cache: 'no-store' });
            if (!res.ok) return null;
            return await res.json();
        } catch (e) { return null; }
    }

    function renderBars(stats) {
        const host = document.getElementById('stats-bars');
        if (!host) return;
        const order = ['Plastic', 'Paper', 'Glass', 'Organic', 'Aluminum', 'Other'];
        const data = order.map((name) => ({ name, value: (stats.by_category || {})[name] || 0 }));
        if (!data.some((d) => d.value > 0)) {
            host.innerHTML = '<p class="stats-zero-state">No classifications recorded yet.</p>';
            return;
        }
        const max = Math.max(1, ...data.map((d) => d.value));
        const total = data.reduce((a, d) => a + d.value, 0) || 1;

        host.innerHTML = data.map((d) => {
            const pct = Math.max(8, Math.round((d.value / max) * 100));
            const portion = Math.round((d.value / total) * 100);
            const txtLight = LIGHT_TEXT.has(d.name) ? 'light' : '';
            return `<div class="stat-bar">
                <div class="stat-bar-fill" style="height:${pct}%; background:${COLORS[d.name]};">
                    <span class="stat-bar-num ${txtLight}">${d.value}</span>
                </div>
                <div class="stat-bar-label">
                    <span class="name">${d.name}</span>
                    <span class="pct">${portion}%</span>
                </div>
            </div>`;
        }).join('');
    }

    function animateNumber(el, target, duration = 1600) {
        const start = parseInt(el.dataset.target || '0', 10) || 0;
        const t0 = performance.now();
        function tick(now) {
            const p = Math.min(1, (now - t0) / duration);
            const e = 1 - Math.pow(1 - p, 3); // easeOutCubic
            el.textContent = Math.round(start + (target - start) * e).toLocaleString();
            if (p < 1) requestAnimationFrame(tick);
            else el.dataset.target = String(target);
        }
        requestAnimationFrame(tick);
    }

    function timeAgo(seconds) {
        if (seconds == null) return '';
        if (seconds < 60) return 'just now';
        if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
        if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
        return Math.floor(seconds / 86400) + 'd ago';
    }

    function renderHero(stats) {
        const badge = document.getElementById('hero-badge-text');
        if (badge && stats.bins && stats.bins.total > 0) {
            const b = stats.bins;
            badge.textContent = `Live · ${b.online} of ${b.total} bin${b.total === 1 ? '' : 's'} online`;
        }
        const today = document.getElementById('hero-today');
        if (today && typeof stats.today === 'number') {
            today.textContent = `${stats.today.toLocaleString()} items sorted`;
        }
        const chip = document.getElementById('chip-classify');
        if (chip && stats.latest && stats.latest.category) {
            const l = stats.latest;
            const swatch = document.getElementById('chip-classify-swatch');
            swatch.textContent = l.category.slice(0, 2).toUpperCase();
            swatch.style.background = COLORS[l.category] || '#E0E0E0';
            swatch.style.color = LIGHT_TEXT.has(l.category) ? '#fff' : '';
            document.getElementById('chip-classify-title').textContent = l.item || l.category;
            const parts = [];
            if (typeof l.confidence === 'number') {
                parts.push(`${Math.round(l.confidence * 100)}% confidence`);
            }
            const ago = timeAgo(l.ago_seconds);
            if (ago) parts.push(ago);
            document.getElementById('chip-classify-sub').textContent =
                parts.join(' · ') || l.category;
            chip.style.display = '';
        }
    }

    function renderKpis(stats) {
        const bins = document.getElementById('stats-active-bins');
        if (bins && stats.bins) bins.textContent = `${stats.bins.online} / ${stats.bins.total}`;
        const rec = document.getElementById('stats-recyclable');
        if (rec) {
            rec.textContent = stats.recyclable_share == null
                ? '—'
                : Math.round(stats.recyclable_share * 100) + '%';
        }
    }

    async function refreshStats() {
        const stats = await fetchStats();
        if (!stats) {
            const host = document.getElementById('stats-bars');
            if (host) host.innerHTML = '<p class="stats-zero-state">Live statistics are unavailable right now.</p>';
            return;
        }
        renderBars(stats);
        renderHero(stats);
        renderKpis(stats);
        const totalEl = document.getElementById('stats-total');
        if (totalEl) animateNumber(totalEl, stats.total || 0);
    }

    function bootStats() {
        refreshStats();
        setInterval(refreshStats, 5000); // section copy promises 5-second updates
    }

    // ── 3D model viewer ───────────────────────────────────────────────
    function initModelViewer() {
        const mv = document.querySelector('model-viewer');
        if (!mv) return;
        // If the model fails to load/decode, degrade to the static poster
        // instead of leaving a broken canvas in the card.
        mv.addEventListener('error', () => {
            const fallback = document.createElement('div');
            fallback.className = 'model-poster';
            fallback.innerHTML =
                '<div class="media-stripes"></div>' +
                '<div class="model-poster-label">3D preview unavailable</div>';
            mv.replaceWith(fallback);
        }, { once: true });
    }

    // ── Reveal-on-scroll polish (no-op when reduced motion) ──────────
    function initReveal() {
        if (!('IntersectionObserver' in window)) return;
        const els = document.querySelectorAll('.module-card, .about-stat-cell, .media-card');
        const io = new IntersectionObserver((entries) => {
            entries.forEach((e) => {
                if (e.isIntersecting) {
                    e.target.style.animation = 'sbFadeIn 480ms ease-out both';
                    io.unobserve(e.target);
                }
            });
        }, { threshold: 0.15 });
        els.forEach((el) => io.observe(el));
    }

    // ── Boot ──────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        initMap();
        bootStats();
        initModelViewer();
        initReveal();
    });
})();
