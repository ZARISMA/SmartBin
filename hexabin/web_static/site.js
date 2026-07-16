/* HexaBin marketing site — Leaflet map + live stats + animated counter */
(function () {
    'use strict';

    // ── Palette (reads brand.css tokens so dark-theme remaps apply) ──
    const CAT_TOKENS = {
        Plastic:  '--sb-plastic',
        Paper:    '--sb-paper',
        Glass:    '--sb-glass',
        Organic:  '--sb-organic',
        Aluminum: '--sb-aluminum',
        Other:    '--sb-other-1',
        Empty:    '--sb-empty',
    };
    function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }
    function catColor(name) {
        return cssVar(CAT_TOKENS[name] || '--sb-empty') || '#8C8C8C';
    }
    function currentTheme() {
        return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
    }
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
        const el = document.getElementById('hexabin-map');
        if (!el || typeof L === 'undefined') return;
        const map = L.map(el, {
            center: [40.1792, 44.5152],
            zoom: 13,
            zoomControl: true,
            scrollWheelZoom: false,
        });
        const BASEMAPS = {
            light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            dark:  'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        };
        const TILE_OPTS = {
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            subdomains: 'abcd',
            maxZoom: 19,
        };
        let tiles = L.tileLayer(BASEMAPS[currentTheme()], TILE_OPTS).addTo(map);
        window.addEventListener('hexabin:themechange', () => {
            map.removeLayer(tiles);
            tiles = L.tileLayer(BASEMAPS[currentTheme()], TILE_OPTS).addTo(map);
        });

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

    // ── Stats (feeds only the hero badge + latest-classification chip;
    //    the Statistics section is capability framing until launch) ────
    async function fetchStats() {
        // Public aggregate endpoint (no auth) — hero keeps its static example on failure
        try {
            const res = await fetch('/api/public/stats', { cache: 'no-store' });
            if (!res.ok) return null;
            return await res.json();
        } catch (e) { return null; }
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
        const chip = document.getElementById('chip-classify');
        if (chip && stats.latest && stats.latest.category) {
            const l = stats.latest;
            const swatch = document.getElementById('chip-classify-swatch');
            swatch.textContent = l.category.slice(0, 2).toUpperCase();
            swatch.style.background = l.category in CAT_TOKENS ? catColor(l.category) : '#E0E0E0';
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

    let lastStats = null;

    async function refreshStats() {
        const stats = await fetchStats();
        if (!stats) return;
        lastStats = stats;
        renderHero(stats);
    }

    function bootStats() {
        refreshStats();
        setInterval(refreshStats, 5000);
        // Repaint category-colored elements immediately on theme switch.
        window.addEventListener('hexabin:themechange', () => {
            if (lastStats) renderHero(lastStats);
        });
    }

    // ── 3D model viewer ───────────────────────────────────────────────
    function initModelViewer() {
        const mv = document.querySelector('.media-card-3d model-viewer');
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
            const ctrls = document.getElementById('model-ctrls');
            if (ctrls) ctrls.hidden = true;
        }, { once: true });
    }

    function resetViewer(mv) {
        if (!mv) return;
        // Interaction moves the camera without touching the attributes, so
        // re-applying them animates the view back to its authored default.
        // 'auto' is the spec default for attributes the markup never set.
        ['camera-orbit', 'camera-target', 'field-of-view'].forEach((name) => {
            const v = mv.getAttribute(name);
            mv.setAttribute(name, v !== null ? v : 'auto');
        });
        if (typeof mv.resetTurntableRotation === 'function') mv.resetTurntableRotation();
    }

    // ── 3D model popup (expand button on the media card) ─────────────
    function initModelModal() {
        const btn = document.getElementById('model-expand');
        const modal = document.getElementById('model-modal');
        const stage = document.getElementById('model-modal-stage');
        if (!btn || !modal || !stage) return;

        function open() {
            const cardViewer = document.querySelector('.media-card-3d model-viewer');
            if (!cardViewer) return;
            if (!stage.querySelector('model-viewer')) {
                // Clone attributes from the card viewer (GLB comes from HTTP cache).
                const mv = cardViewer.cloneNode(false);
                mv.removeAttribute('loading');
                stage.appendChild(mv);
            }
            modal.hidden = false;
            document.body.style.overflow = 'hidden';
        }
        function close() {
            modal.hidden = true;
            document.body.style.overflow = '';
            const mv = stage.querySelector('model-viewer');
            if (mv) mv.remove(); // free the WebGL context while closed
        }

        btn.addEventListener('click', open);
        modal.querySelectorAll('[data-model-close]').forEach((el) => {
            el.addEventListener('click', close);
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !modal.hidden) close();
        });

        const cardReset = document.getElementById('model-reset');
        if (cardReset) {
            cardReset.addEventListener('click', () => {
                resetViewer(document.querySelector('.media-card-3d model-viewer'));
            });
        }
        const modalReset = document.getElementById('model-modal-reset');
        if (modalReset) {
            modalReset.addEventListener('click', () => {
                resetViewer(stage.querySelector('model-viewer'));
            });
        }
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
        initModelModal();
        initReveal();
    });
})();
