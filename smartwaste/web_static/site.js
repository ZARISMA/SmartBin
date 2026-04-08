/* ═══════════════════════════════════════════════════════════════════════════
   SmartBin — Presentation Website JavaScript
   ═══════════════════════════════════════════════════════════════════════════ */

const CAT_COLORS = {
    Plastic:   '#87CEEB',
    Glass:     '#40E0D0',
    Paper:     '#D2B48C',
    Organic:   '#1E4D2B',
    Aluminum:  '#A9A9A9',
    Other:     '#9370DB',
    Empty:     '#8C8C8C',
};

const SMARTBIN_LOCATIONS = [
    { name: 'Republic Square',       lat: 40.1777, lng: 44.5126, desc: 'Central transportation hub' },
    { name: 'Cascade Complex',       lat: 40.1922, lng: 44.5155, desc: 'Cultural landmark & park' },
    { name: 'Northern Avenue',       lat: 40.1850, lng: 44.5100, desc: 'Pedestrian shopping street' },
    { name: 'Yerevan State University', lat: 40.1876, lng: 44.5148, desc: 'University campus' },
    { name: 'Vernissage Market',     lat: 40.1748, lng: 44.5175, desc: 'Open-air market' },
    { name: 'Tsitsernakaberd',       lat: 40.1853, lng: 44.4903, desc: 'Memorial park area' },
    { name: 'Yerevan Mall',          lat: 40.2050, lng: 44.5080, desc: 'Shopping center' },
    { name: 'Dalma Garden Mall',     lat: 40.1590, lng: 44.5260, desc: 'Southern shopping area' },
];

/* ── Scroll Animations ─────────────────────────────────────────────────────── */

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

/* ── Navbar ────────────────────────────────────────────────────────────────── */

const navbar = document.querySelector('.navbar');
const hamburger = document.querySelector('.nav-hamburger');
const navLinks = document.querySelector('.nav-links');

window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 50);
});

if (hamburger) {
    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('open');
    });
    /* Close menu on link click */
    navLinks.querySelectorAll('a').forEach(a => {
        a.addEventListener('click', () => navLinks.classList.remove('open'));
    });
}

/* Active section highlighting */
const sections = document.querySelectorAll('.section[id], .hero[id]');
const navAnchors = document.querySelectorAll('.nav-links a');

window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(s => {
        const top = s.offsetTop - 120;
        if (window.scrollY >= top) current = s.id;
    });
    navAnchors.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
});

/* ── Animated Counter ──────────────────────────────────────────────────────── */

function animateCounter(el, target, duration = 2000) {
    const start = performance.now();
    const update = (now) => {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        /* Ease-out cubic */
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.floor(eased * target).toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
}

/* ── Statistics ─────────────────────────────────────────────────────────────── */

let statsLoaded = false;

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        const total = data.total || 0;
        const cats = data.by_category || {};

        /* Total counter */
        const totalEl = document.getElementById('stats-total');
        if (totalEl) {
            totalEl.dataset.target = total;
        }

        /* Category bars */
        const barsContainer = document.getElementById('stats-bars');
        if (!barsContainer) return;

        if (total === 0) {
            barsContainer.innerHTML = '<p class="stats-zero-state">Deployment starting soon — statistics will appear here once SmartBins are active.</p>';
            return;
        }

        const maxCount = Math.max(...Object.values(cats), 1);

        barsContainer.innerHTML = Object.entries(cats)
            .filter(([label]) => label !== 'Empty')
            .sort((a, b) => b[1] - a[1])
            .map(([label, count]) => {
                const color = CAT_COLORS[label] || '#8C8C8C';
                const pct = (count / maxCount) * 100;
                return `
                    <div class="stat-bar-card glass-card">
                        <div class="stat-bar-header">
                            <span class="stat-bar-label">${label}</span>
                            <span class="stat-bar-count">${count.toLocaleString()}</span>
                        </div>
                        <div class="stat-bar-track">
                            <div class="stat-bar-fill" style="background:${color}" data-width="${pct}%"></div>
                        </div>
                    </div>`;
            }).join('');
    } catch (e) {
        console.warn('Could not load stats:', e);
    }
}

/* Trigger counter and bars when stats section scrolls into view */
const statsSection = document.getElementById('statistics');
if (statsSection) {
    const statsObs = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !statsLoaded) {
                statsLoaded = true;

                /* Animate total */
                const totalEl = document.getElementById('stats-total');
                if (totalEl) {
                    const target = parseInt(totalEl.dataset.target) || 0;
                    animateCounter(totalEl, target);
                }

                /* Animate bars */
                setTimeout(() => {
                    document.querySelectorAll('.stat-bar-fill').forEach(bar => {
                        bar.style.width = bar.dataset.width;
                    });
                }, 300);
            }
        });
    }, { threshold: 0.2 });
    statsObs.observe(statsSection);
}

/* ── Leaflet Map ───────────────────────────────────────────────────────────── */

function initMap() {
    const mapEl = document.getElementById('smartbin-map');
    if (!mapEl || typeof L === 'undefined') return;

    const map = L.map('smartbin-map', {
        center: [40.1792, 44.4991],
        zoom: 13,
        scrollWheelZoom: false,
        attributionControl: false,
    });

    L.control.attribution({ prefix: false }).addTo(map);

    /* Dark-toned tile layer */
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        maxZoom: 19,
    }).addTo(map);

    /* Custom SVG marker icon */
    const smartBinIcon = L.divIcon({
        className: 'smartbin-marker',
        html: `<svg width="32" height="42" viewBox="0 0 32 42" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M16 0C7.16 0 0 7.16 0 16c0 12 16 26 16 26s16-14 16-26C32 7.16 24.84 0 16 0z" fill="#2D5A42"/>
            <circle cx="16" cy="16" r="8" fill="#4a8a6a"/>
            <path d="M13 12.5l6 3.5-6 3.5v-7z" fill="#fff"/>
        </svg>`,
        iconSize: [32, 42],
        iconAnchor: [16, 42],
        popupAnchor: [0, -44],
    });

    SMARTBIN_LOCATIONS.forEach(loc => {
        L.marker([loc.lat, loc.lng], { icon: smartBinIcon })
            .bindPopup(`<strong>${loc.name}</strong><br>${loc.desc}<br><span class="popup-status">Coming Soon</span>`)
            .addTo(map);
    });
}

/* ── Init ──────────────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', async () => {
    await loadStats();
    initMap();
});
