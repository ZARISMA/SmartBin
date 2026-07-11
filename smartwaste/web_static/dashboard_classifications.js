/* SmartBin Classifications browser
 * Server-side pagination + filters over /api/entries (+ /api/entries/count),
 * thumbnails and a lightbox served from /api/entries/{id}/image.
 */
(function () {
    'use strict';

    const LIMIT = 20;
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
    const DARK_CHIPS = new Set(['Organic', 'Other']);

    const table = document.getElementById('class-table');
    const recordsCount = document.getElementById('records-count');
    const pageLabel = document.getElementById('page-label');
    const prevBtn = document.getElementById('page-prev');
    const nextBtn = document.getElementById('page-next');
    const binSelect = document.getElementById('filter-bin');
    const catSelect = document.getElementById('filter-category');
    const qInput = document.getElementById('filter-q');
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxCap = document.getElementById('lightbox-cap');

    const state = { page: 0, bin: '', label: '', q: '' };

    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    }

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

    function filterParams() {
        const p = new URLSearchParams();
        if (state.bin) p.set('bin_id', state.bin);
        if (state.label) p.set('label', state.label);
        if (state.q) p.set('q', state.q);
        return p;
    }

    function chip(label) {
        const cat = label === 'Empty' ? 'Empty' : (CAT.includes(label) ? label : 'Other');
        const dark = DARK_CHIPS.has(cat) ? ' dark' : '';
        return `<span class="cat-chip${dark}" style="background:${COLORS[cat]};">${escapeHtml(label || '—')}</span>`;
    }

    function fmtTs(ts) {
        const s = String(ts || '');
        const [d, t] = s.split(' ');
        if (!t) return `<span class="class-ts">${escapeHtml(s || '—')}</span>`;
        return `<span class="class-ts">${escapeHtml(d)}<br><span class="cc-mono">${escapeHtml(t)}</span></span>`;
    }

    function renderRows(entries) {
        if (!entries.length) {
            table.innerHTML = `<div class="empty-state">
                <h2>No classifications</h2>
                <p>Nothing matches the current filters.</p>
            </div>`;
            return;
        }
        const head = `
            <div class="class-row class-row-head">
                <span></span><span>Time</span><span>Category</span><span>Description</span>
                <span>Brand</span><span>Bin</span><span>Conf.</span><span>Backend</span>
            </div>`;
        table.innerHTML = head + entries.map((e) => {
            const caption = `${e.label || '—'} · ${e.bin_id || '—'} · ${e.timestamp || ''}`;
            return `
            <div class="class-row" data-id="${e.id}" data-cap="${escapeHtml(caption)}">
                <img class="class-thumb" loading="lazy" alt=""
                     data-src="/api/entries/${e.id}/image">
                ${fmtTs(e.timestamp)}
                <span>${chip(e.label)}</span>
                <span class="class-desc" title="${escapeHtml(e.description || '')}">${escapeHtml(e.description || '—')}</span>
                <span class="class-brand">${escapeHtml(e.brand_product || '—')}</span>
                <a class="class-bin cc-mono" href="/bin/${encodeURIComponent(e.bin_id || '')}">${escapeHtml(e.bin_id || '—')}</a>
                <span class="cc-mono">${e.confidence == null ? '—' : Math.round(e.confidence * 100) + '%'}</span>
                <span class="cc-mono">${escapeHtml(e.llm_backend || 'unknown')}</span>
            </div>`;
        }).join('');
        wireRowHandlers();
    }

    function wireRowHandlers() {
        // Attach the error handler before setting src so a fast 404 can't race it.
        table.querySelectorAll('.class-thumb[data-src]').forEach((img) => {
            img.addEventListener('error', () => {
                const ph = document.createElement('div');
                ph.className = 'class-thumb class-thumb-missing';
                ph.textContent = '∅';
                img.replaceWith(ph);
            });
            img.src = img.dataset.src;
        });
        table.querySelectorAll('.class-row[data-id]').forEach((row) => {
            row.addEventListener('click', (ev) => {
                if (ev.target.closest('a')) return; // let the bin link navigate
                if (row.querySelector('.class-thumb-missing')) return; // no image to show
                lightboxImg.src = `/api/entries/${row.dataset.id}/image`;
                lightboxCap.textContent = row.dataset.cap || '';
                lightbox.hidden = false;
            });
        });
    }

    lightbox.addEventListener('click', () => {
        lightbox.hidden = true;
        lightboxImg.removeAttribute('src');
    });

    async function load() {
        const params = filterParams();
        const listParams = new URLSearchParams(params);
        listParams.set('limit', LIMIT);
        listParams.set('offset', state.page * LIMIT);
        const [entries, count] = await Promise.all([
            fetchJSON('/api/entries?' + listParams.toString()),
            fetchJSON('/api/entries/count?' + params.toString()),
        ]);
        const total = (count && count.total) || 0;
        renderRows(entries || []);
        const pages = Math.max(1, Math.ceil(total / LIMIT));
        state.page = Math.min(state.page, pages - 1);
        recordsCount.textContent = `${total.toLocaleString()} records`;
        pageLabel.textContent = `Page ${state.page + 1} of ${pages}`;
        prevBtn.disabled = state.page === 0;
        nextBtn.disabled = (state.page + 1) * LIMIT >= total;
        const hostEl = document.getElementById('cc-server-host');
        if (hostEl) hostEl.textContent = window.location.host || '—';
    }

    async function populateBins() {
        const bins = await fetchJSON('/api/bins');
        if (!bins) return;
        bins.forEach((b) => {
            if (!b.bin_id) return;
            const opt = document.createElement('option');
            opt.value = b.bin_id;
            opt.textContent = b.bin_id;
            binSelect.appendChild(opt);
        });
    }

    binSelect.addEventListener('change', () => { state.bin = binSelect.value; state.page = 0; load(); });
    catSelect.addEventListener('change', () => { state.label = catSelect.value; state.page = 0; load(); });
    let debounceTimer;
    qInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            state.q = qInput.value.trim();
            state.page = 0;
            load();
        }, 300);
    });
    document.getElementById('refresh-btn').addEventListener('click', () => load());
    prevBtn.addEventListener('click', () => { if (state.page > 0) { state.page -= 1; load(); } });
    nextBtn.addEventListener('click', () => { state.page += 1; load(); });

    populateBins();
    load();
})();
