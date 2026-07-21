/* HexaBin camera editor — per-camera rotate / flip / crop.
 *
 * Opens over the per-bin page. For each camera it loads the RAW (untransformed)
 * snapshot, renders it on a <canvas> with the current rotation+flip applied
 * (matching the server's rotate→flip→crop order), and overlays a draggable crop
 * rectangle. Save POSTs every camera to /api/bin/{id}/camera-config.
 *
 * The crop is stored normalized to [0,1] of the *rotated+flipped* frame, so it
 * round-trips through the backend's apply_transform unchanged.
 */
(function () {
    'use strict';

    if (typeof BIN_ID === 'undefined') return;

    const backdrop = document.getElementById('cam-editor');
    const body = document.getElementById('cam-editor-body');
    const toastHost = document.getElementById('toast-host');
    const MAX_W = 380;          // max canvas width (css px)
    const HANDLE = 12;          // corner hit radius (css px)
    const MIN_NORM = 0.05;      // must match camera_config.MIN_CROP_SIZE

    let cams = [];              // [{index, img, rotation, flipH, flipV, crop:[x0,y0,x1,y1], canvas, ds}]

    function toast(message, kind) {
        if (!toastHost) return;
        const el = document.createElement('div');
        el.className = 'toast ' + (kind || '');
        el.textContent = message;
        toastHost.appendChild(el);
        setTimeout(() => { el.classList.add('leave'); setTimeout(() => el.remove(), 260); }, 3000);
    }

    function rotatedDims(nw, nh, rot) {
        return (rot === 90 || rot === 270) ? [nh, nw] : [nw, nh];
    }

    function loadSnapshot(index) {
        return new Promise((resolve) => {
            const img = new Image();
            // Cache-bust so re-opening the editor pulls a fresh frame.
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null);
            img.src = `/api/bin/${encodeURIComponent(BIN_ID)}/camera/${index}/snapshot?t=` + Date.now();
        });
    }

    function drawCam(cam) {
        const { canvas, img, rotation, flipH, flipV } = cam;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!img) {
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#888';
            ctx.font = '13px system-ui';
            ctx.textAlign = 'center';
            ctx.fillText('No camera frame', canvas.width / 2, canvas.height / 2);
            return;
        }
        const nw = img.naturalWidth, nh = img.naturalHeight;
        const ds = cam.ds;
        // rotate→flip in the rotated (screen) frame; see camera_config.apply_transform
        ctx.save();
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.scale((flipH ? -1 : 1) * ds, (flipV ? -1 : 1) * ds);
        ctx.rotate(rotation * Math.PI / 180);
        ctx.drawImage(img, -nw / 2, -nh / 2, nw, nh);
        ctx.restore();

        // Crop overlay (canvas px == normalized × canvas size)
        const [x0, y0, x1, y1] = cam.crop;
        const cw = canvas.width, ch = canvas.height;
        const rx = x0 * cw, ry = y0 * ch, rw = (x1 - x0) * cw, rh = (y1 - y0) * ch;
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(0, 0, cw, ry);
        ctx.fillRect(0, ry + rh, cw, ch - ry - rh);
        ctx.fillRect(0, ry, rx, rh);
        ctx.fillRect(rx + rw, ry, cw - rx - rw, rh);
        ctx.strokeStyle = '#40E0D0';
        ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.fillStyle = '#40E0D0';
        [[rx, ry], [rx + rw, ry], [rx, ry + rh], [rx + rw, ry + rh]].forEach(([hx, hy]) => {
            ctx.fillRect(hx - 4, hy - 4, 8, 8);
        });
    }

    function sizeCanvas(cam) {
        const nw = cam.img ? cam.img.naturalWidth : 640;
        const nh = cam.img ? cam.img.naturalHeight : 480;
        const [rw, rh] = rotatedDims(nw, nh, cam.rotation);
        cam.ds = Math.min(1, MAX_W / rw);
        cam.canvas.width = Math.round(rw * cam.ds);
        cam.canvas.height = Math.round(rh * cam.ds);
    }

    function cornerAt(cam, px, py) {
        const cw = cam.canvas.width, ch = cam.canvas.height;
        const [x0, y0, x1, y1] = cam.crop;
        const corners = {
            nw: [x0 * cw, y0 * ch], ne: [x1 * cw, y0 * ch],
            sw: [x0 * cw, y1 * ch], se: [x1 * cw, y1 * ch],
        };
        for (const k in corners) {
            const [hx, hy] = corners[k];
            if (Math.abs(px - hx) <= HANDLE && Math.abs(py - hy) <= HANDLE) return k;
        }
        return null;
    }

    function insideCrop(cam, px, py) {
        const cw = cam.canvas.width, ch = cam.canvas.height;
        const [x0, y0, x1, y1] = cam.crop;
        return px >= x0 * cw && px <= x1 * cw && py >= y0 * ch && py <= y1 * ch;
    }

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function attachDrag(cam) {
        const canvas = cam.canvas;
        let mode = null;         // 'move' | corner key
        let start = null;        // {px, py, crop}

        function toCanvasPx(ev) {
            const rect = canvas.getBoundingClientRect();
            const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
            return [(ev.clientX - rect.left) * sx, (ev.clientY - rect.top) * sy];
        }

        canvas.addEventListener('pointerdown', (ev) => {
            const [px, py] = toCanvasPx(ev);
            mode = cornerAt(cam, px, py) || (insideCrop(cam, px, py) ? 'move' : null);
            if (!mode) return;
            start = { px, py, crop: cam.crop.slice() };
            canvas.setPointerCapture(ev.pointerId);
        });

        canvas.addEventListener('pointermove', (ev) => {
            if (!mode) return;
            const cw = canvas.width, ch = canvas.height;
            const [px, py] = toCanvasPx(ev);
            const dx = (px - start.px) / cw, dy = (py - start.py) / ch;
            let [x0, y0, x1, y1] = start.crop;
            if (mode === 'move') {
                const w = x1 - x0, h = y1 - y0;
                x0 = clamp(x0 + dx, 0, 1 - w); y0 = clamp(y0 + dy, 0, 1 - h);
                x1 = x0 + w; y1 = y0 + h;
            } else {
                if (mode.includes('w')) x0 = clamp(x0 + dx, 0, x1 - MIN_NORM);
                if (mode.includes('e')) x1 = clamp(x1 + dx, x0 + MIN_NORM, 1);
                if (mode.includes('n')) y0 = clamp(y0 + dy, 0, y1 - MIN_NORM);
                if (mode.includes('s')) y1 = clamp(y1 + dy, y0 + MIN_NORM, 1);
            }
            cam.crop = [x0, y0, x1, y1];
            drawCam(cam);
        });

        const end = () => { mode = null; start = null; };
        canvas.addEventListener('pointerup', end);
        canvas.addEventListener('pointercancel', end);
    }

    function statusText(cam) {
        const flips = [cam.flipH && 'H', cam.flipV && 'V'].filter(Boolean).join('+');
        return `Rotation ${cam.rotation}° · ${flips ? 'flip ' + flips : 'no flip'}`;
    }

    function buildPanel(cam) {
        const panel = document.createElement('div');
        panel.className = 'cam-panel';
        panel.innerHTML = `
            <div class="cam-panel-head">
                <span class="cam-panel-title">Camera ${cam.index + 1}</span>
                <div class="cam-panel-tools">
                    <button type="button" data-tool="rotate">⟳ 90°</button>
                    <button type="button" data-tool="flip-h">Flip H</button>
                    <button type="button" data-tool="flip-v">Flip V</button>
                </div>
            </div>`;
        const canvas = document.createElement('canvas');
        canvas.className = 'cam-canvas';
        panel.appendChild(canvas);
        const status = document.createElement('div');
        status.className = 'cam-panel-status';
        panel.appendChild(status);
        cam.canvas = canvas;

        sizeCanvas(cam);
        drawCam(cam);
        status.textContent = statusText(cam);
        attachDrag(cam);

        panel.querySelectorAll('button[data-tool]').forEach((btn) => {
            btn.addEventListener('click', () => {
                const tool = btn.dataset.tool;
                if (tool === 'rotate') { cam.rotation = (cam.rotation + 90) % 360; sizeCanvas(cam); }
                else if (tool === 'flip-h') cam.flipH = !cam.flipH;
                else if (tool === 'flip-v') cam.flipV = !cam.flipV;
                drawCam(cam);
                status.textContent = statusText(cam);
            });
        });
        return panel;
    }

    async function open() {
        backdrop.hidden = false;
        body.innerHTML = '<div class="cam-loading">Loading camera frames…</div>';
        let data;
        try {
            const res = await fetch(`/api/bin/${encodeURIComponent(BIN_ID)}/camera-config`, { cache: 'no-store' });
            if (res.status === 401) { window.location.href = '/login'; return; }
            data = await res.json();
        } catch (e) {
            body.innerHTML = '<div class="cam-loading">Could not load camera config.</div>';
            return;
        }
        const configs = data.cameras || [];
        cams = [];
        for (const c of configs) {
            const img = await loadSnapshot(c.cam_index);
            cams.push({
                index: c.cam_index,
                img,
                rotation: c.rotation || 0,
                flipH: !!c.flip_h,
                flipV: !!c.flip_v,
                crop: (c.crop && c.crop.length === 4) ? c.crop.slice() : [0, 0, 1, 1],
            });
        }
        body.innerHTML = '';
        if (!cams.length) { body.innerHTML = '<div class="cam-loading">No cameras found.</div>'; return; }
        cams.forEach((cam) => body.appendChild(buildPanel(cam)));
    }

    function close() { backdrop.hidden = true; cams = []; body.innerHTML = ''; }

    async function save() {
        const payload = {
            cameras: cams.map((c) => ({
                cam_index: c.index,
                rotation: c.rotation,
                flip_h: c.flipH,
                flip_v: c.flipV,
                crop: c.crop,
            })),
        };
        const btn = document.getElementById('cam-save');
        btn.disabled = true;
        try {
            const res = await fetch(`/api/bin/${encodeURIComponent(BIN_ID)}/camera-config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const d = await res.json().catch(() => ({}));
            if (!res.ok) { toast(d.error || 'Save failed', 'error'); return; }
            const where = d.applied === 'remote' ? 'sent to bin' : (d.applied === 'saved' ? 'saved (bin offline)' : 'applied');
            toast('Cameras ' + where, 'success');
            close();
        } catch (e) {
            toast('Network error: ' + e.message, 'error');
        } finally {
            btn.disabled = false;
        }
    }

    function resetAll() {
        cams.forEach((cam) => {
            cam.rotation = 0; cam.flipH = false; cam.flipV = false; cam.crop = [0, 0, 1, 1];
            sizeCanvas(cam); drawCam(cam);
            const status = cam.canvas.parentElement.querySelector('.cam-panel-status');
            if (status) status.textContent = statusText(cam);
        });
    }

    // Exposed for the inline onclick in index.html.
    window.openCameraEditor = open;

    document.getElementById('cam-editor-close').addEventListener('click', close);
    document.getElementById('cam-cancel').addEventListener('click', close);
    document.getElementById('cam-save').addEventListener('click', save);
    document.getElementById('cam-reset').addEventListener('click', resetAll);
    backdrop.addEventListener('click', (ev) => { if (ev.target === backdrop) close(); });
})();
