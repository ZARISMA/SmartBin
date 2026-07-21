/* HexaBin Settings — account password + user management.
 * Talks to /api/account/password and /api/users. Reuses the shared toast host
 * and confirm-modal scaffolding provided by _cc_base.html.
 */
(function () {
    'use strict';

    const toastHost = document.getElementById('toast-host');
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    const modalOk = document.getElementById('modal-ok');
    const modalCancel = document.getElementById('modal-cancel');

    const userList = document.getElementById('user-list');
    const currentUser = (document.getElementById('current-user') || {}).textContent || '';
    const passwordForm = document.getElementById('password-form');
    const addUserForm = document.getElementById('add-user-form');

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

    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    }

    async function loadUsers() {
        try {
            const res = await fetch('/api/users', { cache: 'no-store' });
            if (res.status === 401) { window.location.href = '/login'; return; }
            const data = await res.json();
            renderUsers(data.users || [], data.current || currentUser);
        } catch (e) {
            userList.innerHTML = '<li class="user-empty">Could not load users.</li>';
        }
    }

    function renderUsers(list, current) {
        if (!list.length) {
            userList.innerHTML = '<li class="user-empty">No users.</li>';
            return;
        }
        const single = list.length <= 1;
        userList.innerHTML = list.map((u) => {
            const isSelf = u.username === current;
            const created = u.created_at ? ` · <span class="user-created">${escapeHtml(u.created_at)}</span>` : '';
            // The last remaining account cannot be deleted (server enforces this too).
            const delBtn = single
                ? '<span class="user-note">last account</span>'
                : `<button class="sb-btn sb-btn-danger sb-btn-sm" data-del="${escapeHtml(u.username)}">Delete</button>`;
            return `<li class="user-row">
                <div class="user-meta">
                    <span class="user-name">${escapeHtml(u.username)}</span>
                    ${isSelf ? '<span class="user-you">you</span>' : ''}
                    <span class="user-sub">Operator${created}</span>
                </div>
                ${delBtn}
            </li>`;
        }).join('');

        userList.querySelectorAll('button[data-del]').forEach((btn) => {
            btn.addEventListener('click', () => onDelete(btn.dataset.del));
        });
    }

    async function onDelete(username) {
        const extra = username === currentUser
            ? ' This is the account you are signed in as.'
            : '';
        const ok = await confirmModal('Delete user?', `Remove “${username}”?` + extra);
        if (!ok) return;
        try {
            const res = await fetch(`/api/users/${encodeURIComponent(username)}`, { method: 'DELETE' });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) { toast(data.error || 'Delete failed', 'error'); return; }
            toast(`Removed ${username}`, 'success');
            loadUsers();
        } catch (e) {
            toast('Network error: ' + e.message, 'error');
        }
    }

    addUserForm.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        const username = document.getElementById('new_username').value.trim();
        const password = document.getElementById('add_password').value;
        try {
            const res = await fetch('/api/users', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) { toast(data.error || 'Could not add user', 'error'); return; }
            toast(`Added ${data.username || username}`, 'success');
            addUserForm.reset();
            loadUsers();
        } catch (e) {
            toast('Network error: ' + e.message, 'error');
        }
    });

    passwordForm.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        const current = document.getElementById('current_password').value;
        const next = document.getElementById('new_password').value;
        const confirm = document.getElementById('confirm_password').value;
        if (next !== confirm) { toast('New passwords do not match', 'error'); return; }
        try {
            const res = await fetch('/api/account/password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ current_password: current, new_password: next }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) { toast(data.error || 'Could not update password', 'error'); return; }
            toast('Password updated', 'success');
            passwordForm.reset();
        } catch (e) {
            toast('Network error: ' + e.message, 'error');
        }
    });

    loadUsers();
})();
