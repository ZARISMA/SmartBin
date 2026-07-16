/* HexaBin theme switcher.
 * <html data-theme> is set pre-paint by the inline bootstrap in each template's <head>;
 * this script owns the toggle buttons ([data-theme-toggle]), persistence, and change events.
 * Contract: DESIGN.md → Theming.
 */
(function () {
    'use strict';

    var KEY = 'hexabin-theme';
    var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

    function stored() {
        try {
            var t = localStorage.getItem(KEY);
            return t === 'light' || t === 'dark' ? t : null;
        } catch (e) { return null; }
    }

    function current() {
        return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
    }

    function apply(theme, persist) {
        document.documentElement.dataset.theme = theme;
        if (persist) {
            try { localStorage.setItem(KEY, theme); } catch (e) { /* private mode */ }
        }
        document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
            btn.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
        });
        window.dispatchEvent(new CustomEvent('hexabin:themechange', { detail: { theme: theme } }));
    }

    document.addEventListener('click', function (ev) {
        var btn = ev.target.closest('[data-theme-toggle]');
        if (btn) apply(current() === 'dark' ? 'light' : 'dark', true);
    });

    // Follow OS preference only while the user has made no explicit choice.
    if (media && media.addEventListener) {
        media.addEventListener('change', function (ev) {
            if (!stored()) apply(ev.matches ? 'dark' : 'light', false);
        });
    }

    // Sync aria-pressed on load (bootstrap already set data-theme).
    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
            btn.setAttribute('aria-pressed', current() === 'dark' ? 'true' : 'false');
        });
    });
})();
