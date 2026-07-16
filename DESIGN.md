# DESIGN.md — HexaBin Design System Contract

This file is the **single source of truth** for all HexaBin UI: the Control Center (admin dashboard), the presentation site, OpenCV overlays, and Grafana dashboards.

**The contract:**
- All UI must use the colors, fonts, tokens, and component classes defined here — never raw hex values or one-off styles.
- In CSS, reference tokens via `var(--sb-*)`; the tokens are defined in `hexabin/web_static/brand.css` and mirror this document.
- Reuse the shared component classes below before writing any new CSS. If a new pattern is genuinely needed, add it to `brand.css` (shared) or the owning page stylesheet, using existing tokens — and document it here.

## Brand Color Palette

All UI (web dashboard, overlays, Grafana) must use these project colors.

**Primary**
| Name | Hex | Token |
|------|-----|-------|
| Forest Green | `#2D5A42` | `--sb-forest` |
| Deep Smart Blue | `#1A4D6B` | `--sb-blue` |

**Secondary**
| Name | Hex | Token |
|------|-----|-------|
| Stone Gray | `#8C8C8C` | `--sb-stone` |
| Taupe | `#BDB76B` | `--sb-taupe` |

**Modular System (waste category colors)**
| Category | Color | Hex | Token |
|----------|-------|-----|-------|
| Paper | Warm Cellulose | `#D2B48C` | `--sb-paper` |
| Aluminum | Brushed Metallic Gray | `#A9A9A9` | `--sb-aluminum` |
| Organic | Deep Biophilic Green | `#1E4D2B` | `--sb-organic` |
| Glass | Translucent Aqua / Crystal Teal | `#40E0D0` | `--sb-glass` |
| Plastic | Refined Synthetic Tone | `#87CEEB` | `--sb-plastic` |
| Other | Dynamic Module Gradient | `#9370DB` → `#1E90FF` | `--sb-other-1` → `--sb-other-2` |
| Empty | Stone Gray | `#8C8C8C` | `--sb-empty` |

**Semantic**
| Name | Hex | Token |
|------|-----|-------|
| Success | `#4CAF50` | `--sb-success` |
| Warning | `#FF9800` | `--sb-warning` |
| Error | `#C62828` | `--sb-error` |
| Info | `#2196F3` | `--sb-info` |

**Neutral**
| Name | Hex | Token |
|------|-----|-------|
| Pure White | `#FFFFFF` | `--sb-white` |
| Off-White | `#F5F5F7` | `--sb-cream` |
| Light Gray | `#E0E0E0` | `--sb-light` |
| Dark Charcoal | `#333333` | `--sb-charcoal` |
| Ink (body text) | `#1d2722` | `--sb-ink` |

## Typography

Loaded via Google Fonts CDN (see template `<head>`s).

| Role | Font | Token |
|------|------|-------|
| Body / UI | Manrope | `--sb-font-ui` |
| Display / headings | Chakra Petch | `--sb-font-display` |
| Code / metrics / mono | JetBrains Mono | `--sb-font-mono` |

Each token carries a system-font fallback stack — always use the token, never a bare font name.

## Design Tokens

Defined once in `hexabin/web_static/brand.css` on `:root`. Reference via `var(--sb-*)`; never hardcode the underlying values.

- **Colors** — all palette entries above (`--sb-forest`, `--sb-paper`, `--sb-success`, …).
- **Fonts** — `--sb-font-ui`, `--sb-font-display`, `--sb-font-mono`.
- **Radii** — `--sb-radius-sm` (6px), `--sb-radius-md` (10px), `--sb-radius-lg` (14px), `--sb-radius-xl` (16px).
- **Shadows** — `--sb-shadow-card` (subtle card lift), `--sb-shadow-float` (modals/toasts).
- **Semantic (theme-mapped)** — see Theming below. **Rule: surfaces, text, and borders must use the semantic tokens** (`--sb-surface`, `--sb-text`, `--sb-border`, …); the raw palette tokens are for brand hues only (logos, category colors, buttons).

## Theming (light / dark)

Both themes ship on every page. The theme is the `data-theme="light" | "dark"` attribute on `<html>`:

- A tiny **inline bootstrap script** in each template `<head>` (before the stylesheets) sets `data-theme` pre-paint from `localStorage['hexabin-theme']`, falling back to `prefers-color-scheme` — no flash of the wrong theme.
- `hexabin/web_static/theme.js` owns the **`.sb-theme-toggle`** buttons (`[data-theme-toggle]`), persists explicit choices to `localStorage['hexabin-theme']`, follows OS changes only while no explicit choice exists, and dispatches a **`hexabin:themechange`** `CustomEvent` on `window`. JS that paints colors (SVG charts, Leaflet tiles, category swatches) must read tokens via `getComputedStyle` at render time and re-render on that event.

Semantic tokens and their values:

| Token | Light | Dark |
|---|---|---|
| `--sb-bg` (page) | `#F5F5F7` | `#0F1613` |
| `--sb-surface` (cards/sidebar/nav) | `#FFFFFF` | `#161E1A` |
| `--sb-surface-2` (nested/inputs) | `#F5F5F7` | `#1D2722` |
| `--sb-surface-2-hover` | `#ebe9e3` | `#243029` |
| `--sb-text` | `#1d2722` | `#E8EDEA` |
| `--sb-text-muted` | `#8C8C8C` | `#9AA69F` |
| `--sb-border` (hairlines) | `rgba(29,39,34,0.06)` | `rgba(255,255,255,0.08)` |
| `--sb-border-strong` | `rgba(29,39,34,0.10)` | `rgba(255,255,255,0.14)` |
| `--sb-hover` (row/nav hover tint) | `rgba(45,90,66,0.06)` | `rgba(255,255,255,0.06)` |
| `--sb-glass` (sticky nav) | `rgba(245,245,247,0.85)` | `rgba(15,22,19,0.85)` |
| `--sb-accent-ink` (forest as text/icon) | `#2D5A42` | `#6FAE8C` |
| `--sb-shadow-card` / `--sb-shadow-float` | ink-tinted, soft | black-based, stronger |
| `color-scheme` | `light` | `dark` |

Dark-only remaps for legibility on dark surfaces: `--sb-success #66BB6A`, `--sb-warning #FFA726`, `--sb-error #EF5350`, `--sb-info #42A5F5`, `--sb-organic #2E6B45`. All other brand hues are identical in both themes.

**Exception pattern**: elements with a *solid category-color background* (cat chips, bin-mock slats, chart bars) keep literal `var(--sb-ink)` dark text in both themes — a pastel chip always needs dark text. Intrinsically dark surfaces (camera stream `#1a1a1a`, map stage `#1e2226`, `.section-dark`, footer) are theme-independent.

## Shared Component Classes

Provided by `brand.css` and available on every page — reuse these instead of re-implementing:

- **Status pill** — `.sb-pill` with state modifiers `online` / `degraded` / `offline` / `stopped` (dot + uppercase label).
- **Buttons** — `.sb-btn` with variants `.sb-btn-primary`, `.sb-btn-secondary`, `.sb-btn-outline`, `.sb-btn-ghost`, `.sb-btn-danger`.
- **Card** — `.sb-card` (white surface, `--sb-radius-lg`, hairline border).
- **Toasts** — `.toast-host` container + `.toast` (modifiers `success` / `error` / `leave`).
- **Modal** — `.modal-backdrop` + `.modal` + `.modal-actions` (confirm dialog scaffolding in `_cc_base.html`).
- **Theme toggle** — `.sb-theme-toggle` icon button with `data-theme-toggle` attribute + `.sb-icon-sun`/`.sb-icon-moon` inline SVGs (wired by `theme.js`; shows the moon in light mode, the sun in dark).
- **Animations** — `sbPulse`, `sbFadeIn` keyframes.

## CSS File Ownership

| File | Owns | Notes |
|------|------|-------|
| `hexabin/web_static/brand.css` | Tokens + shared components above | Loaded **first** by every template |
| `hexabin/web_static/dashboard.css` | Control Center layout: sidebar, cards grid, filters, page-specific styles | Admin pages only |
| `hexabin/web_static/site.css` | Presentation site: editorial biophilic design, responsive (700px / 1024px breakpoints) | Public `/site` only |
| `hexabin/web_static/style.css` | Legacy dashboard styles | **Back-compat only — do not extend** |

## Surfaces

- **Control Center (admin)** — app chrome: sidebar + cards on `--sb-surface`, page on `--sb-bg`.
- **Presentation site (`/site`)** — editorial biophilic design in `site.css`; same brand palette and fonts, with permanently dark statistics/footer sections (`.section-dark`).
- Both surfaces support **light and dark** themes via the semantic tokens (see Theming above).
