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

## Shared Component Classes

Provided by `brand.css` and available on every page — reuse these instead of re-implementing:

- **Status pill** — `.sb-pill` with state modifiers `online` / `degraded` / `offline` / `stopped` (dot + uppercase label).
- **Buttons** — `.sb-btn` with variants `.sb-btn-primary`, `.sb-btn-secondary`, `.sb-btn-outline`, `.sb-btn-ghost`, `.sb-btn-danger`.
- **Card** — `.sb-card` (white surface, `--sb-radius-lg`, hairline border).
- **Toasts** — `.toast-host` container + `.toast` (modifiers `success` / `error` / `leave`).
- **Modal** — `.modal-backdrop` + `.modal` + `.modal-actions` (confirm dialog scaffolding in `_cc_base.html`).
- **Animations** — `sbPulse`, `sbFadeIn` keyframes.

## CSS File Ownership

| File | Owns | Notes |
|------|------|-------|
| `hexabin/web_static/brand.css` | Tokens + shared components above | Loaded **first** by every template |
| `hexabin/web_static/dashboard.css` | Control Center layout: sidebar, cards grid, filters, page-specific styles | Admin pages only |
| `hexabin/web_static/site.css` | Presentation site: glassmorphism dark theme, responsive (768px / 1024px breakpoints) | Public `/site` only |
| `hexabin/web_static/style.css` | Legacy dashboard styles | **Back-compat only — do not extend** |

## Themes

- **Control Center (admin)** — light theme: `--sb-cream` page background, white `.sb-card` surfaces, `--sb-ink` text.
- **Presentation site (`/site`)** — dedicated glassmorphism **dark** theme in `site.css`; same brand palette and fonts.
