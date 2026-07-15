# CLAUDE.md

Smart Waste AI — real-time waste classification using dual OAK-D USB3 depth cameras and a pluggable LLM vision backend: Google Gemini (cloud), LM Studio (local, OpenAI-compatible API), or a **cascade** (local first, escalate to Gemini below a confidence threshold). Captures side-by-side frames, classifies into 7 categories, logs to PostgreSQL/SQLite and image files, and returns an "open module N" actuation command per classification.

**Entry points:**
- `python main.py` / `hexabin` — manual mode, dual OAK cameras, OpenCV window
- `python mainauto.py` / `hexabin-auto` — auto gate mode with presence detection
- `python mainoak.py` / `hexabin-oak` — OAK-D Native mode (depth + IMU + NN voting)
- `python -m hexabin.control` / `hexabin-run` — **unified edge runner** (used by Docker). Picks pipeline from `HEXABIN_CAMERA_MODE` and strategy from `HEXABIN_STRATEGY`. Supports hot strategy swaps and clean restarts triggered by the admin dashboard.
- `python -m hexabin.web` — Docker/web UI (FastAPI + MJPEG stream + fleet control)

CLI entry points are defined in `pyproject.toml`.

## Quick Start

```bash
source oak_env/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY='your_key_here'
python main.py
```

**Runtime controls (local mode):** `c` — classify, `a` — toggle auto-classify, `q` — quit

## Authentication

Dashboard and API require login. Default credentials: `admin` / `password123`.
Override via `HEXABIN_ADMIN_USERNAME` and `HEXABIN_ADMIN_PASSWORD` env vars.
The presentation site at `/site` is public (no login required).

Auth is **scoped**: the admin session/password opens everything; the edge API key
(`HEXABIN_EDGE_API_KEY` as a Bearer token) is valid ONLY for the ingest endpoints
`/api/report`, `/api/heartbeat`, and `/api/edge/classify` — it does not open admin routes.

## Deployment Roles

### 1. Server only (LAN dashboard, no cameras, no local LLM)
```bash
docker compose up -d
```
Runs FastAPI + PostgreSQL + Grafana. Edge devices POST frames to `/api/edge/classify`
(server-side classification) or finished results to `/api/report` (legacy local mode).
Classification backend defaults to Gemini; point `HEXABIN_LMSTUDIO_URL` at a remote
LLM host (role 3) and set `HEXABIN_LLM_BACKEND=lmstudio|cascade` to go local.

### 2. Server + LLM host (same machine)
```bash
# First: start LM Studio's server with the model loaded
#   lms server start        (or Developer → Start Server in the GUI)
#   curl http://localhost:1234/v1/models   → must list google/gemma-4-12b-qat
docker compose -f docker-compose.server-llm.yml up -d
```
Same server stack, pre-wired to LM Studio on the Docker host via
`host.docker.internal:1234`. Default backend is `cascade`: LM Studio classifies first;
confidence < 70% (or LM Studio failure) escalates the image to Gemini.

### 3. LLM host only
No HexaBin code runs here. Install LM Studio on any machine, load
`google/gemma-4-12b-qat`, start the server and enable **"Serve on Local Network"**.
Verify with `curl http://<ip>:1234/v1/models`, then point the HexaBin server at it via
`HEXABIN_LMSTUDIO_URL=http://<ip>:1234/v1`.
**LM Studio has no authentication — keep port 1234 LAN-only/firewalled, never internet-facing.**

### 4. Full stack on Raspberry Pi (cameras + dashboard)
```bash
docker compose -f docker-compose.edge-full.yml up -d
```
Runs everything on the Pi: cameras, web UI, database, Grafana.

### 5. Lightweight edge (cameras only, classifies via server)
```bash
HEXABIN_SERVER_URL=http://<server-ip>:8000 \
HEXABIN_BIN_ID=bin-01 \
docker compose -f docker-compose.edge.yml up -d
```
Runs the camera pipeline on the Pi, no web UI. Default `HEXABIN_CLASSIFY_MODE=server`:
each captured frame is POSTed to the server, which runs the LLM and replies with the
classification + actuation command in the same response. Set `HEXABIN_CLASSIFY_MODE=local`
for the original on-device Gemini behavior (needs `GEMINI_API_KEY` on the Pi).

Services: Web UI `:8000`, Grafana `:3000` (admin/admin), PostgreSQL `:5433` (server) / `:5432` (edge-full), LM Studio `:1234`.
Full-stack and edge modes require Linux host with cameras on USB.

## Presentation Website

Available at `/site` (e.g. `http://localhost:8000/site`). A single-page marketing/presentation site showcasing the HexaBin product.

**Sections:** Hero, About, Modules (6 waste categories), Live Statistics (from `/api/public/stats` — unauthenticated fleet-wide aggregate `{total, today, by_category, recyclable_share, bins: {online, total}, latest}`, polled every 5 s; the admin `/api/stats` stays auth-gated), Deployment Map (Leaflet/OpenStreetMap, 8 Yerevan landmarks), Media/Video (interactive 3D model; footage/demo marked "Coming soon"), Contact/Footer. All numbers on the site are real (DB + live heartbeats) — unmeasured metrics (accuracy benchmark, CO₂, mean latency) say "Coming soon" instead of placeholder values.

**Files:**
- `hexabin/web_templates/site.html` — Jinja2 template (single-page scrolling)
- `hexabin/web_static/site.css` — Dedicated glassmorphism dark theme, responsive (768px/1024px breakpoints)
- `hexabin/web_static/site.js` — Stats fetching, Leaflet map init, scroll animations, animated counters

**External CDN deps:** Leaflet 1.9.4 (map tiles via CARTO dark basemap), Google Fonts (Manrope + Chakra Petch + JetBrains Mono).

**Contact placeholders:** Phone `+374 12 345 678`, email `info@hexabin.am` — update in `site.html` footer.

## Control Center (Admin UI)

The authenticated admin UI is a multi-page **Control Center** built around a shared sidebar layout. All pages extend `hexabin/web_templates/_cc_base.html`, which provides:

- Left sidebar with HexaBin brand, nav (Devices / Map / Analytics / Alerts / Classifications), server-health footer, signed-in operator chip, and a "Presentation site" link.
- Top bar with kicker + page title, plus page-specific actions injected via `{% block topbar_actions %}`.
- Global toast host (`#toast-host`) and confirm-modal scaffolding (`#modal`) reused by all dashboard JS.

Routes (all require login; all redirect to `/login` otherwise):

| Path | Template | JS | Purpose |
|---|---|---|---|
| `/` | `dashboard.html` | `dashboard.js` | Devices (cards grid + filters) |
| `/map` | `dashboard_map.html` | `dashboard_map.js` | Deployment map with Leaflet, legend, basemap switch, detail panel |
| `/analytics` | `dashboard_analytics.html` | `dashboard_analytics.js` | KPI strip, charts, period switch, CSV export — all real data from `/api/analytics?period=24h\|7d\|30d\|90d\|ytd` |
| `/alerts` | `dashboard_alerts.html` | `dashboard_alerts.js` | Camera-availability alerts from live heartbeats (`/api/alerts`: `camera_count` 0 → `NO_CAMERA` error, 1 → `SINGLE_CAMERA` warning; stale bins excluded) |
| `/classifications` | `dashboard_classifications.html` | `dashboard_classifications.js` | Browse classification records: bin/category/search filters, pagination (`/api/entries` + `/api/entries/count`), thumbnails + lightbox via authed `/api/entries/{id}/image` |
| `/bin/{bin_id}` | `index.html` | — | Per-bin detail view (live stream, controls, stats) |
| `/site` | `site.html` | `site.js` | Public marketing/presentation site |
| `/login` | `login.html` | — | Authentication |

Each authenticated route passes `{ active, user }` into the base template so the sidebar highlights the current page and shows the logged-in operator.

### Devices page (`/`)

Each online bin is rendered as a card with a live thumbnail (MJPEG proxied from the edge), status pill (`online` / `degraded` / `offline` / `stopped`), structured warnings, and these controls:

- **Start / Stop** — pause or resume classifications (in-process; no restart).
- **Restart** — clean exit 0 so the container supervisor respawns with current env.
- **Strategy dropdown** (Manual / Auto Gate) — hot-swaps inside the running dual-OAK process.
- **Pipeline dropdown** (OAK Dual / OAK Native) — writes `HEXABIN_CAMERA_MODE` and triggers a restart.
- **Classify** — force a single classification on the current frame.
- **Auto toggle** — toggle the auto-classify flag.

Commands flow: browser → `POST /api/bin/{id}/command` (server, auth-gated, per-bin rate-limit) → edge sidecar `/command` (Bearer `EDGE_API_KEY`) → mutates `AppState`. Heartbeats carry the live `strategy`, `pipeline`, `camera_count`, `running`, `auto_classify`, and `warnings[]` so the dashboard reflects edge state within ~5 seconds.

Warnings are structured (`code`, `severity`, `message`) and deduped by code in `hexabin/warnings.py`. Known codes today: `CAMERA_COUNT_LOW`, `CAMERA_MISSING`, `SERVER_UNREACHABLE` (edge could not reach `/api/edge/classify` in server mode).

### Static assets

- `brand.css` — design tokens (CSS variables for brand palette, typography, spacing); loaded first by `_cc_base.html`.
- `dashboard.css` — Control Center layout, sidebar, cards, modals, toasts. (Supersedes the older `style.css`, which is kept for back-compat only.)
- `cc_nav.js` — shared sidebar helper loaded by `_cc_base.html` on every page; polls `/api/alerts` and keeps the nav alerts badge (`#cc-alerts-badge`) current.
- `sb-logo.svg` — HexaBin wordmark/logo.
- `models/hexabin.glb` — 3D bin model used by the presentation site / hero. `web.py` registers `.glb` → `model/gltf-binary` and `.gltf` → `model/gltf+json` MIME types at import time so `StaticFiles` serves them with correct `Content-Type`.

**External CDN deps (admin):** Manrope + Chakra Petch + JetBrains Mono (Google Fonts), Leaflet 1.9.4 (on `/map`).

## Database Backend

Set via `HEXABIN_DB_BACKEND`:
- `sqlite` (default) — `waste_dataset/waste.db`, no setup needed
- `postgresql` — requires running PostgreSQL (Docker provides this)

Migration: `python scripts/migrate_json_to_pg.py --source sqlite|json`

## Key Configuration (`hexabin/config.py` + `hexabin/settings.py`)

| Constant | Default | Purpose |
|---|---|---|
| `VALID_CLASSES` | 7 categories | Plastic, Glass, Paper, Organic, Aluminum, Other, Empty |
| `AUTO_INTERVAL` | 6 s | Delay between auto-classifications |
| `CROP_PERCENT` | 0.20 | Crops 20% from each side before AI analysis |
| `MAX_DT` | 0.25 s | Max frame timestamp delta between cameras |
| `MODEL_NAME` | `gemini-3-flash-preview` | Gemini model |
| `LLM_BACKEND` | `gemini` | Classification backend: `gemini`, `lmstudio`, or `cascade` |
| `LMSTUDIO_URL` | `http://localhost:1234/v1` | LM Studio OpenAI-compatible base URL (any OpenAI-compatible server works) |
| `LMSTUDIO_MODEL` | `google/gemma-4-12b-qat` | Local model id (must match `GET {url}/models`) |
| `LMSTUDIO_TIMEOUT` | `120` s | Per-call timeout for local inference |
| `LMSTUDIO_MAX_TOKENS` | `2000` | Completion budget — generous because reasoning models burn tokens before the JSON |
| `CONFIDENCE_THRESHOLD` | `0.70` | Cascade: escalate to Gemini below this confidence |
| `LLM_MAX_CONCURRENCY` | `2` | Concurrent LLM calls the server will run |
| `CLASSIFY_MODE` | `local` | Edge classification: `local` (on-device) or `server` (via `/api/edge/classify`) |
| `CLASSIFY_TIMEOUT` | `180` s | Edge → server classify HTTP timeout |
| `MODULE_MAP` | built-in | JSON category→module override, e.g. `{"Plastic": 1}` ("Empty" never opens) |
| `ACTUATOR` | `log` | Actuation driver: `log`, `none` (`gpio` reserved — see `hexabin/actuator.py`) |
| `MAX_UPLOAD_MB` | `10` | Max decoded image size accepted by ingest endpoints |
| `JPEG_QUALITY` | 85 | Compression for API image uploads |
| `DB_BACKEND` | `sqlite` | Database backend (`sqlite` or `postgresql`) |
| `CAMERA_MODE` | `oak` | Camera backend for web UI (`oak`, `raspberry`, `oak-native`, `none`) |
| `ADMIN_USERNAME` | `admin` | Dashboard login username |
| `ADMIN_PASSWORD` | `password123` | Dashboard login password |
| `BIN_ID` | `bin-01` | Unique identifier for this bin device |
| `EDGE_MODE` | `false` | Enable edge mode (heartbeats + sidecar + server reporting) |
| `SERVER_URL` | `` | Central server URL for edge reporting/classification |
| `EDGE_API_KEY` | `` | Shared secret for edge-to-server auth (ingest endpoints only) |
| `HEARTBEAT_INTERVAL` | `30` | Seconds between edge heartbeats |

All constants flow through `settings.py` (Pydantic BaseSettings) — override via env vars or `.env`.

## Architecture

**Threading model:** Main thread handles capture, UI, and keyboard input. Daemon worker threads handle LLM calls asynchronously. `AppState` (`state.py`) coordinates shared flags thread-safely; its `start_classify()` gate also enforces the dashboard Stop button.

**Data flow (local mode, `CLASSIFY_MODE=local`):**
1. Dual camera frames captured independently → synced by timestamp (`MAX_DT`)
2. Frames concatenated horizontally → cropped → JPEG-encoded
3. Bytes sent to the configured backend (`hexabin/llm.py`) → strict JSON response parsed
4. Result logged to console, `logs/run_*.log`, database, and `waste_dataset/*.jpg`; `actuator.dispatch()` opens the mapped module

**Data flow (server mode, `CLASSIFY_MODE=server` — the hub-and-spoke architecture):**
1. Sensors → End Device: cameras capture and JPEG-encode the frame
2. End Device → Server: one `POST /api/edge/classify` with `image_b64` + sensor data (Bearer `EDGE_API_KEY`)
3. Server → LLM: server forwards the image to the configured backend (LM Studio / Gemini / cascade), bounded by `LLM_MAX_CONCURRENCY`
4. LLM → Server: classification (+ self-reported confidence) parsed and normalized
5. Server → End Device: the same HTTP response carries `{result, command: {action: "open_module", module: N}}`; the edge applies it to `AppState` and calls `actuator.dispatch()`
6. In parallel the server persists the row (+ `confidence`, `llm_backend`) and image, and the dashboard picks it up on its 5-second poll

**Cascade semantics** (`CascadeBackend` in `hexabin/llm.py`): LM Studio first; escalate to Gemini when it errors, reports no confidence, or confidence < `CONFIDENCE_THRESHOLD`. If Gemini then fails but LM Studio produced a parseable low-confidence result, that result is used (degraded beats erroring).

**AI prompt** (`hexabin/prompt.py`): Returns `{"category": ..., "description": ..., "brand_product": ..., "confidence": 0-100}`. Includes Armenian brand examples (Jermuk, BOOM, etc.) for Yerevan deployment.

## Module Structure

```
main.py              ← manual OAK mode (OpenCV window)
mainauto.py          ← auto gate mode (presence-gated classifications)
mainoak.py           ← OAK-D Native mode (depth + IMU + NN sensor fusion)
hexabin/
  actuator.py        ← pluggable bin-module actuation (log now, GPIO drops in later)
  analytics.py       ← period windows + /api/analytics payload assembly (pure, unit-tested)
  app.py             ← shared OAK-camera run loop
  camera.py          ← OAK camera pipeline helper (single device)
  cameraOak.py       ← dual OAK pipeline setup and frame cropping
  cameraraspberry.py ← legacy Raspberry Pi picamera2 setup (kept for web.py)
  control.py         ← unified edge runner (docker entry point)
  warnings.py        ← structured runtime warnings surfaced on the dashboard
  classifier.py      ← classification worker: dispatches local (in-process LLM) vs server mode
  config.py          ← all constants and paths (incl. MODULE_MAP parsing)
  database.py        ← dual SQLite/PostgreSQL persistence layer (bin_id, confidence, llm_backend)
  dataset.py         ← save images, database entries, and edge reporting
  edge_client.py     ← HTTP client for edge→server reporting, heartbeats, classify_remote
  llm.py             ← LLM backends: Gemini (retry + circuit breaker), LM Studio, cascade
  log_setup.py       ← logging configuration
  oak_native.py      ← OAK-D multi-sensor occupancy detector (depth + IMU + NN)
  presence.py        ← pixel-diff background model for bin-occupancy (no API calls)
  prompt.py          ← LLM prompt string (strict JSON incl. confidence)
  schemas.py         ← Pydantic models for edge communication (EdgeReport, EdgeClassifyRequest/Response, BinHeartbeat)
  settings.py        ← Pydantic BaseSettings (layered config, env-var overrides)
  state.py           ← thread-safe AppState class
  strategies.py      ← classification-trigger strategies (Manual, PresenceGate)
  ui.py              ← OpenCV overlay rendering
  utils.py           ← shared frame helpers
  web.py             ← FastAPI web UI with auth, multi-bin dashboard, edge endpoints
  web_templates/         ← Jinja2 HTML templates
    _cc_base.html        ← Control Center shared layout (sidebar + topbar + modal/toast hosts)
    dashboard.html       ← Devices page (`/`) — bin cards, filters, stat strip
    dashboard_map.html   ← Map page    (`/map`) — Leaflet, legend, detail panel
    dashboard_analytics.html ← Analytics page (`/analytics`) — KPIs, charts, export
    dashboard_alerts.html    ← Alerts page (`/alerts`) — camera-availability alerts
    dashboard_classifications.html ← Classifications page (`/classifications`) — record browser
    index.html           ← per-bin detail view (`/bin/{id}`)
    login.html           ← authentication page
    site.html            ← presentation/marketing website (public)
  web_static/            ← CSS/JS static files
    brand.css            ← brand design tokens (CSS variables)
    dashboard.css        ← Control Center styles (sidebar, cards, modals)
    dashboard.js         ← Devices page JS (polling, filters, commands)
    dashboard_map.js     ← Map page JS (Leaflet init, markers, basemap switch)
    dashboard_analytics.js ← Analytics page JS (charts, period switch, CSV)
    dashboard_alerts.js    ← Alerts page JS (5s polling, severity rows, badge)
    dashboard_classifications.js ← Classifications page JS (filters, pagination, lightbox)
    cc_nav.js            ← shared sidebar alerts badge (all Control Center pages)
    site.css / site.js   ← presentation site assets
    sb-logo.svg          ← brand logo
    models/hexabin.glb  ← 3D bin model (served via StaticFiles)
    style.css            ← legacy dashboard styles (kept for back-compat)
scripts/
  migrate_json_to_pg.py  ← one-time data migration to PostgreSQL
  build_pitch_hti.py     ← generates HexaBin_HTI_Pitch.pptx (Armenian grant pitch deck)
grafana/             ← Grafana provisioning and dashboards
```

## Brand Color Palette

All UI (web dashboard, overlays, Grafana) must use these project colors.

**Primary**
- Forest Green: `#2D5A42`
- Deep Smart Blue: `#1A4D6B`

**Secondary**
- Stone Gray: `#8C8C8C`
- Taupe: `#BDB76B`

**Modular System (waste category colors)**
| Category | Color | Hex |
|----------|-------|-----|
| Paper | Warm Cellulose | `#D2B48C` |
| Aluminum | Brushed Metallic Gray | `#A9A9A9` |
| Organic | Deep Biophilic Green | `#1E4D2B` |
| Glass | Translucent Aqua / Crystal Teal | `#40E0D0` |
| Plastic | Refined Synthetic Tone | `#87CEEB` |
| Other | Dynamic Module Gradient | `#9370DB` → `#1E90FF` |
| Empty | Stone Gray | `#8C8C8C` |

**Semantic**
- Success: `#4CAF50`
- Warning: `#FF9800`
- Error: `#C62828`
- Info: `#2196F3`

**Neutral**
- Pure White: `#FFFFFF`
- Off-White: `#F5F5F7`
- Light Gray: `#E0E0E0`
- Dark Charcoal: `#333333`

## Testing

```bash
python -m pytest tests/
python -m pytest tests/ --cov=hexabin --cov-report=term-missing
```

Test files follow `tests/test_<module>.py`. Install deps: `pip install -r requirements-test.txt`.
## Rules

- **No hardcoding**: no URLs/hosts/ports/keys/secrets in source — use env vars; defaults belong in `.env`.
- **Security**: fail fast if `JWT_SECRET`/`DATABASE_URL` missing (no fallbacks); authenticate + authorize every endpoint; validate/whitelist all input (never raw `req.body` to DB); uploads MIME+extension checked & authed; rate-limit auth endpoints; never self-assign elevated roles at register; Socket.IO authed via JWT handshake; CORS restricted to `CORS_ORIGIN`. Never leak internal errors to clients in prod.
- **Authorization**: permission-based only (see RBAC). Client prefers `useAuth().hasPermission('slug')` over role checks.
- **Audit**: every state-changing admin endpoint calls `AuditService`.
- **Full-stack consistency**: a change affecting both sides ships both in the same PR; server response-shape changes update the matching client interface.
- **Logging**: **Pino only, never `console.*`** — `createLogger('Module')`; `warn`/`error`/`fatal` persist to DB + admin panel. Pass errors as `logger.error({ err }, 'msg')`.
- **Tests** (Vitest): server in `server/tests/` (mock at the service boundary), client in `client/src/tests/` (mock `api/`, not axios) — both mirror source layout. New service/controller/hook ships ≥1 happy + 1 error test. Verify changes actually run before declaring done.
- **Modularity**: one concern per file (separate display / logic / data fetching); >~300 lines is a signal to split — flag pre-existing oversized files, don't silently rewrite.
- **Naming**: components `PascalCase.tsx`, hooks `useCamelCase.ts`, other TS `camelCase.ts`, server classes/services `PascalCase.js` else `camelCase.js`, migrations `YYYYMMDDHHMMSS-kebab.js`, tests mirror source + `.test`.
- **Git**: Conventional Commits (`feat(scope):`, `fix(scope):`, …); descriptive branches; don't force-push `main`; never commit `.env`/`node_modules`/seed binaries.