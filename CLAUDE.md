# CLAUDE.md

Smart Waste AI — real-time waste classification using dual OAK-D USB3 depth cameras and Google Gemini vision API. Captures side-by-side frames, classifies into 7 categories, logs to PostgreSQL/SQLite and image files.

**Entry points:**
- `python main.py` / `smartwaste` — manual mode, dual OAK cameras, OpenCV window
- `python mainauto.py` / `smartwaste-auto` — auto gate mode with presence detection
- `python mainoak.py` / `smartwaste-oak` — OAK-D Native mode (depth + IMU + NN voting)
- `python mainraspberry.py` — Raspberry Pi dual cameras
- `python -m smartwaste.web` — Docker/web UI (FastAPI + MJPEG stream)

CLI entry points are defined in `pyproject.toml`.

## Quick Start

```bash
source oak_env/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY='your_key_here'
python main.py
```

**Runtime controls (local mode):** `c` — classify, `a` — toggle auto-classify, `q` — quit

## Docker Mode

```bash
docker-compose up -d        # starts app + PostgreSQL + Grafana
docker-compose logs -f app
```

Services: Web UI `:8000`, Grafana `:3000` (admin/admin), PostgreSQL `:5432`.
Requires Linux host with OAK-D cameras on USB.

## Presentation Website

Available at `/site` (e.g. `http://localhost:8000/site`). A single-page marketing/presentation site showcasing the SmartBin product.

**Sections:** Hero, About, Modules (6 waste categories), Live Statistics (from `/api/stats`), Deployment Map (Leaflet/OpenStreetMap, 8 Yerevan landmarks), Media/Video (placeholder for 3D renders and real-life footage), Contact/Footer.

**Files:**
- `smartwaste/web_templates/site.html` — Jinja2 template (single-page scrolling)
- `smartwaste/web_static/site.css` — Dedicated glassmorphism dark theme, responsive (768px/1024px breakpoints)
- `smartwaste/web_static/site.js` — Stats fetching, Leaflet map init, scroll animations, animated counters

**External CDN deps:** Leaflet 1.9.4 (map tiles via CARTO dark basemap), Google Fonts Inter.

**Contact placeholders:** Phone `+374 12 345 678`, email `info@smartbin.am` — update in `site.html` footer.

## Database Backend

Set via `SMARTWASTE_DB_BACKEND`:
- `sqlite` (default) — `waste_dataset/waste.db`, no setup needed
- `postgresql` — requires running PostgreSQL (Docker provides this)

Migration: `python scripts/migrate_json_to_pg.py --source sqlite|json`

## Key Configuration (`smartwaste/config.py` + `smartwaste/settings.py`)

| Constant | Default | Purpose |
|---|---|---|
| `VALID_CLASSES` | 7 categories | Plastic, Glass, Paper, Organic, Aluminum, Other, Empty |
| `AUTO_INTERVAL` | 6 s | Delay between auto-classifications |
| `CROP_PERCENT` | 0.20 | Crops 20% from each side before AI analysis |
| `MAX_DT` | 0.25 s | Max frame timestamp delta between cameras |
| `MODEL_NAME` | `gemini-3-flash-preview` | Gemini model |
| `JPEG_QUALITY` | 85 | Compression for API image uploads |
| `DB_BACKEND` | `sqlite` | Database backend (`sqlite` or `postgresql`) |
| `CAMERA_MODE` | `oak` | Camera backend for web UI (`oak`, `raspberry`, `oak-native`) |

All constants flow through `settings.py` (Pydantic BaseSettings) — override via env vars or `.env`.

## Architecture

**Threading model:** Main thread handles capture, UI, and keyboard input. Daemon worker threads handle Gemini API calls asynchronously. `AppState` (`state.py`) coordinates shared flags thread-safely.

**Data flow:**
1. Dual camera frames captured independently → synced by timestamp (`MAX_DT`)
2. Frames concatenated horizontally → cropped → JPEG-encoded
3. Bytes sent to Gemini with structured prompt → strict JSON response parsed
4. Result logged to console, `logs/run_*.log`, database, and `waste_dataset/*.jpg`

**AI prompt** (`smartwaste/prompt.py`): Returns `{"category": ..., "description": ..., "brand_product": ...}`. Includes Armenian brand examples (Jermuk, BOOM, etc.) for Yerevan deployment.

## Module Structure

```
main.py              ← manual OAK mode (OpenCV window)
mainauto.py          ← auto gate mode (presence-gated classifications)
mainoak.py           ← OAK-D Native mode (depth + IMU + NN sensor fusion)
mainraspberry.py     ← Raspberry Pi dual camera mode
smartwaste/
  app.py             ← shared OAK-camera run loop
  camera.py          ← OAK camera pipeline helper (single device)
  cameraOak.py       ← dual OAK pipeline setup and frame cropping
  cameraraspberry.py ← Raspberry Pi picamera2 setup
  classifier.py      ← Gemini API call, JSON parsing, retry logic
  config.py          ← all constants and paths
  database.py        ← dual SQLite/PostgreSQL persistence layer
  dataset.py         ← save images and database entries
  log_setup.py       ← logging configuration
  oak_native.py      ← OAK-D multi-sensor occupancy detector (depth + IMU + NN)
  presence.py        ← pixel-diff background model for bin-occupancy (no API calls)
  prompt.py          ← Gemini prompt string
  settings.py        ← Pydantic BaseSettings (layered config, env-var overrides)
  state.py           ← thread-safe AppState class
  strategies.py      ← classification-trigger strategies (Manual, PresenceGate)
  ui.py              ← OpenCV overlay rendering
  utils.py           ← shared frame helpers
  web.py             ← FastAPI web UI with MJPEG stream
  web_templates/     ← Jinja2 HTML templates
    index.html       ← operational dashboard (live stream, controls)
    site.html        ← presentation/marketing website
  web_static/        ← CSS/JS static files
    style.css        ← dashboard styles
    site.css         ← presentation site styles
    site.js          ← presentation site JS (stats, map, animations)
scripts/
  migrate_json_to_pg.py ← one-time data migration to PostgreSQL
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
python -m pytest tests/ --cov=smartwaste --cov-report=term-missing
```

Test files follow `tests/test_<module>.py`. Install deps: `pip install -r requirements-test.txt`.
