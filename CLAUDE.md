# CLAUDE.md

Smart Waste AI — real-time waste classification using dual OAK-D USB3 depth cameras and Google Gemini vision API. Captures side-by-side frames, classifies into 7 categories, logs to PostgreSQL/SQLite and image files.

**Entry points:**
- `python main.py` / `smartwaste` — manual mode, dual OAK cameras, OpenCV window
- `python mainauto.py` / `smartwaste-auto` — auto gate mode with presence detection
- `python mainoak.py` / `smartwaste-oak` — OAK-D Native mode (depth + IMU + NN voting)
- `python mainraspberry.py` — Raspberry Pi dual cameras
- `python -m smartwaste.web` — Docker/web UI (FastAPI + MJPEG stream)

CLI entry points are defined in `pyproject.toml`.

<<<<<<< HEAD
## Quick Start
=======
**Primary entry points:** `main.py` (local OAK dual), `mainoak.py` (OAK-D native sensors), `mainraspberry.py` (Pi cameras), `python -m smartwaste.web` (Docker/web UI)

## Environment Setup

**Python requirement: 3.10+** (3.11 recommended; ships by default on Raspberry Pi OS Bookworm)
>>>>>>> 34aad8e095f07c3e89ce542e0a6bda57512e48f1

```bash
source oak_env/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY='your_key_here'
python main.py
```

**Runtime controls (local mode):** `c` — classify, `a` — toggle auto-classify, `q` — quit

<<<<<<< HEAD
## Docker Mode

```bash
docker-compose up -d        # starts app + PostgreSQL + Grafana
docker-compose logs -f app
```

Services: Web UI `:8000`, Grafana `:3000` (admin/admin), PostgreSQL `:5432`.
Requires Linux host with OAK-D cameras on USB.

## Database Backend
=======
## Running the Application

### Entry Points (Local Mode — OpenCV Window)

All local entry points render live camera frames in an OpenCV window and require a connected display (or `export DISPLAY=:0` over SSH).

---

#### `main.py` — Manual Mode (Dual OAK Cameras)

The default entry point. Captures frames from **two OAK-D USB3 cameras** side by side. Classification is triggered manually or on a timer.

```bash
python main.py
python main.py --model gemini-2.0-flash --auto-interval 10 --location Yerevan
```

**How it works:**
1. Opens two OAK-D cameras via `depthai`, each running an independent pipeline (`cameraOak.make_pipeline`)
2. Each frame is cropped (`CROP_PERCENT`) and resized to `DISPLAY_SIZE` (800x800)
3. Frames from both cameras are synced by timestamp (must be within `MAX_DT` = 0.25s) and concatenated horizontally into a 1600x800 combined frame
4. The combined frame is displayed with an overlay showing classification status and history
5. Classification is sent to the Gemini API in a background daemon thread

**Uses:** `smartwaste/app.py` run loop + `smartwaste/strategies.ManualStrategy`

**Keyboard controls:**
- `c` — classify current combined frame (one-shot)
- `a` — toggle auto-classify (fires every `AUTO_INTERVAL` seconds, default 6)
- `q` — quit

**CLI options:**

| Flag | Env var | Purpose |
|---|---|---|
| `--model NAME` | `SMARTWASTE_MODEL_NAME` | Gemini model to use |
| `--auto-interval SEC` | `SMARTWASTE_AUTO_INTERVAL` | Seconds between auto-classifications |
| `--location NAME` | `SMARTWASTE_LOCATION` | Location tag written to dataset |

**Requires:** 2x OAK-D USB3 cameras

---

#### `mainauto.py` — Automatic Gate Mode (Dual OAK Cameras)

Same dual-camera setup as `main.py`, but uses **local pixel-diff presence detection** to automatically gate Gemini API calls. No manual trigger needed — it detects when an item enters the bin and classifies it once.

```bash
python mainauto.py
python mainauto.py --threshold 15.0 --detect-n 5 --empty-n 8
```

**How it works:**
1. Same dual OAK camera capture and frame sync as `main.py`
2. Uses `smartwaste/strategies.PresenceGateStrategy` with a `PresenceDetector` (pixel-diff background subtraction)
3. State machine: `Calibrating` (learning background) → `Ready/IDLE` → `Detected` → `Classified` → `Ready/IDLE`
4. During calibration, collects `BG_WARMUP_FRAMES` (default 40) frames to build a background model
5. Each `CHECK_INTERVAL` (default 0.5s), computes a motion score against the background
6. When motion exceeds `MOTION_THRESHOLD` for `DETECT_CONFIRM_N` (default 3) consecutive checks, fires one Gemini API call
7. After classification, waits for `EMPTY_CONFIRM_N` (default 6) consecutive empty checks before resetting

**Uses:** `smartwaste/app.py` run loop + `smartwaste/strategies.PresenceGateStrategy`

**Keyboard controls:**
- `c` — force-classify current frame (manual override)
- `r` — reset background model from current frame
- `q` — quit

**CLI options:**

| Flag | Env var | Purpose |
|---|---|---|
| `--model NAME` | `SMARTWASTE_MODEL_NAME` | Gemini model to use |
| `--threshold FLOAT` | `SMARTWASTE_MOTION_THRESHOLD` | Pixel-diff threshold (0–255) |
| `--detect-n N` | `SMARTWASTE_DETECT_CONFIRM_N` | Consecutive detections to confirm |
| `--empty-n N` | `SMARTWASTE_EMPTY_CONFIRM_N` | Consecutive empties to clear |
| `--location NAME` | `SMARTWASTE_LOCATION` | Location tag written to dataset |

**Requires:** 2x OAK-D USB3 cameras

---

#### `mainoak.py` — OAK-D Native Mode (Single Camera, Sensor Fusion)

Uses a **single OAK-D camera** with three hardware sensors running on the Myriad X VPU for occupancy detection. No pixel-diff needed — classification is triggered by hardware sensor consensus.

```bash
python mainoak.py
python mainoak.py --depth-threshold 200 --imu-threshold 1.5 --votes 2
```

**How it works:**
1. Opens one OAK-D device and builds an extended pipeline (`smartwaste/oak_native.py`) with:
   - **Stereo depth** (CAM_B + CAM_C → StereoDepth) — measures distance change in a centre ROI; lighting-independent
   - **IMU accelerometer** — detects the physical shock when an item is dropped into the bin
   - **MobileNetSSD neural network** — runs object detection on the Myriad X VPU (on-device, no internet needed)
2. Each sensor casts a boolean vote: `depth_occupied`, `drop_flag`, `nn_occupied`
3. When `OAK_VOTES_NEEDED` (default 2) or more sensors agree, occupancy is confirmed
4. State machine: `Calibrating` → `Ready` → `Detected` → `Classifying` → `Classified` → `Ready`
   - **Calibrating:** collects `OAK_CALIB_FRAMES` (default 30) depth samples and `IMU_BASELINE_SAMPLES` (default 50) IMU samples to establish baselines
   - **Ready:** polls sensors every `OAK_CHECK_INTERVAL` (default 0.4s); needs `OAK_DETECT_CONFIRM_N` (default 3) consecutive positive votes
   - **Detected:** sends RGB frame to Gemini
   - **Classified:** waits for `OAK_EMPTY_CONFIRM_N` (default 5) consecutive zero-vote checks before returning to Ready
5. Display shows a 3-line status overlay: state, vote breakdown (Depth/IMU/NN), and raw sensor readings

**Sensor graceful degradation:** If IMU or NN is unavailable (hardware not present, blob download fails), the system continues with fewer sensors. Depth is the only required sensor.

**Keyboard controls:**
- `c` — force-classify current frame (bypasses sensor votes)
- `r` — reset and recalibrate all sensors
- `q` — quit

**CLI options:**

| Flag | Env var | Purpose |
|---|---|---|
| `--model NAME` | `SMARTWASTE_MODEL_NAME` | Gemini model to use |
| `--depth-threshold MM` | `SMARTWASTE_DEPTH_CHANGE_THRESHOLD` | Depth change (mm) to declare occupied |
| `--imu-threshold FLOAT` | `SMARTWASTE_IMU_SHOCK_THRESHOLD` | Acceleration delta (m/s²) to flag a drop |
| `--votes N` | `SMARTWASTE_OAK_VOTES_NEEDED` | Sensor votes needed to trigger classify |
| `--location NAME` | `SMARTWASTE_LOCATION` | Location tag written to dataset |

**Requires:** 1x OAK-D USB3 camera (with stereo pair for depth)

---

#### `mainraspberry.py` — Raspberry Pi Camera Mode

Captures frames from **two Raspberry Pi cameras** using `picamera2`. Same manual/auto classification flow as `main.py` but with Pi camera hardware.

```bash
python mainraspberry.py
```

**How it works:**
1. Discovers Pi cameras via `Picamera2.global_camera_info()`, requires at least 2
2. Creates two `Picamera2` instances configured for 1280x720 BGR888 preview
3. Each frame is captured via `grab_frame()`, cropped and resized identically to `main.py`
4. Frames are concatenated horizontally and displayed with overlay
5. Supports both manual (`c` key) and auto-classify (`a` key) modes

**Keyboard controls:**
- `c` — classify current combined frame (one-shot)
- `a` — toggle auto-classify (fires every `AUTO_INTERVAL` seconds)
- `q` — quit

**No CLI options** — configure via env vars or `.env` file.

**Requires:** 2x Raspberry Pi cameras (Pi 5 dual native ports, or camera multiplexer with `dtoverlay=camera-mux-4port`)

---

#### `python -m smartwaste.web` — Web UI (Docker / Headless)

The web entry point for running inside Docker or on headless machines. Replaces `cv2.imshow` with a FastAPI web server serving an MJPEG stream.

```bash
python -m smartwaste.web
```

**How it works:**
1. On startup, launches a background camera thread based on `SMARTWASTE_CAMERA_MODE`:
   - `oak` — dual OAK cameras (same logic as `main.py`)
   - `raspberry` — dual Pi cameras (same logic as `mainraspberry.py`)
   - `oak-native` — single OAK-D with sensor fusion (same logic as `mainoak.py`)
2. Camera thread writes frames to a shared buffer; the MJPEG endpoint reads from it at ~30 FPS
3. Classification is triggered via HTTP API (`POST /api/classify`) or auto-classify (`POST /api/toggle-auto`)
4. Serves a Jinja2 HTML dashboard at `/` with live video, controls, and classification history

**Web endpoints:**
- `GET /` — HTML dashboard
- `GET /stream` — MJPEG live video stream
- `POST /api/classify` — trigger one classification
- `POST /api/toggle-auto` — toggle auto-classify on/off
- `GET /api/state` — current label, detail, auto status, history
- `GET /api/entries` — database classification records
- `GET /api/stats` — total count and per-category breakdown

**Requires:** Same camera hardware as the selected mode. No display needed.

### Docker mode (Web UI + PostgreSQL + Grafana)

The web UI supports all three camera backends. Set `SMARTWASTE_CAMERA_MODE` to select which one:

```bash
# Dual OAK cameras (default)
SMARTWASTE_CAMERA_MODE=oak docker-compose up -d

# Raspberry Pi cameras
SMARTWASTE_CAMERA_MODE=raspberry docker-compose up -d

# OAK-D native (single camera, sensor fusion)
SMARTWASTE_CAMERA_MODE=oak-native docker-compose up -d
```

Or set it in `.env` and just run `docker-compose up -d`.

```bash
# View logs
docker-compose logs -f app
```

**Services:**
- **App (Web UI):** http://localhost:8000 — live camera feed, classification controls, history
- **Grafana:** http://localhost:3000 — metrics dashboard (admin/admin)
- **PostgreSQL:** localhost:5432 — waste classification database (not HTTP — use `docker-compose exec postgres psql -U smartwaste` to query)

**Requirements:** Linux host with cameras connected. OAK-D modes need USB3; Raspberry Pi mode needs picamera2.
>>>>>>> 34aad8e095f07c3e89ce542e0a6bda57512e48f1

Set via `SMARTWASTE_DB_BACKEND`:
- `sqlite` (default) — `waste_dataset/waste.db`, no setup needed
- `postgresql` — requires running PostgreSQL (Docker provides this)

Migration: `python scripts/migrate_json_to_pg.py --source sqlite|json`

## Key Configuration (`smartwaste/config.py` + `smartwaste/settings.py`)

<<<<<<< HEAD
| Constant | Default | Purpose |
=======
```bash
# Migrate from SQLite to PostgreSQL (preferred — has sensor data)
python scripts/migrate_json_to_pg.py --source sqlite

# Migrate from metadata.json (sensor fields filled with 0.0)
python scripts/migrate_json_to_pg.py --source json
```

## Hardware Requirements

- 2x OAK-D USB3 cameras (uses `depthai` SDK) — for `main.py` / `mainauto.py` / Docker `oak` mode
- **OR** 1x OAK-D USB3 camera — for `mainoak.py` / Docker `oak-native` mode (depth + IMU + NN)
- **OR** 2x Raspberry Pi cameras (uses `picamera2`) — for `mainraspberry.py` / Docker `raspberry` mode
- Internet access for Gemini API calls

### Raspberry Pi camera setup

```bash
# Install picamera2 (already available on Pi OS Bookworm)
sudo apt install -y python3-picamera2

# Or inside a venv:
pip install picamera2

# Enable cameras in /boot/firmware/config.txt:
# Pi 5 dual native ports — default (camera_auto_detect=1 is already on)
# Camera multiplexer:
#   dtoverlay=camera-mux-4port
```

## Key Configuration Constants (in `smartwaste/config.py`)

| Constant | Value | Purpose |
>>>>>>> 34aad8e095f07c3e89ce542e0a6bda57512e48f1
|---|---|---|
| `VALID_CLASSES` | 7 categories | Plastic, Glass, Paper, Organic, Aluminum, Other, Empty |
| `AUTO_INTERVAL` | 6 s | Delay between auto-classifications |
| `CROP_PERCENT` | 0.20 | Crops 20% from each side before AI analysis |
| `MAX_DT` | 0.25 s | Max frame timestamp delta between cameras |
| `MODEL_NAME` | `gemini-3-flash-preview` | Gemini model |
| `JPEG_QUALITY` | 85 | Compression for API image uploads |
<<<<<<< HEAD
| `DB_BACKEND` | `sqlite` | Database backend |

All constants flow through `settings.py` (Pydantic BaseSettings) — override via env vars or `.env`.
=======
| `DB_BACKEND` | `sqlite` | Database backend (`sqlite` or `postgresql`) |
| `CAMERA_MODE` | `oak` | Camera backend for web UI (`oak`, `raspberry`, `oak-native`) |
>>>>>>> 34aad8e095f07c3e89ce542e0a6bda57512e48f1

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
<<<<<<< HEAD
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
  web_static/        ← CSS/JS static files
=======
main.py                  ← entry point (dual OAK cameras, manual/auto)
mainauto.py              ← entry point (dual OAK cameras, auto gate mode)
mainoak.py               ← entry point (single OAK-D, depth/IMU/NN sensor fusion)
mainraspberry.py         ← entry point (Raspberry Pi cameras)
smartwaste/
  __init__.py
  __main__.py            ← entry point for web UI (python -m smartwaste.web)
  config.py              ← all constants and paths
  cameraOak.py           ← OAK pipeline setup and frame cropping
  cameraraspberry.py     ← Raspberry Pi picamera2 setup and frame capture
  classifier.py          ← Gemini API call, JSON parsing
  database.py            ← dual SQLite/PostgreSQL persistence layer
  dataset.py             ← save images and database entries
  log_setup.py           ← logging configuration
  oak_native.py          ← OAK-D multi-sensor occupancy detector
  prompt.py              ← Gemini prompt string
  settings.py            ← Pydantic BaseSettings (layered config)
  state.py               ← thread-safe AppState class
  ui.py                  ← OpenCV overlay rendering
  web.py                 ← FastAPI web UI with MJPEG stream
  web_templates/         ← Jinja2 HTML templates
  web_static/            ← CSS/JS static files
>>>>>>> 34aad8e095f07c3e89ce542e0a6bda57512e48f1
scripts/
  migrate_json_to_pg.py ← one-time data migration to PostgreSQL
grafana/             ← Grafana provisioning and dashboards
```

## Testing

```bash
python -m pytest tests/
python -m pytest tests/ --cov=smartwaste --cov-report=term-missing
```

Test files follow `tests/test_<module>.py`. Install deps: `pip install -r requirements-test.txt`.
