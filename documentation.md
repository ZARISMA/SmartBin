# SmartBin Documentation

**Version:** 0.1.0
**Project:** Smart Waste AI
**Target Deployment:** Yerevan, Armenia

Real-time waste classification system using dual OAK-D USB3 depth cameras and the Google Gemini Vision API. Captures side-by-side frames from two cameras, classifies waste into seven categories, and logs results to a database with image files. Supports multiple deployment modes from local development to distributed edge devices reporting to a central server.

**Waste Categories:** Plastic, Glass, Paper, Organic, Aluminum, Other, Empty

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Quick Start Guide](#2-quick-start-guide)
3. [Architecture Overview](#3-architecture-overview)
4. [Configuration Reference](#4-configuration-reference)
5. [Entry Points](#5-entry-points)
6. [Module Reference](#6-module-reference)
7. [Web Application and API Reference](#7-web-application-and-api-reference)
8. [HTML Templates and Frontend](#8-html-templates-and-frontend)
9. [Database Schema and Operations](#9-database-schema-and-operations)
10. [AI Classification Pipeline](#10-ai-classification-pipeline)
11. [Sensor Fusion — OAK-D Native Mode](#11-sensor-fusion--oak-d-native-mode)
12. [Presence Detection — Auto-Gate Mode](#12-presence-detection--auto-gate-mode)
13. [Docker Deployment](#13-docker-deployment)
14. [Edge Architecture](#14-edge-architecture)
15. [Grafana Monitoring](#15-grafana-monitoring)
16. [Presentation Website](#16-presentation-website)
17. [Brand Color Palette](#17-brand-color-palette)
18. [Logging System](#18-logging-system)
19. [Testing](#19-testing)
20. [Development Tools](#20-development-tools)
21. [Project File Tree](#21-project-file-tree)

---

## 1. System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Camera | 1x OAK-D USB3 (OAK-1 W / OAK-1 Lite) | 2x OAK-D USB3 |
| Alternative Camera | 2x Raspberry Pi Cameras (picamera2) | — |
| USB | USB 3.0 ports | USB 3.0 hub with external power |
| RAM | 2 GB | 4 GB+ |
| Storage | 1 GB free | 10 GB+ (image dataset grows) |

### Software

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| OS (production) | Linux (Raspberry Pi OS, Ubuntu) |
| OS (development) | Windows 11, Linux, macOS |
| Docker (optional) | Docker Engine 20+ with Compose v2 |

### Python Dependencies

**Production (`requirements.txt`):**

| Package | Purpose |
|---------|---------|
| `opencv-python` | Frame capture, image processing, OpenCV window |
| `depthai` | OAK-D camera pipeline (DepthAI SDK 3.x) |
| `google-genai` | Google Gemini Vision API client |
| `openpyxl` | Excel export support |
| `pandas` | Data manipulation |
| `blobconverter` | Download pre-compiled MobileNetSSD blob for Myriad X |
| `tenacity` | Retry with exponential backoff for API calls |
| `pydantic-settings>=2.0` | Layered configuration with env var support |
| `psycopg2-binary` | PostgreSQL database driver |
| `fastapi` | Web framework for dashboard and REST API |
| `uvicorn[standard]` | ASGI server for FastAPI |
| `jinja2` | HTML template engine |
| `python-multipart` | Form data parsing for login |
| `itsdangerous` | Session cookie signing |

**Edge (`requirements-edge.txt`):**

| Package | Purpose |
|---------|---------|
| `opencv-python-headless` | OpenCV without GUI (headless edge) |
| `depthai` | OAK-D camera pipeline |
| `google-genai` | Gemini Vision API |
| `blobconverter` | MobileNetSSD blob download |
| `tenacity` | API retry logic |
| `pydantic-settings>=2.0` | Configuration |

**Testing (`requirements-test.txt`):**

| Package | Purpose |
|---------|---------|
| `pytest` | Test framework |
| `pytest-cov` | Coverage reporting |

---

## 2. Quick Start Guide

### Local Development Setup

```bash
# 1. Create and activate virtual environment
python -m venv oak_env
source oak_env/bin/activate          # Linux/macOS
# oak_env\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
export GEMINI_API_KEY='your_key_here'         # Linux/macOS
# $env:GEMINI_API_KEY='your_key_here'         # Windows PowerShell

# 4. Connect two OAK-D cameras via USB 3.0

# 5. Run manual mode
python main.py
```

### Runtime Keyboard Controls

| Key | Action | Available In |
|-----|--------|-------------|
| `c` | Classify current frame | All modes |
| `a` | Toggle auto-classify | Manual, Raspberry Pi |
| `r` | Reset background / recalibrate | Auto-gate, OAK Native |
| `q` | Quit | All modes |

### Running Each Mode

```bash
# Manual mode — dual OAK cameras, keyboard-triggered classification
python main.py --model gemini-3-flash-preview --auto-interval 6 --location Yerevan

# Auto-gate mode — presence detection triggers classification automatically
python mainauto.py --threshold 12.0 --detect-n 3 --empty-n 6

# OAK-D Native mode — sensor fusion (depth + motion + NN)
python mainoak.py --votes 2 --threshold 12.0

# Raspberry Pi mode — dual Pi cameras
python mainraspberry.py

# Web/Docker mode — FastAPI dashboard
python -m smartwaste.web
```

### CLI Entry Points (via `pyproject.toml`)

After `pip install -e .`:

```bash
smartwaste              # → main:main (manual mode)
smartwaste-auto         # → mainauto:main (auto-gate mode)
smartwaste-oak          # → mainoak:main (OAK native mode)
```

---

## 3. Architecture Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SmartBin System                                │
│                                                                         │
│  ┌──────────┐  ┌──────────┐                                            │
│  │ Camera A  │  │ Camera B  │   (OAK-D USB3 or Raspberry Pi)            │
│  └────┬─────┘  └────┬─────┘                                            │
│       │              │                                                  │
│       ▼              ▼                                                  │
│  ┌──────────────────────────┐                                           │
│  │  Frame Sync (MAX_DT)     │  Reject if timestamps differ > 0.25s     │
│  └────────────┬─────────────┘                                           │
│               │                                                         │
│               ▼                                                         │
│  ┌──────────────────────────┐                                           │
│  │  crop_sides(CROP_PERCENT)│  Remove 20% from each side               │
│  │  resize(DISPLAY_SIZE)    │  Scale to 800x800 per camera              │
│  │  hconcat([A, B])         │  Combine side-by-side → 1600x800         │
│  └────────────┬─────────────┘                                           │
│               │                                                         │
│               ▼                                                         │
│  ┌──────────────────────────┐                                           │
│  │  Strategy Decision       │  Manual: keyboard / timer                 │
│  │  (classify trigger)      │  Auto-gate: presence detection            │
│  └────────────┬─────────────┘  OAK Native: sensor fusion voting        │
│               │                                                         │
│               ▼                                                         │
│  ┌──────────────────────────┐                                           │
│  │  JPEG Encode             │  cv2.imencode @ JPEG_QUALITY (85)         │
│  └────────────┬─────────────┘                                           │
│               │                                                         │
│               ▼  (daemon thread)                                        │
│  ┌──────────────────────────┐                                           │
│  │  Gemini Vision API       │  base64 JPEG + PROMPT → JSON             │
│  │  (retry + circuit break) │  {category, description, brand_product}   │
│  └────────────┬─────────────┘                                           │
│               │                                                         │
│               ▼                                                         │
│  ┌──────────────────────────┐  ┌──────────────────────────┐             │
│  │  Database Insert         │  │  Image Save              │             │
│  │  (SQLite / PostgreSQL)   │  │  waste_dataset/*.jpg     │             │
│  └──────────────────────────┘  └──────────────────────────┘             │
│               │                                                         │
│               ▼  (if EDGE_MODE)                                         │
│  ┌──────────────────────────┐                                           │
│  │  POST /api/report        │  → Central server                        │
│  └──────────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Threading Model

The application uses a main thread + daemon worker thread architecture:

- **Main thread:** Camera capture, OpenCV window rendering, keyboard input, strategy logic
- **Daemon worker thread(s):** Gemini API calls run in short-lived daemon threads spawned by `launch_classify()`. The `AppState._is_classifying` mutex ensures only one classification runs at a time.

Thread safety is managed through `threading.Lock` in:
- `AppState` — protects `_label`, `_detail`, `_is_classifying`, `_history`
- `database.py` — `_init_lock` for lazy initialization; `ThreadedConnectionPool` for PostgreSQL
- `classifier.py` — `_cb_lock` for circuit breaker state

### Strategy Pattern

Classification triggers are decoupled from the capture loop via the Strategy pattern (`smartwaste/app.py`):

```
Strategy (ABC)
├── ManualStrategy          — keyboard 'c' + optional auto timer
└── PresenceGateStrategy    — pixel-diff presence gates API calls
```

The `run_loop()` function accepts any `Strategy` subclass. OAK Native mode (`mainoak.py`) implements its own loop with a state machine instead of using `run_loop()`.

### Module Dependency Graph

```
main.py / mainauto.py
    └── smartwaste/app.py (run_loop)
        ├── smartwaste/strategies.py (ManualStrategy, PresenceGateStrategy)
        │   ├── smartwaste/presence.py (PresenceDetector)
        │   └── smartwaste/utils.py (encode_frame, launch_classify)
        │       └── smartwaste/classifier.py (classify, Gemini API)
        │           ├── smartwaste/prompt.py (PROMPT)
        │           └── smartwaste/dataset.py (save_entry)
        │               ├── smartwaste/database.py (insert_entry)
        │               └── smartwaste/edge_client.py (report_classification)
        ├── smartwaste/cameraOak.py (make_pipeline, crop_sides)
        ├── smartwaste/state.py (AppState)
        ├── smartwaste/ui.py (draw_overlay)
        ├── smartwaste/config.py (constants)
        │   └── smartwaste/settings.py (Pydantic BaseSettings)
        └── smartwaste/log_setup.py (logging)

mainoak.py
    ├── smartwaste/oak_native.py (OAKOccupancyDetector)
    │   └── smartwaste/presence.py (PresenceDetector)
    ├── smartwaste/cameraOak.py
    ├── smartwaste/state.py
    ├── smartwaste/ui.py (draw_nn_detections)
    └── smartwaste/utils.py

smartwaste/web.py (FastAPI)
    ├── smartwaste/cameraOak.py / cameraraspberry.py / oak_native.py
    ├── smartwaste/state.py
    ├── smartwaste/database.py
    ├── smartwaste/schemas.py (EdgeReport, BinHeartbeat)
    └── smartwaste/edge_client.py (heartbeat thread)
```

---

## 4. Configuration Reference

### Configuration Priority (highest to lowest)

1. **CLI arguments** — entry points export CLI values as `SMARTWASTE_*` env vars before importing modules
2. **Shell environment variables** — `SMARTWASTE_*` prefix
3. **`.env` file** — in the project root, loaded by Pydantic
4. **Defaults** — defined in `smartwaste/settings.py`

### Settings Layer (`smartwaste/settings.py`)

The `Settings` class uses `pydantic_settings.BaseSettings` with:
- `env_prefix="SMARTWASTE_"` — all settings use this prefix except `GEMINI_API_KEY`
- `env_file=".env"` — auto-loads from project root
- `case_sensitive=False`
- `extra="ignore"` — unknown env vars are silently ignored

### Complete Configuration Table

#### Gemini API

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `gemini_api_key` | `GEMINI_API_KEY` or `SMARTWASTE_GEMINI_API_KEY` | str | `""` | Google Gemini API key (required) |
| `model_name` | `SMARTWASTE_MODEL_NAME` | str | `"gemini-3-flash-preview"` | Gemini model to use for classification |

#### Deployment

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `location` | `SMARTWASTE_LOCATION` | str | `"Yerevan"` | Location tag written to dataset entries |

#### Camera and Capture

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `jpeg_quality` | `SMARTWASTE_JPEG_QUALITY` | int | `85` | JPEG compression quality (1-100) for Gemini API uploads |
| `crop_percent` | `SMARTWASTE_CROP_PERCENT` | float | `0.20` | Fraction cropped from each side of frame (0.0-0.5) |
| `max_dt` | `SMARTWASTE_MAX_DT` | float | `0.25` | Max seconds between dual camera frames for sync |
| `auto_interval` | `SMARTWASTE_AUTO_INTERVAL` | int | `6` | Seconds between auto-classifications in manual mode |

#### Auto-Gate Mode (mainauto.py)

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `motion_threshold` | `SMARTWASTE_MOTION_THRESHOLD` | float | `12.0` | Pixel-diff score to trigger object detection (0-255) |
| `detect_confirm_n` | `SMARTWASTE_DETECT_CONFIRM_N` | int | `3` | Consecutive above-threshold checks to confirm presence |
| `empty_confirm_n` | `SMARTWASTE_EMPTY_CONFIRM_N` | int | `6` | Consecutive below-threshold checks to confirm bin empty |
| `bg_learning_rate` | `SMARTWASTE_BG_LEARNING_RATE` | float | `0.03` | Background model learning rate (0.0-1.0) |
| `bg_warmup_frames` | `SMARTWASTE_BG_WARMUP_FRAMES` | int | `40` | Frames to warm up background model before detecting |
| `check_interval` | `SMARTWASTE_CHECK_INTERVAL` | float | `0.5` | Seconds between presence checks |

#### OAK-D Native Mode (mainoak.py)

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `oak_display_w` | `SMARTWASTE_OAK_DISPLAY_W` | int | `960` | Display width for single-camera mode |
| `oak_display_h` | `SMARTWASTE_OAK_DISPLAY_H` | int | `540` | Display height for single-camera mode |
| `depth_roi_fraction` | `SMARTWASTE_DEPTH_ROI_FRACTION` | float | `0.4` | Region of interest fraction for depth sensor |
| `depth_change_threshold` | `SMARTWASTE_DEPTH_CHANGE_THRESHOLD` | int | `150` | Minimum depth change (mm) to detect object |
| `oak_calib_frames` | `SMARTWASTE_OAK_CALIB_FRAMES` | int | `30` | Frames for OAK sensor calibration |
| `imu_sample_rate_hz` | `SMARTWASTE_IMU_SAMPLE_RATE_HZ` | int | `100` | IMU sampling rate |
| `imu_shock_threshold` | `SMARTWASTE_IMU_SHOCK_THRESHOLD` | float | `1.3` | Accelerometer threshold for drop detection |
| `imu_baseline_samples` | `SMARTWASTE_IMU_BASELINE_SAMPLES` | int | `50` | IMU samples to establish baseline |
| `drop_flag_duration` | `SMARTWASTE_DROP_FLAG_DURATION` | float | `3.0` | Seconds to keep motion spike flag active |
| `nn_model_name` | `SMARTWASTE_NN_MODEL_NAME` | str | `"mobilenet-ssd"` | Neural network model for on-device detection |
| `nn_shaves` | `SMARTWASTE_NN_SHAVES` | int | `6` | Myriad X shaves allocated to NN |
| `nn_confidence` | `SMARTWASTE_NN_CONFIDENCE` | float | `0.5` | Minimum confidence for NN detections |
| `oak_votes_needed` | `SMARTWASTE_OAK_VOTES_NEEDED` | int | `2` | Sensor votes required to trigger classification |
| `oak_detect_confirm_n` | `SMARTWASTE_OAK_DETECT_CONFIRM_N` | int | `3` | Consecutive vote-passing checks to confirm detection |
| `oak_empty_confirm_n` | `SMARTWASTE_OAK_EMPTY_CONFIRM_N` | int | `5` | Consecutive zero-vote checks to confirm bin empty |
| `oak_check_interval` | `SMARTWASTE_OAK_CHECK_INTERVAL` | float | `0.4` | Seconds between state-machine ticks |
| `motion_spike_factor` | `SMARTWASTE_MOTION_SPIKE_FACTOR` | float | `3.0` | Presence score jump multiplier for drop event |

#### API Retry and Circuit Breaker

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `api_retry_attempts` | `SMARTWASTE_API_RETRY_ATTEMPTS` | int | `3` | Max tenacity retry attempts per classify call |
| `api_retry_min_wait` | `SMARTWASTE_API_RETRY_MIN_WAIT` | float | `1.0` | First backoff wait in seconds |
| `api_retry_max_wait` | `SMARTWASTE_API_RETRY_MAX_WAIT` | float | `8.0` | Max backoff wait in seconds |
| `cb_failure_threshold` | `SMARTWASTE_CB_FAILURE_THRESHOLD` | int | `5` | Consecutive final failures to open circuit |
| `cb_recovery_sec` | `SMARTWASTE_CB_RECOVERY_SEC` | float | `60.0` | Seconds to keep circuit open before recovery |

#### Database

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `db_backend` | `SMARTWASTE_DB_BACKEND` | str | `"sqlite"` | Database backend: `"sqlite"` or `"postgresql"` |
| `db_host` | `SMARTWASTE_DB_HOST` | str | `"localhost"` | PostgreSQL host |
| `db_port` | `SMARTWASTE_DB_PORT` | int | `5432` | PostgreSQL port |
| `db_name` | `SMARTWASTE_DB_NAME` | str | `"smartwaste"` | PostgreSQL database name |
| `db_user` | `SMARTWASTE_DB_USER` | str | `"smartwaste"` | PostgreSQL username |
| `db_password` | `SMARTWASTE_DB_PASSWORD` | str | `"smartwaste"` | PostgreSQL password |

#### Web UI

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `web_host` | `SMARTWASTE_WEB_HOST` | str | `"0.0.0.0"` | FastAPI bind host |
| `web_port` | `SMARTWASTE_WEB_PORT` | int | `8000` | FastAPI bind port |
| `camera_mode` | `SMARTWASTE_CAMERA_MODE` | str | `"oak"` | Camera backend: `"oak"`, `"raspberry"`, `"oak-native"`, or `"none"` |

#### Authentication

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `admin_username` | `SMARTWASTE_ADMIN_USERNAME` | str | `"admin"` | Dashboard login username |
| `admin_password` | `SMARTWASTE_ADMIN_PASSWORD` | str | `"password123"` | Dashboard login password |
| `secret_key` | `SMARTWASTE_SECRET_KEY` | str | `"smartwaste-session-secret-change-in-prod"` | Session cookie signing secret |

#### Bin Identity

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `bin_id` | `SMARTWASTE_BIN_ID` | str | `"bin-01"` | Unique identifier for this bin device |

#### Edge Mode

| Setting | Env Var | Type | Default | Description |
|---------|---------|------|---------|-------------|
| `edge_mode` | `SMARTWASTE_EDGE_MODE` | bool | `False` | Enable edge mode (POST results to central server) |
| `server_url` | `SMARTWASTE_SERVER_URL` | str | `""` | Central server URL (e.g. `http://192.168.1.100:8000`) |
| `edge_api_key` | `SMARTWASTE_EDGE_API_KEY` | str | `""` | Shared secret for edge-to-server authentication |
| `heartbeat_interval` | `SMARTWASTE_HEARTBEAT_INTERVAL` | int | `30` | Seconds between edge heartbeat signals |

### Derived Constants (`smartwaste/config.py`)

Constants not overridable (derived from install location):

| Constant | Value | Description |
|----------|-------|-------------|
| `BASE_DIR` | Project root | Calculated from `__file__` |
| `LOG_DIR` | `{BASE_DIR}/logs` | Runtime log files |
| `DATASET_DIR` | `{BASE_DIR}/waste_dataset` | Classified images and SQLite database |
| `DB_FILE` | `{DATASET_DIR}/waste.db` | SQLite database file path |
| `VALID_CLASSES` | `["Plastic", "Glass", "Paper", "Organic", "Aluminum", "Other", "Empty"]` | Accepted classification categories |
| `WINDOW` | `"Smart Waste AI (Dual OAK)"` | OpenCV window title (dual mode) |
| `OAK_WINDOW` | `"Smart Waste AI (OAK Native)"` | OpenCV window title (OAK native) |
| `DISPLAY_SIZE` | `(800, 800)` | Target frame dimensions per camera |

### Example `.env` File

```env
GEMINI_API_KEY=your_api_key_here
SMARTWASTE_LOCATION=Yerevan
SMARTWASTE_MODEL_NAME=gemini-3-flash-preview
SMARTWASTE_CAMERA_MODE=oak-native
SMARTWASTE_JPEG_QUALITY=85
SMARTWASTE_CROP_PERCENT=0.20
SMARTWASTE_AUTO_INTERVAL=6
SMARTWASTE_MOTION_THRESHOLD=12.0
SMARTWASTE_DETECT_CONFIRM_N=3
SMARTWASTE_EMPTY_CONFIRM_N=6
SMARTWASTE_CHECK_INTERVAL=0.5
SMARTWASTE_DEPTH_CHANGE_THRESHOLD=150
SMARTWASTE_IMU_SHOCK_THRESHOLD=1.3
SMARTWASTE_OAK_VOTES_NEEDED=2
SMARTWASTE_NN_CONFIDENCE=0.5
SMARTWASTE_API_RETRY_ATTEMPTS=3
SMARTWASTE_API_RETRY_MIN_WAIT=1.0
SMARTWASTE_API_RETRY_MAX_WAIT=8.0
SMARTWASTE_CB_FAILURE_THRESHOLD=5
SMARTWASTE_CB_RECOVERY_SEC=60
SMARTWASTE_DB_BACKEND=sqlite
SMARTWASTE_DB_HOST=localhost
SMARTWASTE_DB_PORT=5432
SMARTWASTE_DB_NAME=smartwaste
SMARTWASTE_DB_USER=smartwaste
SMARTWASTE_DB_PASSWORD=smartwaste
SMARTWASTE_WEB_HOST=0.0.0.0
SMARTWASTE_WEB_PORT=8000
```

---

## 5. Entry Points

### `main.py` — Manual Mode (Dual OAK Cameras)

**CLI command:** `python main.py` or `smartwaste`

**Description:** Manual classification mode using two OAK-D USB3 cameras. Frames from both cameras are displayed in an OpenCV window. The user triggers classification by pressing `c`, or enables auto-classify with `a`.

**CLI Arguments:**

| Argument | Env Var | Description |
|----------|---------|-------------|
| `--model NAME` | `SMARTWASTE_MODEL_NAME` | Gemini model name |
| `--auto-interval SEC` | `SMARTWASTE_AUTO_INTERVAL` | Seconds between auto-classifications |
| `--location NAME` | `SMARTWASTE_LOCATION` | Deployment location tag |

**Functions:**
- `_parse() -> argparse.Namespace` — Parses CLI arguments
- `main() -> None` — Exports CLI overrides as env vars, then calls `run_loop(ManualStrategy())`

**Key behavior:**
- Requires exactly 2 OAK-D devices connected via USB
- Opens a 1600x800 OpenCV window showing both camera feeds side-by-side
- On Linux, provides udev rules hint if cameras are not found

### `mainauto.py` — Auto-Gate Mode (Dual OAK Cameras)

**CLI command:** `python mainauto.py` or `smartwaste-auto`

**Description:** Automatic presence-detection gate mode. A rolling background model detects when an object enters the bin and fires a single Gemini API call per item. The user does not need to press keys for normal operation.

**CLI Arguments:**

| Argument | Env Var | Description |
|----------|---------|-------------|
| `--model NAME` | `SMARTWASTE_MODEL_NAME` | Gemini model |
| `--threshold FLOAT` | `SMARTWASTE_MOTION_THRESHOLD` | Pixel-diff threshold (0-255) |
| `--detect-n INT` | `SMARTWASTE_DETECT_CONFIRM_N` | Consecutive detections to confirm |
| `--empty-n INT` | `SMARTWASTE_EMPTY_CONFIRM_N` | Consecutive empties to clear |
| `--location NAME` | `SMARTWASTE_LOCATION` | Location tag |

**State machine:** Calibrating -> Ready/IDLE -> Detected -> Classified -> Ready/IDLE

**Functions:**
- `_parse() -> argparse.Namespace` — Parses CLI arguments
- `main() -> None` — Exports CLI overrides, calls `run_loop(PresenceGateStrategy())`

### `mainoak.py` — OAK-D Native Mode (Sensor Fusion)

**CLI command:** `python mainoak.py` or `smartwaste-oak`

**Description:** Sensor fusion mode supporting 1 or 2 OAK cameras. The primary camera runs three sensors (presence, motion spike, MobileNetSSD) that vote on bin occupancy. A second camera (optional) provides a second viewing angle for the Gemini dual-view prompt.

**CLI Arguments:**

| Argument | Env Var | Description |
|----------|---------|-------------|
| `--model NAME` | `SMARTWASTE_MODEL_NAME` | Gemini model |
| `--threshold FLOAT` | `SMARTWASTE_MOTION_THRESHOLD` | Presence motion threshold |
| `--votes N` | `SMARTWASTE_OAK_VOTES_NEEDED` | Sensor votes needed |
| `--location NAME` | `SMARTWASTE_LOCATION` | Location tag |

**State machine (`OakState` enum):**

```
Calibrating → Ready → Detected → Classifying → Classified → Ready
                ↑                                            │
                └────────────────────────────────────────────┘
```

| State | Description | Transition |
|-------|-------------|------------|
| `CALIBRATING` | Warming up presence detector background model | -> `READY` when calibration completes |
| `READY` | Waiting for object, monitoring sensor votes | -> `DETECTED` after `OAK_DETECT_CONFIRM_N` consecutive vote-passing checks |
| `DETECTED` | Object confirmed, firing Gemini API | -> `CLASSIFYING` when API call starts |
| `CLASSIFYING` | Waiting for Gemini response | -> `CLASSIFIED` when `is_classifying` becomes False |
| `CLASSIFIED` | Showing result, waiting for bin to empty | -> `READY` after `OAK_EMPTY_CONFIRM_N` consecutive zero-vote checks |

**Dual vs. Single Camera:**
- **2 cameras:** Frames from both cameras are cropped, resized to `DISPLAY_SIZE`, and concatenated side-by-side. Display size is `1600x800`.
- **1 camera:** Single frame displayed at `OAK_DISPLAY_W x OAK_DISPLAY_H` (960x540).

**Key functions:**
- `_export_cli_overrides() -> None` — Sets env vars before module imports
- `_draw_overlay(frame, oak_state, votes, app_state, detector, calib_pct, title) -> None` — Draws 4-line status bar (title+state, votes, sensor readings, result)
- `_active_count(detector) -> int` — Returns number of available sensors (2 or 3)
- `_tick(state, votes, app_state, detector, ...) -> tuple[OakState, int, int]` — Advances state machine by one tick
- `_handle_key(key, oak_state, votes, app_state, detector, ...) -> OakState` — Processes keyboard input
- `main() -> None` — Full entry point: device discovery, pipeline setup, main loop

### `mainraspberry.py` — Raspberry Pi Mode

**CLI command:** `python mainraspberry.py`

**Description:** Dual Raspberry Pi camera mode using `picamera2`. Functionally similar to manual mode but uses Pi cameras instead of OAK devices.

**Requirements:**
- 2 Pi cameras connected (CSI or USB)
- `camera_auto_detect=1` in `/boot/firmware/config.txt` (Raspberry Pi 5)
- Desktop session active (for OpenCV window)

**Key behavior:**
- Creates `Picamera2` instances for camera indices 0 and 1
- Captures 1280x720 BGR888 frames
- Supports same `c`/`a`/`q` keyboard controls as manual mode

### `python -m smartwaste.web` — Web/Docker Mode

**Description:** Runs the FastAPI web dashboard and API server. Used in Docker deployments and for remote access.

**Behavior:**
- Starts Uvicorn on `WEB_HOST:WEB_PORT` (default `0.0.0.0:8000`)
- Spawns camera thread based on `CAMERA_MODE` setting
- If `EDGE_MODE` is enabled, starts heartbeat thread
- Serves dashboard at `/`, per-bin views at `/bin/{id}`, public site at `/site`

---

## 6. Module Reference

### `smartwaste/settings.py` — Layered Configuration

**Purpose:** Centralized configuration using Pydantic BaseSettings with environment variable overrides.

**Class: `Settings(BaseSettings)`**

| Attribute | Description |
|-----------|-------------|
| `model_config` | `SettingsConfigDict` with `env_prefix="SMARTWASTE_"`, `.env` file loading |
| All settings | See [Configuration Reference](#4-configuration-reference) |

**Module-level singleton:**
```python
settings = Settings()  # imported by config.py and classifier.py
```

The `.env` file path is resolved relative to the package directory: `Path(__file__).parent.parent / ".env"`.

### `smartwaste/config.py` — Runtime Constants

**Purpose:** Provides flat module-level constants derived from the `settings` singleton. Existing code imports constants from here:

```python
from .config import MODEL_NAME, VALID_CLASSES
```

All values are read from `settings.*` at module load time. File/directory paths (`BASE_DIR`, `LOG_DIR`, `DATASET_DIR`, `DB_FILE`) are computed from the install location and are not overridable.

### `smartwaste/state.py` — Thread-Safe Application State

**Purpose:** Coordinates shared flags and display state between the main thread and classifier daemon threads.

**Class: `AppState`**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `() -> None` | Initializes with `_label="Ready"`, `_is_classifying=False`, `auto_classify=False`, `last_capture_time=0.0` |
| `get_display` | `() -> tuple[str, str, bool]` | Returns `(label, detail, auto_classify)` under lock |
| `set_status` | `(label: str, detail: str) -> None` | Updates label and detail under lock |
| `start_classify` | `() -> bool` | Atomic check-and-set: returns `True` if not already classifying, `False` if busy |
| `finish_classify` | `() -> None` | Clears `_is_classifying` flag under lock |
| `add_to_history` | `(label: str) -> None` | Prepends `(HH:MM, label)` to history ring (max 5 entries) |
| `get_history` | `() -> list[tuple[str, str]]` | Returns up to 5 recent `(time, label)` pairs |
| `is_classifying` | `@property -> bool` | Read-only check of classifying state |
| `toggle_auto` | `() -> bool` | Flips `auto_classify` and returns new value |

**Thread safety:** `_lock` (threading.Lock) protects `_label`, `_detail`, `_is_classifying`, and `_history`. `auto_classify` and `last_capture_time` are main-thread-only.

### `smartwaste/app.py` — Shared OAK Camera Run Loop

**Purpose:** Generic capture/display/classify loop for dual OAK cameras, parameterized by a Strategy.

**Class: `Strategy(ABC)`**

| Method | Signature | Description |
|--------|-----------|-------------|
| `setup` | `(state: AppState) -> None` | Called once before loop starts |
| `on_combined_frame` | `(combined: np.ndarray, state: AppState) -> None` | Called each iteration with synced dual-camera frame (abstract) |
| `on_key` | `(key: int, combined: np.ndarray \| None, state: AppState) -> None` | Called for non-quit keypresses (default: no-op) |

**Function: `run_loop(strategy: Strategy) -> None`**

1. Creates `AppState`, calls `strategy.setup(state)`
2. Opens OpenCV window (1600x800)
3. Discovers 2 OAK devices, creates pipelines via `make_pipeline()`
4. Main loop:
   - Capture frames from both queues
   - Crop sides, resize to `DISPLAY_SIZE`, sync by `MAX_DT`
   - Concatenate horizontally, draw overlay, display
   - Call `strategy.on_combined_frame()` if combined frame available
   - Handle `q` key (quit), pass other keys to `strategy.on_key()`
5. Cleanup: destroy windows, stop pipelines

### `smartwaste/strategies.py` — Classification Trigger Strategies

**Purpose:** Concrete Strategy implementations for manual and auto-gate modes.

**Class: `ManualStrategy(Strategy)`**

| Method | Description |
|--------|-------------|
| `on_combined_frame` | If `auto_classify` is on and `AUTO_INTERVAL` has elapsed, triggers classification |
| `on_key` | `c` -> classify once; `a` -> toggle auto-classify |

**Class: `PresenceGateStrategy(Strategy)`**

| Method | Description |
|--------|-------------|
| `setup` | Sets `auto_classify = True` |
| `on_combined_frame` | Runs presence detector at `CHECK_INTERVAL`; manages state machine (Calibrating -> Ready -> Detected -> Classified); fires exactly one API call per item |
| `on_key` | `c` -> force-classify; `r` -> reset background model |

**PresenceGateStrategy state transitions:**
- `is_empty and _bin_occupied` -> clear bin, reset background, back to Ready
- `is_occupied and not _bin_occupied` -> mark occupied, set "Detected"
- `_bin_occupied and not _item_classified` -> fire API, mark classified

### `smartwaste/classifier.py` — Gemini Classification Worker

**Purpose:** Calls the Gemini Vision API with retry logic and circuit breaker protection.

**Gemini Client:**
```python
client = genai.Client(api_key=settings.gemini_api_key)
```
Built at module load time. Raises `RuntimeError` if `GEMINI_API_KEY` is not set.

**Function: `_extract_json(text: str) -> dict`**

Parses Gemini response text into a dict:
1. Strips markdown code fences (` ```json ... ``` `)
2. Tries `json.loads()` on cleaned text
3. Falls back to finding first `{` and last `}` and parsing the substring
4. Raises `json.JSONDecodeError` if no JSON found

**Function: `_call_gemini(img_bytes: bytes) -> str`**

Decorated with `@retry` (tenacity):
- `stop_after_attempt(api_retry_attempts)` (default: 3)
- `wait_exponential(min=1.0, max=8.0)`
- Quota errors (429, RESOURCE_EXHAUSTED) are NOT retried

Sends base64-encoded JPEG + PROMPT to Gemini and returns raw text.

**Function: `classify(img_bytes: bytes, img_original, state) -> None`**

Main worker (runs in daemon thread):
1. Check circuit breaker — skip if open
2. Set status "Classifying..."
3. Call `_call_gemini(img_bytes)` with retry
4. Record success (reset circuit breaker)
5. Extract JSON, normalize category label
6. Set status to result, add to history
7. If not "Empty", call `save_entry()`
8. On error: record failure, check for quota hit, update status
9. Always: call `state.finish_classify()` in `finally`

**Circuit Breaker:**

| State | Behavior |
|-------|----------|
| Closed (normal) | API calls proceed. Failures increment `_cb_failures`. |
| Open | After `cb_failure_threshold` (5) consecutive failures, circuit opens for `cb_recovery_sec` (60s). All calls skip during this period. |
| Half-open | After recovery window elapses, one call is allowed. Success closes circuit; failure re-opens it. |

### `smartwaste/prompt.py` — Gemini Prompt

**Purpose:** Contains the exact prompt string sent to Gemini with every classification request.

**Constant: `PROMPT` (str)**

The prompt instructs Gemini to:
1. Recognize two side-by-side camera views (LEFT = Camera A, RIGHT = Camera B)
2. Identify objects inside a green plastic pipe, ignoring the pipe itself
3. Ignore the transparent protective glass cover and its reflections
4. Respond in strict JSON format: `{"category": "...", "description": "...", "brand_product": "..."}`
5. Use exactly one of: Plastic, Glass, Paper, Organic, Aluminum, Other, Empty
6. Recognize Armenian brands (Jermuk, Bjni, BOOM)
7. Return `category="Empty"` if pipe is empty

### `smartwaste/presence.py` — Pixel-Diff Presence Detector

**Purpose:** Local bin-occupancy detection using a rolling background model. No API calls.

**Class: `PresenceDetector`**

| Property/Method | Signature | Description |
|-----------------|-----------|-------------|
| `ready` | `@property -> bool` | True once warmup complete |
| `warmup_progress` | `@property -> tuple[int, int]` | `(current, total)` warmup frame counts |
| `update` | `(gray: np.ndarray) -> tuple[float, bool, bool]` | Feed grayscale frame; returns `(score, is_occupied, is_empty)` |
| `accept_as_background` | `(gray: np.ndarray) -> None` | Hard-snap background to current frame |
| `reset` | `(gray: np.ndarray \| None = None) -> None` | Full reset; if `gray` provided, skips warmup |

**Algorithm:**
1. **Warmup phase** (first `BG_WARMUP_FRAMES` frames): Fast-learns background using `cv2.accumulateWeighted` with rate 0.15. Never fires detections.
2. **Detection phase:** Computes `score = mean(|current - background|)`.
   - If `score >= MOTION_THRESHOLD`: increment `_detect_streak`, reset `_empty_streak`
   - If `score < MOTION_THRESHOLD`: increment `_empty_streak`, reset `_detect_streak`, slowly drift background
3. `is_occupied = _detect_streak >= DETECT_CONFIRM_N`
4. `is_empty = _empty_streak >= EMPTY_CONFIRM_N`

### `smartwaste/oak_native.py` — OAK Multi-Sensor Occupancy Detector

**Purpose:** Builds a DepthAI 3.x pipeline with presence detection, motion spike detection, and MobileNetSSD on the Myriad X VPU.

**Named Tuples:**

```python
class Detection(NamedTuple):
    xmin: float; ymin: float; xmax: float; ymax: float
    confidence: float; label_idx: int

class SensorVotes(NamedTuple):
    presence_occupied: bool    # pixel-diff background model
    motion_spike: bool         # sudden score jump
    nn_occupied: bool          # MobileNetSSD detections > 0
    votes: int                 # sum of three booleans
    rgb_frame: np.ndarray | None
    presence_score: float      # pixel-diff score (0-255)
    motion_delta: float        # score jump magnitude
    nn_count: int              # number of NN detections
    nn_detections: list[Detection]
```

**Function: `_try_get_blob() -> tuple[str | None, bool]`**

Downloads MobileNetSSD blob via `blobconverter.from_zoo()`. Returns `(path, True)` on success or `(None, False)` if unavailable.

**Function: `build_oak_pipeline(device) -> tuple[pipeline, rgb_q, nn_q, nn_available]`**

Creates DepthAI pipeline:
1. RGB camera node on `CAM_A` with full-resolution BGR888p output queue
2. MobileNetSSD detection network (if blob available): 300x300 input from camera preview, 2 inference threads
3. Starts pipeline

**Class: `OAKOccupancyDetector`**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(device: dai.Device)` | Builds pipeline, initializes PresenceDetector and motion state |
| `calibrate` | `() -> bool` | Feeds frames to presence detector during warmup; returns True when ready |
| `update` | `() -> SensorVotes` | Drains all queues, computes all three sensor votes, returns results |
| `calibration_progress` | `() -> int` | Returns 0-100 percentage |
| `reset` | `() -> None` | Clears all state, restarts calibration |
| `stop` | `() -> None` | Stops DepthAI pipeline |
| `presence_ready` | `@property -> bool` | Whether presence detector is calibrated |
| `nn_available` | `@property -> bool` | Whether NN pipeline is active |

**Motion spike detection:** Replaces physical IMU. Detects when `(current_score - previous_score) > MOTION_THRESHOLD * MOTION_SPIKE_FACTOR`. The spike flag auto-expires after `DROP_FLAG_DURATION` seconds.

### `smartwaste/camera.py` — Single OAK Camera Pipeline

**Purpose:** Creates and starts a camera pipeline for one OAK device.

**Function: `make_pipeline(device: dai.Device) -> tuple[pipeline, queue]`**

Creates a DepthAI pipeline with a single camera node on `CAM_A`, outputting full-resolution BGR888p frames to a queue (maxSize=4, non-blocking).

**Function: `crop_sides(frame: np.ndarray, crop_percent: float) -> np.ndarray`**

Removes `crop_percent` fraction from both left and right sides of the frame. Returns the frame unchanged if `crop_percent <= 0`.

### `smartwaste/cameraOak.py` — Dual OAK Camera Pipeline

Identical to `camera.py`. Contains the same `make_pipeline()` and `crop_sides()` functions. Used by `main.py`, `mainauto.py`, and `mainoak.py`.

### `smartwaste/cameraraspberry.py` — Raspberry Pi Camera Module

**Purpose:** Manages Raspberry Pi cameras via `picamera2`.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `make_cameras` | `(num_cameras: int = 2) -> list` | Creates and starts `Picamera2` instances (1280x720 BGR888) |
| `grab_frame` | `(cam) -> np.ndarray` | Captures single BGR frame |
| `stop_cameras` | `(cameras: list) -> None` | Stops all cameras cleanly |
| `crop_sides` | `(frame, crop_percent) -> np.ndarray` | Same cropping as OAK module |

### `smartwaste/database.py` — Dual-Backend Persistence Layer

**Purpose:** SQLite and PostgreSQL database operations with automatic backend selection.

**Backend selection:** `DB_BACKEND == "postgresql" and psycopg2 available`

**Schemas:**

SQLite:
```sql
CREATE TABLE IF NOT EXISTS waste_entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT,
    label         TEXT,
    description   TEXT,
    brand_product TEXT,
    location      TEXT,
    weight        TEXT,
    timestamp     TEXT,
    simulated_temperature   REAL,
    simulated_humidity      REAL,
    simulated_vibration     REAL,
    simulated_air_pollution REAL,
    simulated_smoke         REAL,
    bin_id        TEXT DEFAULT 'bin-01'
);
```

PostgreSQL:
```sql
CREATE TABLE IF NOT EXISTS waste_entries (
    id                      SERIAL PRIMARY KEY,
    filename                TEXT,
    label                   TEXT NOT NULL,
    description             TEXT,
    brand_product           TEXT,
    location                TEXT,
    weight                  TEXT,
    timestamp               TIMESTAMPTZ DEFAULT NOW(),
    simulated_temperature   DOUBLE PRECISION,
    simulated_humidity      DOUBLE PRECISION,
    simulated_vibration     DOUBLE PRECISION,
    simulated_air_pollution DOUBLE PRECISION,
    simulated_smoke         DOUBLE PRECISION,
    bin_id                  TEXT DEFAULT 'bin-01'
);
CREATE INDEX IF NOT EXISTS idx_waste_label ON waste_entries(label);
CREATE INDEX IF NOT EXISTS idx_waste_ts ON waste_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_waste_bin_id ON waste_entries(bin_id);
```

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `init_db` | `() -> None` | Creates table if not exists; runs `_migrate_add_bin_id()` |
| `insert_entry` | `(entry: dict, env: dict) -> int \| None` | Inserts classification row; returns row ID |
| `get_entries` | `(limit=100, offset=0, bin_id=None) -> list[dict]` | Recent entries, newest first, optional bin filter |
| `get_label_counts` | `(bin_id=None) -> dict[str, int]` | `{label: count}` for all categories |
| `get_entry_count` | `(bin_id=None) -> int` | Total number of entries |
| `get_entries_by_bin` | `(bin_id, limit=100, offset=0) -> list[dict]` | Shortcut for filtered `get_entries` |
| `get_label_counts_by_bin` | `(bin_id) -> dict[str, int]` | Shortcut for filtered `get_label_counts` |
| `get_active_bins` | `() -> list[dict]` | Distinct bin_ids with entry counts and last timestamps |

**Connection pooling:** PostgreSQL uses `psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=5)`.

**Lazy initialization:** `_ensure_init()` calls `init_db()` once on first database operation.

**Migration:** `_migrate_add_bin_id()` adds `bin_id` column to existing tables that lack it.

### `smartwaste/dataset.py` — Dataset Entry Management

**Purpose:** Saves classified images and inserts database entries. Optionally reports to central server in edge mode.

**Function: `_environment_data() -> dict`**

Generates simulated sensor data:

| Key | Range |
|-----|-------|
| `simulated_temperature` | 15.0 - 30.0 |
| `simulated_humidity` | 30.0 - 70.0 |
| `simulated_vibration` | 0.0 - 0.1 |
| `simulated_air_pollution` | 5.0 - 50.0 |
| `simulated_smoke` | 0.0 - 1.0 |

**Function: `save_entry(label: str, img, description: str, brand_product: str) -> None`**

1. Generates filename: `{label}_{YYYY-MM-DD_HH-MM-SS}.jpg`
2. Saves image to `DATASET_DIR` via `cv2.imwrite()`
3. Constructs entry dict with all fields
4. Calls `insert_entry(entry, _environment_data())`
5. If `EDGE_MODE and SERVER_URL`: imports and calls `report_classification()` with image bytes

### `smartwaste/edge_client.py` — Edge-to-Server HTTP Client

**Purpose:** HTTP client for edge devices to report classifications and send heartbeats to a central server. Uses `urllib` (stdlib) to minimize dependencies.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `_headers` | `() -> dict[str, str]` | Returns `{"Content-Type": "application/json", "Authorization": "Bearer {EDGE_API_KEY}"}` |
| `_post` | `(path: str, payload: dict) -> bool` | POST JSON to `SERVER_URL + path`, 10s timeout, returns success |
| `report_classification` | `(entry: dict, env: dict, image_bytes=None) -> bool` | POSTs classification to `/api/report` with optional base64 image |
| `send_heartbeat` | `() -> bool` | POSTs to `/api/heartbeat` with `bin_id`, status, camera_mode, uptime |
| `start_heartbeat_thread` | `() -> None` | Starts daemon thread sending heartbeats every `HEARTBEAT_INTERVAL` seconds |

**Uptime tracking:** `_start_time = time.monotonic()` captured at module load.

### `smartwaste/schemas.py` — Pydantic Request Models

**Purpose:** Request body models for edge communication endpoints.

**Class: `EdgeReport(BaseModel)`**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bin_id` | str | required | Device identifier |
| `label` | str | required | Classification label |
| `description` | str | `"N/A"` | Item description |
| `brand_product` | str | `"Unknown"` | Recognized brand |
| `location` | str | `""` | Bin location |
| `weight` | str | `""` | Item weight |
| `timestamp` | str | required | ISO datetime string |
| `simulated_temperature` | float | `0.0` | Simulated temp sensor |
| `simulated_humidity` | float | `0.0` | Simulated humidity sensor |
| `simulated_vibration` | float | `0.0` | Simulated vibration sensor |
| `simulated_air_pollution` | float | `0.0` | Simulated air quality sensor |
| `simulated_smoke` | float | `0.0` | Simulated smoke sensor |
| `image_b64` | str \| None | `None` | Optional base64-encoded JPEG |

**Class: `BinHeartbeat(BaseModel)`**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bin_id` | str | required | Device identifier |
| `status` | str | `"online"` | `"online"` or `"idle"` |
| `camera_mode` | str | `""` | Camera backend type |
| `uptime_seconds` | float | `0.0` | Seconds since process start |

### `smartwaste/web.py` — FastAPI Web Application

See [Section 7: Web Application and API Reference](#7-web-application-and-api-reference) for complete documentation.

### `smartwaste/ui.py` — OpenCV Overlay Rendering

**Purpose:** Draws status overlays on OpenCV frames.

**Color Palette (BGR):**

| Name | BGR Value | Hex |
|------|-----------|-----|
| `_C_WHITE` | `(255, 255, 255)` | #FFFFFF |
| `_C_GRAY` | `(140, 140, 140)` | #8C8C8C |
| `_C_DARK_PANEL` | `(20, 26, 11)` | #0B1A14 |
| `_C_ACCENT` | `(66, 90, 45)` | #2D5A42 |
| `_C_MODE_AUTO` | `(80, 175, 76)` | #4CAF50 |
| `_C_MODE_MANUAL` | `(107, 77, 26)` | #1A4D6B |

**Category Colors (BGR):**

| Category | BGR | Hex |
|----------|-----|-----|
| Plastic | `(235, 206, 135)` | #87CEEB |
| Glass | `(208, 224, 64)` | #40E0D0 |
| Paper | `(140, 180, 210)` | #D2B48C |
| Organic | `(43, 77, 30)` | #1E4D2B |
| Aluminum | `(169, 169, 169)` | #A9A9A9 |
| Other | `(219, 112, 147)` | #9370DB |
| Empty | `(140, 140, 140)` | #8C8C8C |

**Function: `draw_overlay(img, label, detail, auto_on, history=None) -> None`**

Modifies `img` in-place. Draws:
1. Semi-transparent dark top panel (72% opacity, 19% frame height)
2. Accent divider line
3. "Smart Waste AI" title
4. Category label with color dot
5. Detail line (descriptor/brand)
6. Mode badge ("AUTO" or "MANUAL") in top-right corner
7. Classification history (right column, up to 5 entries)

Font scaling is relative to frame width: `base = max(w / 1600.0, 0.35)`.

**Function: `draw_nn_detections(img, detections) -> None`**

Draws green bounding boxes with confidence labels for each NN detection. Uses Pascal VOC 21-class labels:
```
background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat,
chair, cow, diningtable, dog, horse, motorbike, person, pottedplant,
sheep, sofa, train, tvmonitor
```

### `smartwaste/utils.py` — Shared Frame Helpers

**Function: `encode_frame(frame) -> bytes | None`**

JPEG-encodes a BGR frame using `cv2.imencode()` with `JPEG_QUALITY`. Returns bytes or None on failure.

**Function: `launch_classify(img_bytes: bytes | None, frame_copy, state: AppState) -> None`**

If `img_bytes` is truthy, starts a daemon thread running `classify(img_bytes, frame_copy, state)`. If bytes are None/empty, sets error status and calls `finish_classify()`.

### `smartwaste/log_setup.py` — Logging Configuration

**Purpose:** Configures the "smartwaste" logger with file and console handlers.

**Constants:**
- `RUN_ID` — Timestamp string `YYYYMMDD_HHMMSS` (set at import time)
- `LOG_FILE` — `logs/run_{RUN_ID}.log`
- `ERR_JSON_FILE` — `logs/last_api_error_{RUN_ID}.json`

**Function: `get_logger() -> logging.Logger`**

Returns the "smartwaste" logger. On first call, configures:
- Level: `INFO`
- Format: `[%(asctime)s] %(levelname)s %(message)s`
- Handlers: `FileHandler(LOG_FILE)` + `StreamHandler(stderr)`

Creates `logs/` directory automatically.

---

## 7. Web Application and API Reference

### Application Setup

- **Framework:** FastAPI
- **Server:** Uvicorn
- **Templates:** Jinja2 (`smartwaste/web_templates/`)
- **Static files:** Mounted at `/static` from `smartwaste/web_static/`
- **Session middleware:** Starlette `SessionMiddleware` with `SECRET_KEY`

### Authentication

Two authentication methods are supported:

1. **Session cookie:** `request.session.get("user")` — set after successful form login
2. **Bearer token:** `Authorization: Bearer {ADMIN_PASSWORD}` — used by edge devices and API clients

Default credentials: `admin` / `password123` (override via env vars).

### Lifespan

On startup:
- Starts camera thread (`_start_camera_thread()`) based on `CAMERA_MODE`
- If `EDGE_MODE` is enabled, starts heartbeat thread

### Complete Route Reference

#### Authentication Routes (Public)

| Method | Path | Auth | Request | Response | Description |
|--------|------|------|---------|----------|-------------|
| GET | `/login` | No | — | HTML (login.html) | Login page; redirects to `/` if already authenticated |
| POST | `/login` | No | Form: `username`, `password` | 302 redirect | Authenticate; sets session on success, returns 401 on failure |
| POST | `/logout` | Yes | — | 302 redirect to `/login` | Clears session |

#### Public Routes

| Method | Path | Auth | Response | Description |
|--------|------|------|----------|-------------|
| GET | `/site` | No | HTML (site.html) | Public presentation website |

#### Protected Dashboard Routes

| Method | Path | Auth | Response | Description |
|--------|------|------|----------|-------------|
| GET | `/` | Yes | HTML (dashboard.html) | Multi-bin overview dashboard |
| GET | `/bin/{bin_id}` | Yes | HTML (index.html) | Per-bin detail view with live stream |
| GET | `/stream` | Yes | MJPEG stream | Live video feed (~30 FPS) |

#### Protected API Endpoints

| Method | Path | Auth | Query Params | Request Body | Response | Description |
|--------|------|------|-------------|-------------|----------|-------------|
| POST | `/api/classify` | Yes | — | — | `{"status": "classifying"}` (200) or 503/409 | Trigger classification on current frame |
| POST | `/api/toggle-auto` | Yes | — | — | `{"auto_classify": bool}` | Toggle auto-classify mode |
| GET | `/api/state` | Yes | — | — | `{"label", "detail", "auto_on", "is_classifying", "history"}` | Current app state |
| GET | `/api/entries` | Yes | `limit` (20), `offset` (0), `bin_id` | — | `[{entry}, ...]` | Recent classification entries |
| GET | `/api/stats` | Yes | `bin_id` | — | `{"total": int, "by_category": {...}}` | Classification statistics |
| GET | `/api/bins` | Yes | — | — | `[{bin}, ...]` | Active bins from database |
| GET | `/api/dashboard` | Yes | — | — | `{"bins": [...], "total_bins", "total_entries"}` | Dashboard summary data |

#### Edge Device Endpoints

| Method | Path | Auth | Request Body | Response | Description |
|--------|------|------|-------------|----------|-------------|
| POST | `/api/report` | Yes | `EdgeReport` JSON | `{"status": "ok", "id": int}` | Log classification from edge device |
| POST | `/api/heartbeat` | Yes | `BinHeartbeat` JSON | `{"status": "ok"}` | Update bin online status |

### Camera Thread Management

The camera thread is selected by `CAMERA_MODE`:

| Mode | Thread Function | Requirements |
|------|----------------|-------------|
| `oak` | `_camera_loop_oak()` | 2x OAK-D cameras |
| `raspberry` | `_camera_loop_raspberry()` | 2x Pi cameras |
| `oak-native` | `_camera_loop_oak_native()` | 1+ OAK-D camera |
| `none` | No thread started | No cameras (server-only mode) |

Camera loops capture frames, apply cropping/resizing, update the shared frame buffer via `_set_frame()`, and trigger auto-classification when enabled.

### Bin Registry

An in-memory registry tracks active bins:

```python
@dataclass
class BinInfo:
    bin_id: str
    status: str = "online"      # online, idle, offline
    last_seen: datetime
    camera_mode: str = ""
    uptime_seconds: float = 0.0
```

Bins are marked offline if `(now - last_seen) > 60 seconds`. The `/api/dashboard` endpoint merges heartbeat registry data with database statistics.

---

## 8. HTML Templates and Frontend

### `dashboard.html` — Multi-Bin Overview

**Template variables:** None (all data loaded via JavaScript).

**Layout:**
- Full-width dashboard with dark theme
- Summary statistics bar: Total Bins, Online Now, Total Classifications
- Dynamic grid of bin cards (CSS `auto-fill`, min 320px columns)

**JavaScript polling:**
- `pollDashboard()` every 5 seconds — fetches `/api/dashboard`, rebuilds grid
- Each bin card links to `/bin/{bin_id}`

**Bin card content:** Bin ID with online/offline status dot, location, camera mode, classification count, last activity timestamp.

### `index.html` — Per-Bin Detail View

**Template variables:**
- `bin_id` (str) — Current bin identifier
- `has_local_camera` (bool) — True if live stream available

**Layout:** 2-column (stream panel left, info sidebar right, responsive column below 900px).

**Sections:**
1. **Stream panel** — `<img src="/stream">` MJPEG feed (only if `has_local_camera`)
2. **Status card** — Current classification label with category color dot, detail text
3. **Controls** — Classify button (with ripple animation), Toggle Auto button
4. **History card** — Last 5 classifications with timestamps (slide-in animation)
5. **Statistics card** — Total count, category bars with gradient fills
6. **Entries table** — Last 10 entries with time, category, description, brand

**JavaScript polling intervals:**
- `pollState()` — every 1 second (if local camera)
- `pollStats()` — every 5 seconds
- `pollEntries()` — every 5 seconds

**Particle animation:** 25 particles with random position/velocity/alpha, drawn on background canvas with Forest Green color.

**Category colors (JavaScript):**
```javascript
const CAT_COLORS = {
    Plastic: '#87CEEB', Glass: '#40E0D0', Paper: '#D2B48C',
    Organic: '#1E4D2B', Aluminum: '#A9A9A9', Other: '#9370DB',
    Empty: '#8C8C8C'
};
```

### `login.html` — Authentication Page

**Template variables:**
- `error` (str | None) — Error message to display

**Layout:** Centered glassmorphism login card with particle background.

**Form:** POST to `/login` with `username` and `password` fields. Includes link to `/site` for unauthenticated visitors.

### `site.html` — Public Presentation Website

**No authentication required.**

**Sections:**
1. **Navbar** — Fixed, responsive with hamburger menu. Active link highlighting on scroll.
2. **Hero** — Full-viewport animated section with gradient headline, CTA button, floating orb animations.
3. **About** — Two-column grid with stats row: "7 AI Categories", "2 Depth Cameras", "<2s Classification", "24/7 Operation".
4. **Modules** — 6 cards (3x2 grid) for waste categories (Paper, Aluminum, Organic, Glass, Plastic, Other) with category color accent bars.
5. **Statistics** — Large animated total count, category bars loaded from `/api/stats`. Counter animation with ease-out cubic easing.
6. **Map** — Leaflet map centered on Yerevan `[40.1792, 44.4991]` zoom 13. CARTO dark basemap tiles. 8 marker locations with popups (Yerevan landmarks).
7. **Media** — 3 placeholder cards for coming-soon content (3D renders, real-life footage).
8. **Footer** — Grid layout: brand info, quick links, contact (`+374 12 345 678`, `info@smartbin.am`), social icons.

**External CDN dependencies:**
- Leaflet 1.9.4 (JS + CSS)
- CARTO dark basemap tile layer
- Google Fonts Inter

**Intersection Observer:** Triggers fade-in animations at 0.1 threshold, statistics counter animation at 0.2 threshold.

### Static Files

**`style.css`** — Dashboard styles

CSS Variables:
```css
--forest-green: #2D5A42;
--smart-blue: #1A4D6B;
--stone-gray: #8C8C8C;
--success: #4CAF50;
--error: #C62828;
--glass-fill: rgba(45, 90, 66, 0.06);
--glass-border: rgba(45, 90, 66, 0.18);
--radius: 14px;
--radius-lg: 18px;
```

Key animations: `meshShift` (gradient background), `glowRotate` + `glowPulse` (stream container), `rippleOut` (button click), `slideInRight` (history items), `shimmer` (stat bars).

**`site.css`** — Presentation site styles

Responsive breakpoints: 768px (mobile), 1024px (tablet).

Key features: Glassmorphism cards (backdrop blur), full-viewport sections, floating orb animations (`orbFloat` keyframes), CSS-only hamburger menu.

**`site.js`** — Presentation site JavaScript

Functions:
- `loadStats()` — Fetches `/api/stats`, animates counter, generates category bars
- `initMap()` — Creates Leaflet map with custom SVG markers at 8 Yerevan locations
- `animateCounter(el, target, duration)` — Smooth counter with `requestAnimationFrame`
- Intersection Observer for fade-in and statistics animations
- Navbar scroll handler (adds `scrolled` class at 50px)
- Active section highlighting based on scroll position

---

## 9. Database Schema and Operations

### Table: `waste_entries`

| Column | SQLite Type | PostgreSQL Type | Default | Description |
|--------|------------|----------------|---------|-------------|
| `id` | INTEGER PK AUTOINCREMENT | SERIAL PK | auto | Unique row ID |
| `filename` | TEXT | TEXT | — | Path to saved JPEG image |
| `label` | TEXT | TEXT NOT NULL | — | Classification category |
| `description` | TEXT | TEXT | — | Item description from Gemini |
| `brand_product` | TEXT | TEXT | — | Recognized brand/product |
| `location` | TEXT | TEXT | — | Deployment location |
| `weight` | TEXT | TEXT | — | Item weight (placeholder) |
| `timestamp` | TEXT | TIMESTAMPTZ | NOW() | Classification timestamp |
| `simulated_temperature` | REAL | DOUBLE PRECISION | — | Simulated temperature (15-30 C) |
| `simulated_humidity` | REAL | DOUBLE PRECISION | — | Simulated humidity (30-70%) |
| `simulated_vibration` | REAL | DOUBLE PRECISION | — | Simulated vibration (0-0.1) |
| `simulated_air_pollution` | REAL | DOUBLE PRECISION | — | Simulated air quality (5-50 ppm) |
| `simulated_smoke` | REAL | DOUBLE PRECISION | — | Simulated smoke level (0-1) |
| `bin_id` | TEXT | TEXT | `'bin-01'` | Bin device identifier |

### PostgreSQL Indices

| Index | Column | Purpose |
|-------|--------|---------|
| `idx_waste_label` | `label` | Fast category lookups |
| `idx_waste_ts` | `timestamp` | Time-range queries |
| `idx_waste_bin_id` | `bin_id` | Per-bin filtering |

### Connection Management

- **SQLite:** Direct `sqlite3.connect()` per operation. File at `waste_dataset/waste.db`.
- **PostgreSQL:** `psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=5)`. Connections are obtained via `pool.getconn()` and returned via `pool.putconn()`.

### Migration Script: `scripts/migrate_json_to_pg.py`

**Usage:**
```bash
python scripts/migrate_json_to_pg.py --source sqlite    # migrate from SQLite
python scripts/migrate_json_to_pg.py --source json      # migrate from metadata.json
python scripts/migrate_json_to_pg.py --source both      # migrate from both
```

Connects to PostgreSQL using `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` env vars. Creates the table, reads source data, inserts rows with missing sensor fields defaulting to 0.0.

---

## 10. AI Classification Pipeline

### Gemini Prompt

The complete prompt sent to Gemini with every classification request:

```
You are an AI waste classification system.
The camera view shows the inside of a fixed green plastic pipe.

You are given ONE image that contains TWO camera angles side-by-side:
- LEFT half: Camera A
- RIGHT half: Camera B

Note: The camera has a transparent protective glass cover. Ignore it entirely
— do not classify it as "Glass" or any other category. Any reflections or
glare from the glass should also be ignored.

Task:
1) Identify what is inside the pipe, ignoring the pipe itself.
2) The first frame may show an empty pipe — use it as background reference.
3) For subsequent frames, classify only objects inside the pipe.
4) Ignore green pipe/background surfaces. If the TRASH item is
   green/yellow/transparent, DO NOT ignore it.
5) Respond ONLY in valid JSON (no markdown, no extra text).

Output JSON format (exact keys):
{
  "category": "Plastic/Glass/Paper/Organic/Aluminum/Other/Empty",
  "description": "One sentence describing material, color, and shape.",
  "brand_product": "Recognized brand and product name, try also to recognize
   the Armenian brands and product name like Jermuk, Bjni, BOOM and etc.
   or 'Unknown'."
}

Rules:
- If pipe is empty -> category="Empty", description="N/A",
  brand_product="Unknown"
- Use exactly one of these categories: Plastic, Glass, Paper, Organic,
  Aluminum, Other, Empty
- If mixed/unclear -> Other

Do NOT:
- Mention the pipe or background
- Output anything besides valid JSON
- Use markdown/code fences
- Invent new categories
```

### Image Preparation Flow

1. **Capture:** Two camera frames captured independently
2. **Sync:** Frames rejected if timestamps differ by more than `MAX_DT` (0.25s)
3. **Crop:** `crop_sides(frame, CROP_PERCENT)` removes 20% from each side
4. **Resize:** Each frame scaled to `DISPLAY_SIZE` (800x800)
5. **Concatenate:** `cv2.hconcat([frame_a, frame_b])` produces 1600x800 image
6. **Encode:** `cv2.imencode(".jpg", frame, [JPEG_QUALITY])` at quality 85
7. **Base64:** Encoded bytes are base64-encoded for the Gemini API

### API Call with Retry

```python
@retry(
    retry=retry_if_exception(_is_retryable),    # skip retrying 429 errors
    stop=stop_after_attempt(3),                  # max 3 attempts
    wait=wait_exponential(min=1.0, max=8.0),     # 1s, 2s, 4s, 8s backoff
    before_sleep=before_sleep_log(logger, WARNING),
    reraise=True,
)
def _call_gemini(img_bytes: bytes) -> str:
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}},
                {"text": PROMPT},
            ],
        }],
    )
    return resp.text
```

**Quota markers (not retried):** `"429"`, `"RESOURCE_EXHAUSTED"`, `"Quota exceeded"`, `"limit: 0"`

### Circuit Breaker

The circuit breaker prevents hammering a down API:

1. **Closed** (normal): API calls proceed. Each final failure (after all retries) increments `_cb_failures`.
2. **Open:** After 5 consecutive final failures, the circuit opens for 60 seconds. All classify calls are skipped with status "API paused".
3. **Half-open:** After the recovery window, one call is allowed through. Success resets the counter; failure re-opens the circuit.

### Response Processing

1. Strip markdown fences from Gemini response
2. Parse JSON (`_extract_json`)
3. Extract `category`, `description`, `brand_product` with fallback defaults
4. Normalize category: `str.capitalize()`, validate against `VALID_CLASSES`
5. Unknown categories become `"Other"`
6. Update `AppState` with result
7. If not `"Empty"`, call `save_entry()` to persist

---

## 11. Sensor Fusion — OAK-D Native Mode

### Overview

OAK Native mode uses three software-based sensors on a single OAK-D camera to determine bin occupancy before triggering a Gemini API call:

| Sensor | Source | Detection Method |
|--------|--------|-----------------|
| **Presence** | Pixel-diff background model | Mean absolute diff > `MOTION_THRESHOLD` |
| **Motion** | Score spike detection | Score jump > `MOTION_THRESHOLD * MOTION_SPIKE_FACTOR` |
| **Neural Network** | MobileNetSSD on Myriad X VPU | Any detection with confidence > `NN_CONFIDENCE` |

### Vote System

Each sensor casts a binary vote (0 or 1). Classification triggers when:

```
votes = int(presence) + int(motion_spike) + int(nn_count > 0)
if votes >= OAK_VOTES_NEEDED:     # default: 2 out of 2-3 sensors
    trigger classification
```

The number of active sensors is `2 + int(nn_available)`:
- Presence and Motion are always available (software-only)
- NN requires successful MobileNetSSD blob download

### Presence Sensor

Uses `PresenceDetector` (pixel-diff background model):
- Warmup: First `BG_WARMUP_FRAMES` (40) frames build the background
- Detection: `score = mean(|current_gray - background|)`
- Occupied when `score >= MOTION_THRESHOLD` (12.0) for `DETECT_CONFIRM_N` (3) checks

### Motion Spike Sensor

Replaces physical IMU accelerometer with software detection:
- Monitors the delta between consecutive presence scores
- Spike detected when `score_jump > MOTION_THRESHOLD * MOTION_SPIKE_FACTOR` (12.0 * 3.0 = 36.0)
- Spike flag auto-expires after `DROP_FLAG_DURATION` (3.0) seconds

### Neural Network Sensor

MobileNetSSD running on the Myriad X VPU:
- **Model:** `mobilenet-ssd` (Pascal VOC 21 classes)
- **Input:** 300x300 RGB preview from camera
- **Shaves:** 6 (out of available Myriad X cores)
- **Confidence threshold:** 0.5
- **Blob source:** Downloaded via `blobconverter.from_zoo()` at startup
- **Output:** Bounding boxes with class labels and confidence scores

### State Machine

```
Calibrating ─── calibrate() returns True ──> Ready
                                               │
                     votes >= OAK_VOTES_NEEDED │ (for OAK_DETECT_CONFIRM_N ticks)
                                               ▼
                                            Detected
                                               │
                        start_classify() OK    │
                                               ▼
                                           Classifying
                                               │
                        is_classifying = False  │
                                               ▼
                                           Classified
                                               │
                        votes == 0             │ (for OAK_EMPTY_CONFIRM_N ticks)
                                               ▼
                                             Ready
```

Tick interval: `OAK_CHECK_INTERVAL` (0.4s)

### Dual Camera Support

When 2 OAK devices are detected:
- **Camera 1 (primary):** Runs sensor fusion pipeline (presence + motion + NN)
- **Camera 2 (secondary):** Simple RGB pipeline for second viewing angle
- Frames from both cameras are cropped, resized, and concatenated for the Gemini prompt
- Frame sync uses `MAX_DT` (0.25s)

---

## 12. Presence Detection — Auto-Gate Mode

### Overview

Auto-gate mode (`mainauto.py`) uses a simpler single-sensor approach compared to OAK Native mode. The `PresenceDetector` maintains a rolling background model and fires exactly one Gemini API call per item arrival.

### Algorithm

**Phase 1 — Warmup (first 40 frames):**
```
for each gray frame:
    cv2.accumulateWeighted(gray_float, background, 0.15)  # fast learning
    warmup_count += 1
```
No detections are fired during warmup. The background converges quickly.

**Phase 2 — Detection:**
```
score = mean(|gray_float - background|)     # 0 to 255 scale

if score >= MOTION_THRESHOLD (12.0):
    detect_streak += 1
    empty_streak = 0
else:
    empty_streak += 1
    detect_streak = 0
    cv2.accumulateWeighted(gray_float, background, BG_LEARNING_RATE)  # slow drift

is_occupied = detect_streak >= DETECT_CONFIRM_N (3)
is_empty = empty_streak >= EMPTY_CONFIRM_N (6)
```

### PresenceGateStrategy State Machine

```
Calibrating ─── detector.ready ──> Ready (bin empty)
                                      │
                     is_occupied       │ (not previously occupied)
                                      ▼
                                   Detected ─── fire one API call ──> (waiting)
                                      │
                     is_empty          │ (and was occupied)
                                      ▼
                              Bin Cleared ─── accept_as_background ──> Ready
```

**Key invariant:** Exactly one API call per item arrival. The `_item_classified` flag prevents duplicate calls.

---

## 13. Docker Deployment

### Option 1: Server Mode (LAN Dashboard, No Cameras)

**File:** `docker-compose.yml`

**Use case:** Run the web dashboard and database on a laptop or server. No cameras needed. Edge devices POST results here.

```bash
docker compose up -d
```

**Services:**

| Service | Image | Port | Description |
|---------|-------|------|-------------|
| `postgres` | `postgres:16-alpine` | 5432 | PostgreSQL database with health check |
| `app` | Built from `Dockerfile` | 8000 | FastAPI web UI (`CAMERA_MODE=none`) |
| `grafana` | `grafana/grafana:11.0.0` | 3000 | Monitoring dashboards |

**Dockerfile (server):**
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libusb-1.0-0 libpq-dev gcc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "smartwaste.web"]
```

**Volumes:** `pgdata`, `waste_images`, `grafana_data`

### Option 2: Full Stack on Raspberry Pi

**File:** `docker-compose.edge-full.yml`

**Use case:** Run everything on the Pi: cameras, web UI, database, and Grafana.

```bash
docker compose -f docker-compose.edge-full.yml up -d
```

**Differences from server mode:**
- `app` service runs with `privileged: true` and `/dev/bus/usb` volume mount
- `CAMERA_MODE` defaults to `oak-native` (overridable)
- `BIN_ID` and `LOCATION` configurable via env vars

### Option 3: Lightweight Edge (Cameras Only)

**File:** `docker-compose.edge.yml`

**Use case:** Run camera + classifier on the Pi. No web UI. POSTs results to a central server.

```bash
SMARTWASTE_SERVER_URL=http://<server-ip>:8000 \
SMARTWASTE_BIN_ID=bin-01 \
docker compose -f docker-compose.edge.yml up -d
```

**Dockerfile.edge:**
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libusb-1.0-0
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt
COPY . .
CMD ["python", "mainoak.py"]
```

**Key settings:**

| Env Var | Default | Description |
|---------|---------|-------------|
| `SMARTWASTE_EDGE_MODE` | `true` | Enable edge reporting |
| `SMARTWASTE_SERVER_URL` | `http://192.168.1.100:8000` | Central server URL |
| `SMARTWASTE_EDGE_API_KEY` | `password123` | Auth token for server |
| `SMARTWASTE_BIN_ID` | `bin-01` | Unique bin identifier |
| `SMARTWASTE_EDGE_ENTRYPOINT` | `mainoak.py` | Python script to run |

**Features:** `privileged: true`, `/dev/bus/usb` mount, `restart: unless-stopped`

### Service Access Summary

| Service | URL | Credentials |
|---------|-----|-------------|
| Web Dashboard | `http://<host>:8000` | admin / password123 |
| Presentation Site | `http://<host>:8000/site` | No login required |
| Grafana | `http://<host>:3000` | admin / admin |
| PostgreSQL | `<host>:5432` | smartwaste / smartwaste |

---

## 14. Edge Architecture

### Communication Flow

```
Edge Device (Raspberry Pi)               Central Server
┌─────────────────────┐                 ┌──────────────────────┐
│  Camera → Classify   │ ──POST ────── │  /api/report          │
│                      │   /api/report  │  (save entry + image) │
│  Heartbeat Thread    │ ──POST ────── │  /api/heartbeat       │
│  (every 30s)         │   /api/heartbeat│ (update bin registry)│
└─────────────────────┘                 └──────────────────────┘
```

### Authentication

Edge devices authenticate using Bearer tokens:
```
Authorization: Bearer {EDGE_API_KEY}
```
The server validates this against `ADMIN_PASSWORD`. Both sides must share the same secret.

### Classification Report

When `EDGE_MODE=true` and a classification completes, `save_entry()` additionally:
1. Reads the saved JPEG file
2. Calls `report_classification(entry, env, image_bytes)`
3. The edge client base64-encodes the image and POSTs to `/api/report`
4. The server decodes the image, saves it to `DATASET_DIR`, and inserts the database row

### Heartbeat Mechanism

The heartbeat thread (`start_heartbeat_thread()`) runs in a daemon thread:
- Sends POST to `/api/heartbeat` every `HEARTBEAT_INTERVAL` (30) seconds
- Payload: `{bin_id, status: "online", camera_mode, uptime_seconds}`
- Uptime is `time.monotonic() - _start_time` (monotonic clock since module load)

### Bin Registry (Server Side)

The server maintains an in-memory `_bin_registry` dict:
- Updated on each heartbeat
- Bins marked `"offline"` if `(now - last_seen) > 60 seconds`
- The `/api/dashboard` endpoint merges registry data with database stats
- Provides: bin_id, status, last_seen, camera_mode, uptime_seconds

---

## 15. Grafana Monitoring

### Access

- **URL:** `http://<host>:3000`
- **Credentials:** admin / admin (default)
- **Dashboard:** "Smart Waste AI" (UID: `smartwaste-main`)

### Datasource

PostgreSQL provisioned via `grafana/provisioning/datasources/postgres.yml`:
```yaml
datasources:
  - name: SmartWaste PostgreSQL
    type: postgres
    url: postgres:5432
    database: smartwaste
    user: smartwaste
```

### Dashboard Panels

| # | Panel | Type | Size | SQL Query |
|---|-------|------|------|-----------|
| 1 | Total Classifications | stat | 6x4 | `SELECT COUNT(*) AS total FROM waste_entries` |
| 2 | Category Distribution | piechart | 6x8 | `SELECT label AS metric, COUNT(*) AS value FROM waste_entries GROUP BY label ORDER BY value DESC` |
| 3 | Classifications Over Time | timeseries (stacked) | 12x8 | `SELECT date_trunc('hour', timestamp) AS time, label, COUNT(*) FROM waste_entries WHERE $__timeFilter(timestamp) GROUP BY 1, 2 ORDER BY 1` |
| 4 | Recent Classifications | table | 12x8 | `SELECT timestamp, label, description, brand_product, location FROM waste_entries ORDER BY timestamp DESC LIMIT 20` |
| 5 | Simulated Environment Sensors | timeseries | 12x8 | `SELECT timestamp AS time, simulated_temperature, simulated_humidity, simulated_air_pollution FROM waste_entries WHERE $__timeFilter(timestamp) ORDER BY timestamp` |

**Time range:** Last 24 hours (configurable in Grafana UI).

**Color thresholds (Total Classifications):** Green (default), Yellow (>= 100), Red (>= 500).

---

## 16. Presentation Website

### Access

Available at `/site` (e.g., `http://localhost:8000/site`). No authentication required.

### Structure

| Section | Description |
|---------|-------------|
| **Navbar** | Fixed, responsive, hamburger menu on mobile, scroll-aware active links |
| **Hero** | Full-viewport, animated badge with pulse, gradient headline, CTA button, floating orbs |
| **About** | Two-column: text + stats row (7 Categories, 2 Cameras, <2s Speed, 24/7 Uptime) |
| **Modules** | 6 cards (3x2 grid) for Paper, Aluminum, Organic, Glass, Plastic, Other |
| **Statistics** | Live data from `/api/stats`, animated counter, category bars sorted descending |
| **Map** | Leaflet map with 8 Yerevan deployment markers on CARTO dark basemap |
| **Media** | 3 placeholder cards for upcoming video/3D content |
| **Footer** | Brand info, links, contact (+374 12 345 678, info@smartbin.am), social icons |

### Map Locations

8 markers on the Leaflet map centered on Yerevan `[40.1792, 44.4991]`:

| Location | Coordinates | Description |
|----------|-------------|-------------|
| Republic Square | 40.1776, 44.5126 | Central public area |
| Cascade Complex | 40.1895, 44.5158 | Art gallery and park |
| Vernissage Market | 40.1780, 44.5170 | Open-air arts market |
| Northern Avenue | 40.1825, 44.5130 | Pedestrian shopping street |
| Yerevan State University | 40.1860, 44.5220 | University campus |
| Erebuni Museum | 40.1520, 44.5000 | Historical museum |
| Dalma Garden Mall | 40.1650, 44.4850 | Shopping center |
| Tsitsernakaberd Memorial | 40.1850, 44.4900 | Memorial and park |

### External Dependencies

| Dependency | CDN | Purpose |
|------------|-----|---------|
| Leaflet 1.9.4 | unpkg.com | Interactive map |
| CARTO Dark Tiles | basemaps.cartocdn.com | Dark map theme |
| Google Fonts Inter | fonts.googleapis.com | Typography |

---

## 17. Brand Color Palette

### Primary Colors

| Name | Hex | Usage |
|------|-----|-------|
| Forest Green | `#2D5A42` | Primary accent, headings, borders |
| Deep Smart Blue | `#1A4D6B` | Secondary accent, gradients |

### Secondary Colors

| Name | Hex | Usage |
|------|-----|-------|
| Stone Gray | `#8C8C8C` | Muted text, dividers |
| Taupe | `#BDB76B` | Accent highlights |

### Waste Category Colors

| Category | Name | Hex | CSS Variable |
|----------|------|-----|-------------|
| Paper | Warm Cellulose | `#D2B48C` | — |
| Aluminum | Brushed Metallic Gray | `#A9A9A9` | — |
| Organic | Deep Biophilic Green | `#1E4D2B` | — |
| Glass | Translucent Aqua / Crystal Teal | `#40E0D0` | — |
| Plastic | Refined Synthetic Tone | `#87CEEB` | — |
| Other | Dynamic Module Gradient | `#9370DB` to `#1E90FF` | — |
| Empty | Stone Gray | `#8C8C8C` | — |

### Semantic Colors

| Name | Hex | Usage |
|------|-----|-------|
| Success | `#4CAF50` | Auto mode badge, online status |
| Warning | `#FF9800` | Caution states |
| Error | `#C62828` | Error messages, offline status |
| Info | `#2196F3` | Informational highlights |

### Neutral Colors

| Name | Hex | Usage |
|------|-----|-------|
| Pure White | `#FFFFFF` | Primary text on dark backgrounds |
| Off-White | `#F5F5F7` | Body text |
| Light Gray | `#E0E0E0` | Borders, dividers |
| Dark Charcoal | `#333333` | Panel backgrounds |

### CSS Variables (Dashboard)

```css
:root {
    --forest-green: #2D5A42;
    --smart-blue: #1A4D6B;
    --stone-gray: #8C8C8C;
    --success: #4CAF50;
    --error: #C62828;
    --glass-fill: rgba(45, 90, 66, 0.06);
    --glass-fill-h: rgba(45, 90, 66, 0.12);
    --glass-border: rgba(45, 90, 66, 0.18);
    --radius: 14px;
    --radius-lg: 18px;
}
```

### OpenCV BGR Equivalents

For use in `smartwaste/ui.py` (OpenCV uses BGR, not RGB):

| Category | BGR Tuple |
|----------|-----------|
| Plastic | `(235, 206, 135)` |
| Glass | `(208, 224, 64)` |
| Paper | `(140, 180, 210)` |
| Organic | `(43, 77, 30)` |
| Aluminum | `(169, 169, 169)` |
| Other | `(219, 112, 147)` |
| Empty | `(140, 140, 140)` |

---

## 18. Logging System

### Configuration

- **Logger name:** `"smartwaste"`
- **Level:** `INFO`
- **Format:** `[%(asctime)s] %(levelname)s %(message)s`

### Output Destinations

| Destination | Path | Description |
|-------------|------|-------------|
| Console | stderr | Real-time terminal output |
| Log file | `logs/run_YYYYMMDD_HHMMSS.log` | Persistent session log |
| Error file | `logs/last_api_error_YYYYMMDD_HHMMSS.json` | Raw Gemini API error text |

### Runtime ID

Each session generates a unique `RUN_ID` based on the timestamp at module import:
```python
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### Directory

The `logs/` directory is automatically created by `log_setup.py`. The `waste_dataset/` directory is created by `dataset.py`.

---

## 19. Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=smartwaste --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_classifier.py -v
```

### Test Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "--tb=short -q"

[tool.coverage.run]
source = ["smartwaste"]
omit = ["tests/*"]
```

### Hardware Mocking

Tests mock hardware dependencies to run on any machine:
- `depthai` and `picamera2` are replaced with `MagicMock` if not installed
- `GEMINI_API_KEY` is set to `"test-key-for-pytest"` via environment
- `_start_camera_thread` is patched in web tests to prevent camera initialization

### Test File Reference

| File | Module Tested | Tests | Coverage Focus |
|------|--------------|-------|----------------|
| `test_config.py` | `config.py` | 31 | Constant types, ranges, valid categories |
| `test_state.py` | `state.py` | 31 | Thread-safe state, concurrency (30+ threads), mutex behavior |
| `test_classifier.py` | `classifier.py` | 37 | JSON extraction, Gemini mocking, category normalization, error handling |
| `test_presence.py` | `presence.py` | 31 | Warmup, detection streaks, background learning, reset |
| `test_strategies.py` | `strategies.py` | 32 | ManualStrategy and PresenceGateStrategy: triggers, key handlers, state machine |
| `test_camera.py` | `cameraOak.py` | 21 | `crop_sides()` geometry, content preservation, grayscale |
| `test_utils.py` | `utils.py` | 18 | JPEG encoding, launch_classify thread behavior |
| `test_ui.py` | `ui.py` | 17 | draw_overlay on various frame sizes, history rendering |
| `test_dataset.py` | `dataset.py` | 16 | Environment data ranges, save_entry behavior |
| `test_database.py` | `database.py` | 26 | SQLite CRUD, schema validation, unicode handling |
| `test_prompt.py` | `prompt.py` | 14 | Prompt structure, categories, JSON keys, dual camera, Armenian brands |
| `test_web.py` | `web.py` | 9 | FastAPI endpoints: status codes, response formats |

### Notable Concurrency Tests

From `test_state.py`:
- **`test_only_one_start_classify_wins_under_contention`**: 30 threads race to call `start_classify()`. Exactly 1 returns True, 29 return False.
- **`test_concurrent_set_status_no_exception`**: 50 concurrent writers.
- **`test_concurrent_reads_and_writes_no_exception`**: 1 writer + 4 readers, 200 calls each.

---

## 20. Development Tools

### Ruff (Linting and Formatting)

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I"]    # pycodestyle, pyflakes, isort
ignore = ["E501"]                  # line-too-long handled by formatter
```

### Mypy (Type Checking)

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true      # cv2, depthai, google-genai lack stubs
```

### Build System

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"
```

---

## 21. Project File Tree

```
SmartBin/
├── main.py                     # Manual mode entry point (dual OAK cameras)
├── mainauto.py                 # Auto-gate mode entry point (presence detection)
├── mainoak.py                  # OAK-D Native mode entry point (sensor fusion)
├── mainraspberry.py            # Raspberry Pi dual camera mode
├── pyproject.toml              # Project metadata, CLI scripts, tool config
├── requirements.txt            # Production dependencies
├── requirements-edge.txt       # Lightweight edge dependencies
├── requirements-test.txt       # Test dependencies (pytest, pytest-cov)
├── Dockerfile                  # Server/web container (Python 3.11-slim)
├── Dockerfile.edge             # Lightweight edge container
├── docker-compose.yml          # Server mode (LAN dashboard, no cameras)
├── docker-compose.edge.yml     # Lightweight edge (cameras only, POST to server)
├── docker-compose.edge-full.yml # Full stack on Raspberry Pi
├── CLAUDE.md                   # Developer quick-reference
├── documentation.md            # This file
├── .env                        # Environment variables (not committed)
├── .gitignore                  # Git ignore rules
├── .dockerignore               # Docker build exclusions
│
├── smartwaste/                 # Main Python package
│   ├── settings.py             # Pydantic BaseSettings (layered config)
│   ├── config.py               # Runtime constants from settings
│   ├── state.py                # Thread-safe AppState class
│   ├── app.py                  # Strategy ABC + run_loop() for dual OAK
│   ├── strategies.py           # ManualStrategy, PresenceGateStrategy
│   ├── classifier.py           # Gemini API call, retry, circuit breaker
│   ├── prompt.py               # Gemini prompt string
│   ├── presence.py             # Pixel-diff presence detector
│   ├── oak_native.py           # OAK multi-sensor occupancy detector
│   ├── camera.py               # Single OAK camera pipeline
│   ├── cameraOak.py            # Dual OAK pipeline + crop_sides()
│   ├── cameraraspberry.py      # Raspberry Pi picamera2 module
│   ├── database.py             # SQLite/PostgreSQL persistence layer
│   ├── dataset.py              # Save images + DB entries + edge reporting
│   ├── edge_client.py          # HTTP client for edge-to-server comms
│   ├── schemas.py              # Pydantic models (EdgeReport, BinHeartbeat)
│   ├── web.py                  # FastAPI web UI + REST API + MJPEG stream
│   ├── ui.py                   # OpenCV overlay rendering
│   ├── utils.py                # encode_frame(), launch_classify()
│   ├── log_setup.py            # Logging configuration
│   │
│   ├── web_templates/          # Jinja2 HTML templates
│   │   ├── dashboard.html      # Multi-bin overview (authenticated)
│   │   ├── index.html          # Per-bin detail view (live stream, controls)
│   │   ├── login.html          # Authentication page
│   │   └── site.html           # Public presentation website
│   │
│   └── web_static/             # CSS/JS static files
│       ├── style.css           # Dashboard glassmorphism dark theme
│       ├── site.css            # Presentation site styles (responsive)
│       └── site.js             # Stats, Leaflet map, animations
│
├── scripts/
│   └── migrate_json_to_pg.py   # One-time migration to PostgreSQL
│
├── tests/                      # Test suite (pytest)
│   ├── conftest.py             # Shared fixtures and hardware mocks
│   ├── test_camera.py          # crop_sides() geometry tests
│   ├── test_classifier.py      # JSON extraction, Gemini mock, errors
│   ├── test_config.py          # Constant types, ranges, categories
│   ├── test_database.py        # SQLite CRUD operations
│   ├── test_dataset.py         # Environment data, save_entry
│   ├── test_presence.py        # Warmup, detection, background model
│   ├── test_prompt.py          # Prompt structure, categories, brands
│   ├── test_state.py           # Thread safety, concurrency
│   ├── test_strategies.py      # Manual + presence gate triggers
│   ├── test_ui.py              # draw_overlay basic behavior
│   ├── test_utils.py           # JPEG encoding, launch_classify
│   └── test_web.py             # FastAPI endpoint tests
│
├── grafana/                    # Grafana provisioning
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── postgres.yml    # PostgreSQL datasource config
│   │   └── dashboards/
│   │       └── dashboard.yml   # Dashboard provisioning config
│   └── dashboards/
│       └── smartwaste.json     # Dashboard definition (5 panels)
│
├── logs/                       # Runtime logs (auto-created)
│   ├── run_*.log               # Session logs
│   └── last_api_error_*.json   # API error dumps
│
└── waste_dataset/              # Classified images + SQLite DB (auto-created)
    ├── waste.db                # SQLite database
    └── *.jpg                   # Classified waste images
```
