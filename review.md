# SmartBin Code Review — Production Readiness Assessment

**Date:** 2026-03-20
**Scope:** Full codebase review of `OAK-1W/` — dual-camera waste classification system

---

## Overall Verdict

The project is a solid prototype that proves the core idea works: two OAK cameras, Gemini vision API, real-time classification. The module separation is clean and the presence-detection gate (`mainauto.py`) is a smart architectural choice. But the gap between "working prototype" and "product a corporation would ship" is significant. This review covers every dimension of that gap.

---

## 1. CRITICAL — Fix Before Any Deployment

### 1.1 Thread Safety Bugs

**`state.py:44` — `toggle_auto()` is not thread-safe.** It reads and writes `self.auto_classify` without holding `self._lock`, while `get_display()` reads it under the lock. This is a data race.

**`dataset.py:27` — `_metadata` list is a global mutable shared across threads.** `save_entry()` appends to it and rewrites the JSON file, but it's called from daemon threads (via `classifier.py`). Two concurrent classifications could corrupt `metadata.json`. Protect with a lock or move to a queue-based design.

### 1.2 Fake Sensor Data Ships as Real Data

`dataset.py:30-37` — `_environment_data()` generates random numbers for temperature, humidity, vibration, air pollution, and smoke. These fake values are written to the Excel log, metadata JSON, and SQLite database with no indication that they are simulated. Anyone consuming this data (investors, researchers, partners) will assume it's real sensor output.

**Action:** Either integrate real sensors (BME280, MQ-2, etc.) via I2C/GPIO on the Raspberry Pi, or clearly mark these columns as `simulated_*` in all storage backends and UI. Do not silently ship random data as ground truth.

### 1.3 No API Key Rotation or Secret Management

`classifier.py:22` — The Gemini client is created once at module import time using a raw environment variable. There is no `.env` file loading, no secret manager integration, no key rotation support. If the key leaks, there's no way to rotate without restarting the process.

**Action:** Use `python-dotenv` for local development. For production on Pi, use a systemd `EnvironmentFile` or a hardware secret store. Build the client lazily (not at import time) so the key can be refreshed.

### 1.4 Excel File Corruption Risk

`dataset.py:62-66` — Every single classification rewrites the entire Excel file: read all rows, concat, write all rows. On a Raspberry Pi SD card with 10,000+ entries, this is:
- Slow (full read + write on every insert)
- Fragile (power loss during write = corrupted file)
- A concurrency hazard (two threads writing simultaneously)

**Action:** Drop Excel as a primary storage format. Use SQLite (which you already have) as the source of truth. Generate Excel exports on-demand via a separate script or API endpoint.

---

## 2. HIGH — Architecture & Engineering

### 2.1 No Tests Exist

Zero test files in the entire project. No unit tests, no integration tests, no smoke tests. This makes every code change a deployment gamble.

**What to test first (highest impact):**
- `_extract_json()` — parse edge cases: markdown fences, garbage prefix/suffix, nested JSON, unicode
- `PresenceDetector` — state transitions: warmup -> ready, detect streaks, empty streaks, reset behavior
- `AppState` — concurrent access: start/finish classify from multiple threads
- `crop_sides()` — boundary conditions: 0%, 50%, negative values
- `save_entry()` — file I/O mocking, verify JSON structure, verify DB insert

**Framework:** `pytest` + `pytest-cov`. Target 80%+ coverage on the `smartwaste/` package. Add a `tests/` directory with `conftest.py` for shared fixtures (mock frames, mock state).

### 2.2 No CI/CD Pipeline

No GitHub Actions, no linting, no formatting enforcement, no automated checks of any kind. Every push to `master` is unvalidated.

**Action — create `.github/workflows/ci.yml`:**
```
- ruff check + ruff format --check (linting + formatting)
- mypy --strict smartwaste/ (type checking)
- pytest --cov=smartwaste tests/ (tests with coverage)
```

Also add `pyproject.toml` to define the project properly (see section 2.8).

### 2.3 Duplicated Entry Points

`main.py` and `mainauto.py` share ~60% identical code: camera init, device detection, frame capture loop, error messages, cleanup. When you fix a bug in one, you must remember to fix it in the other.

**Action:** Extract the shared skeleton into a base class or a `run_loop()` function in `smartwaste/app.py` that accepts a strategy/callback for the classification trigger logic. The two entry points become thin wrappers:

```python
# main.py
from smartwaste.app import run_loop
from smartwaste.strategies import ManualStrategy
run_loop(ManualStrategy())

# mainauto.py
from smartwaste.app import run_loop
from smartwaste.strategies import PresenceGateStrategy
run_loop(PresenceGateStrategy())
```

### 2.4 OAK-D Depth Sensor Is Completely Unused

You're using OAK-D cameras — they have stereo depth, IMU, and an on-device neural accelerator (Myriad X). The current code only uses the RGB stream (`CAM_A`). This is like buying a Tesla and only using the radio.

**What depth unlocks:**
- **Volume estimation** — measure how full the bin is, estimate object size
- **Better presence detection** — depth change is far more reliable than pixel-diff (lighting-invariant)
- **Object segmentation** — depth discontinuities separate the object from the pipe background
- **3D bounding boxes** — useful for robotics/sorting integration later

**Action:** Add a `StereoDepth` node to the pipeline in `camera.py`. Use depth for presence detection instead of (or alongside) the pixel-diff approach. Report fill-level as a percentage.

### 2.5 No On-Device Inference (Edge AI)

Every classification requires a round-trip to Google's servers. On a Raspberry Pi in Armenia, that's 200-500ms+ latency, plus internet dependency. The OAK's Myriad X VPU can run MobileNet, YOLO, or custom models at 30+ FPS with zero internet.

**Roadmap:**
1. **Short-term:** Train a simple MobileNetV2 classifier on your own `waste_dataset/` images (you're already collecting labeled data). Convert to OpenVINO blob. Run on-device for instant classification.
2. **Medium-term:** Use Gemini only as a fallback for low-confidence on-device predictions. This cuts API costs by 80%+ and works offline.
3. **Long-term:** Fine-tune on-device model with new data collected in the field. The Gemini labels become your training pipeline.

### 2.6 No Configuration System

All config is in `config.py` as module-level constants. Changing anything (thresholds, model name, camera count) requires editing source code and redeploying.

**Action:** Implement a layered config system:
1. `config.py` — default values (what you have now)
2. `.env` file — per-deployment overrides (API keys, device IDs, location)
3. CLI arguments — runtime overrides (`--model gemini-2.0-flash --auto --threshold 15`)
4. Environment variables — container/systemd overrides

Use `pydantic-settings` or `python-dotenv` + `argparse`. Make every constant in `config.py` overridable without code changes.

### 2.7 No Error Recovery or Retry Logic

`classifier.py:73-86` — If the Gemini API call fails, the error is logged and the status is updated. No retry. No exponential backoff. No circuit breaker. A single transient network error means a missed classification forever.

**Action:**
- Add retry with exponential backoff (3 attempts, 1s/2s/4s) using `tenacity` library
- Add a circuit breaker: after N consecutive failures, stop calling the API for M seconds and show a clear status
- Queue failed classifications for retry when the connection recovers

### 2.8 No Python Packaging

No `pyproject.toml`, no `setup.py`, no package metadata. The project can't be installed with `pip install -e .`, can't declare its Python version requirement, can't declare entry points.

**Action — create `pyproject.toml`:**
```toml
[project]
name = "smartwaste"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "opencv-python",
    "depthai",
    "pandas",
    "google-genai",
    "openpyxl",
]

[project.scripts]
smartwaste = "main:main"
smartwaste-auto = "mainauto:main"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## 3. MEDIUM — Data, Storage & Observability

### 3.1 Triple Storage Backend Is Redundant

You write every classification to JSON, Excel, AND SQLite. Three write paths means three failure modes, three places to keep in sync, and three migration strategies. The JSON file is O(n) on every write (rewrite entire array). The Excel file is O(n) too and even more fragile.

**Action:** Make SQLite the single source of truth. Add a CLI command or script to export to CSV/Excel/JSON on demand. Delete the JSON and Excel write paths from the hot path entirely.

### 3.2 No Data Validation or Schema Enforcement

The Gemini response is parsed with `_extract_json()` which does basic bracket-matching. There's no validation that the `description` field is actually a string, that `brand_product` isn't absurdly long, or that the JSON has no extra unexpected fields.

**Action:** Define a Pydantic model for the API response:
```python
class ClassificationResult(BaseModel):
    category: Literal["Plastic", "Glass", "Paper", "Organic", "Aluminum", "Other", "Empty"]
    description: str = Field(max_length=500)
    brand_product: str = Field(max_length=200)
```
Validate every response through it. This also gives you automatic serialization for free.

### 3.3 No Log Rotation

`log_setup.py` creates a new log file per run, but there's no cleanup. After months of deployment, the `logs/` directory will consume significant SD card space.

**Action:** Use `logging.handlers.RotatingFileHandler` (e.g., 5 MB max, 3 backups). Or add a cron job / systemd timer to clean logs older than 7 days.

### 3.4 No Metrics or Monitoring

There is no way to know, remotely, whether a deployed bin is working. No uptime tracking, no classification success rate, no API latency metrics, no error rate dashboards.

**Action (staged):**
1. **Immediate:** Log structured metrics (classification count, latency, error count) to a local SQLite `metrics` table
2. **Next:** Expose a simple HTTP endpoint (`/health`, `/metrics`) using a lightweight server (Flask/FastAPI)
3. **Later:** Push metrics to a central service (Prometheus, Grafana Cloud, or even a simple webhook)

### 3.5 Location and Device Identity Are Hardcoded

`dataset.py:52` — `"location": "Yerevan"` is hardcoded. There's no device ID, bin ID, or deployment site identifier. When you have 10 bins deployed, you can't tell which one generated which data.

**Action:** Add `DEVICE_ID`, `BIN_ID`, and `LOCATION` to the config system. Auto-generate a unique device ID from the Pi's serial number or MAC address if not explicitly set.

---

## 4. MEDIUM — UX & Robustness

### 4.1 No Headless / Remote Mode

The system requires a display (OpenCV window). On a deployed bin, there's no monitor. You can't run it headless, can't monitor it over SSH without X forwarding.

**Action:**
- Add a `--headless` flag that skips all `cv2.imshow` / `cv2.namedWindow` calls
- Add a lightweight web UI (even just serving the latest frame as JPEG over HTTP) for remote monitoring
- Log status to console/file in headless mode so you can `tail -f` over SSH

### 4.2 No Graceful Single-Camera Degradation

`main.py:34` — If only 1 camera is detected, the system crashes. A real product should degrade gracefully: run with one camera if only one is available, log a warning, and continue classifying.

**Action:** Make the camera count configurable (1 or 2). If 2 are expected but only 1 is found, warn but continue. Adjust the frame concatenation logic to handle single-frame input.

### 4.3 Hardcoded UI Layout

`ui.py` — Pixel coordinates (`(15, 15)`, `(1575, 135)`, etc.) are hardcoded for 1600x800. If the window size changes, the overlay breaks. Font sizes, margins, colors — all magic numbers.

**Action:** Compute overlay positions relative to frame dimensions. Define colors and font sizes as named constants. Consider adding classification history display (last 5 items) to the overlay.

### 4.4 No Startup Self-Test

The system doesn't verify at startup that:
- The Gemini API key is valid (not just present, but actually works)
- The cameras produce valid frames (not just that they're detected)
- The output directories are writable
- The SQLite database is accessible

**Action:** Add a `self_test()` function that runs at startup: make a lightweight API call (or validate the key format), capture one test frame from each camera, write and delete a test file. Fail fast with clear messages.

---

## 5. LOWER PRIORITY — Polish & Scale

### 5.1 No Containerization

No `Dockerfile`, no `docker-compose.yml`. Deployment is manual: SSH into the Pi, pull code, install deps, hope for the best.

**Action:**
```dockerfile
FROM python:3.11-slim
# ... install system deps for OpenCV and depthai
COPY . /app
RUN pip install -e .
CMD ["smartwaste-auto"]
```
Use `docker-compose` to add optional services (metrics dashboard, web UI).

### 5.2 No Structured Prompt Versioning

`prompt.py` contains a single hardcoded prompt string. When you iterate on the prompt (and you will — constantly), there's no versioning, no A/B testing, no way to track which prompt produced which results.

**Action:** Add a `PROMPT_VERSION` string. Log it with every classification. Store it in the database row. This lets you compare accuracy across prompt iterations.

### 5.3 Weight Field Is Always Empty

`dataset.py:53` — `"weight": ""` is always an empty string. The database schema has it, the Excel has it, but nothing populates it.

**Action:** Either integrate a load cell / scale sensor (HX711 + load cell is <$5 and works with RPi GPIO), or remove the field entirely. Empty fields in production data look like bugs.

### 5.4 No Classification Confidence Score

Gemini's response is treated as absolute truth. There's no confidence score, no "I'm not sure" pathway. When the model is uncertain, it still picks a category.

**Action:** Add a `confidence` field to the prompt's expected JSON output. Set a threshold (e.g., 0.7) below which the system classifies as "Other" or flags for human review. This is critical for building trust in the system.

### 5.5 No Data Privacy Controls

Captured images are stored permanently with no retention policy, no anonymization, no way for a user to request deletion. If a person's hand or face appears in a frame, that's PII stored indefinitely.

**Action:** Add a configurable retention period (e.g., delete images older than 30 days). Consider adding face/hand detection and blurring before storage. Document your data handling policy.

---

## 6. Summary — Prioritized Action Plan

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Fix thread safety bugs (1.1) | Small | Prevents data corruption |
| P0 | Remove or label fake sensor data (1.2) | Small | Prevents credibility damage |
| P0 | Add secret management (1.3) | Small | Security baseline |
| P1 | Add tests + CI (2.1, 2.2) | Medium | Foundation for everything else |
| P1 | Deduplicate entry points (2.3) | Medium | Maintainability |
| P1 | Add config system (2.6) | Medium | Deployability |
| P1 | Add retry/backoff (2.7) | Small | Reliability |
| P1 | Consolidate storage to SQLite (3.1) | Medium | Performance + correctness |
| P2 | Use depth sensor (2.4) | Large | Major feature differentiator |
| P2 | On-device inference (2.5) | Large | Offline capability, cost reduction |
| P2 | Add headless mode + web UI (4.1) | Medium | Real-world deployability |
| P2 | Monitoring + metrics (3.4) | Medium | Operational visibility |
| P2 | Pydantic response validation (3.2) | Small | Data quality |
| P2 | pyproject.toml (2.8) | Small | Proper Python project |
| P3 | Single-camera fallback (4.2) | Small | Robustness |
| P3 | Startup self-test (4.4) | Small | Fail-fast |
| P3 | Docker (5.1) | Medium | Deployment |
| P3 | Prompt versioning (5.2) | Small | Experiment tracking |
| P3 | Confidence scores (5.4) | Small | Trust |
| P3 | Data privacy controls (5.5) | Medium | Compliance |

---

## 7. What You're Doing Right

Don't lose these strengths while improving:

- **Clean module separation** — each file has a single responsibility. This is better than most prototypes.
- **Presence-gated API calls** — the `PresenceDetector` is a smart optimization that saves API costs and reduces latency. Most teams wouldn't think of this until they see the bill.
- **Dual-camera approach** — two angles significantly reduce classification ambiguity. This is a genuine competitive advantage.
- **Thread-safe state** — `AppState` with lock-based classify gating shows you understand concurrency. Just fix the two gaps noted above.
- **Armenian brand recognition** — domain-specific prompt tuning for the deployment location. This shows product thinking, not just engineering.
- **Progressive data collection** — every classification builds your training dataset. This is the flywheel that will enable on-device inference later.
