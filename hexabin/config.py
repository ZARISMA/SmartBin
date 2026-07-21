"""
hexabin/config.py — all runtime constants, sourced from the settings layer.

Existing code continues to import from here unchanged::

    from .config import MODEL_NAME

To override without editing source, set env vars (HEXABIN_* prefix) or add
entries to a .env file in the project root.  See hexabin/settings.py.
"""

import json
import logging
import os

from .settings import settings

# ── File / directory paths (not overridable; derived from install location) ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET_DIR = os.path.join(BASE_DIR, "waste_dataset")
DB_FILE = os.path.join(DATASET_DIR, "waste.db")
# Per-camera geometry (rotate/flip/crop) saved by the dashboard; applied-state
# that the capture loop reloads on startup so an edit survives a restart.
CAMERA_CONFIG_FILE = os.path.join(DATASET_DIR, "camera_config.json")

# ── Database ──────────────────────────────────────────────────────────────────
DB_BACKEND = settings.db_backend
DB_HOST = settings.db_host
DB_PORT = settings.db_port
DB_NAME = settings.db_name
DB_USER = settings.db_user
DB_PASSWORD = settings.db_password

# ── Web UI ────────────────────────────────────────────────────────────────────
WEB_HOST = settings.web_host
WEB_PORT = settings.web_port
CAMERA_MODE = settings.camera_mode

# ── Authentication ────────────────────────────────────────────────────────────
ADMIN_USERNAME = settings.admin_username
ADMIN_PASSWORD = settings.admin_password
SECRET_KEY = settings.secret_key

# ── Bin identity ──────────────────────────────────────────────────────────────
BIN_ID = settings.bin_id

# ── Edge mode ─────────────────────────────────────────────────────────────────
EDGE_MODE = settings.edge_mode
SERVER_URL = settings.server_url
EDGE_API_KEY = settings.edge_api_key
HEARTBEAT_INTERVAL = settings.heartbeat_interval

# ── Classification ─────────────────────────────────────────────────────────────
MODEL_NAME = settings.model_name
VALID_CLASSES = ["Plastic", "Glass", "Paper", "Organic", "Aluminum", "Other", "Empty"]
# Categories that count toward the recycling diversion rate ("Other"/"Empty" don't).
RECYCLABLE_CLASSES = ["Plastic", "Glass", "Paper", "Organic", "Aluminum"]
LOCATION = settings.location

# ── LLM backend ────────────────────────────────────────────────────────────────
LLM_BACKEND = settings.llm_backend
LMSTUDIO_URL = settings.lmstudio_url
LMSTUDIO_MODEL = settings.lmstudio_model
LMSTUDIO_TIMEOUT = settings.lmstudio_timeout
LMSTUDIO_MAX_TOKENS = settings.lmstudio_max_tokens
CONFIDENCE_THRESHOLD = settings.confidence_threshold
LLM_MAX_CONCURRENCY = settings.llm_max_concurrency
LLM_QUEUE_TIMEOUT = settings.llm_queue_timeout

# ── Edge classification ────────────────────────────────────────────────────────
CLASSIFY_MODE = settings.classify_mode
CLASSIFY_TIMEOUT = settings.classify_timeout

# ── Ingest limits ──────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024

# ── Actuation ──────────────────────────────────────────────────────────────────
ACTUATOR = settings.actuator

# Which physical bin module opens for each category. "Empty" never opens one.
DEFAULT_MODULE_MAP = {
    "Plastic": 1,
    "Glass": 2,
    "Paper": 3,
    "Organic": 4,
    "Aluminum": 5,
    "Other": 6,
}


def _parse_module_map(raw: str) -> dict[str, int]:
    """Parse the HEXABIN_MODULE_MAP JSON override; fall back to the default."""
    if not raw.strip():
        return dict(DEFAULT_MODULE_MAP)
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("module map must be a JSON object")
        parsed = {}
        for key, value in data.items():
            if key not in VALID_CLASSES or key == "Empty":
                continue
            parsed[str(key)] = int(value)
        if not parsed:
            raise ValueError("module map has no valid categories")
        return parsed
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Invalid HEXABIN_MODULE_MAP %r (%s) — using default mapping.", raw, exc
        )
        return dict(DEFAULT_MODULE_MAP)


MODULE_MAP = _parse_module_map(settings.module_map)

# ── Camera / display ───────────────────────────────────────────────────────────
WINDOW = "Smart Waste AI (Dual OAK)"
DISPLAY_SIZE = (800, 800)
JPEG_QUALITY = settings.jpeg_quality
AUTO_INTERVAL = settings.auto_interval
CROP_PERCENT = settings.crop_percent
MAX_DT = settings.max_dt

# ── Auto-gate mode (mainauto.py) ───────────────────────────────────────────────
MOTION_THRESHOLD = settings.motion_threshold
DETECT_CONFIRM_N = settings.detect_confirm_n
EMPTY_CONFIRM_N = settings.empty_confirm_n
BG_LEARNING_RATE = settings.bg_learning_rate
BG_WARMUP_FRAMES = settings.bg_warmup_frames
CHECK_INTERVAL = settings.check_interval

# ── OAK-D Native mode (mainoak.py) ────────────────────────────────────────────
OAK_WINDOW = "Smart Waste AI (OAK Native)"
OAK_DISPLAY_W = settings.oak_display_w
OAK_DISPLAY_H = settings.oak_display_h

DEPTH_ROI_FRACTION = settings.depth_roi_fraction
DEPTH_CHANGE_THRESHOLD = settings.depth_change_threshold
OAK_CALIB_FRAMES = settings.oak_calib_frames

IMU_SAMPLE_RATE_HZ = settings.imu_sample_rate_hz
IMU_SHOCK_THRESHOLD = settings.imu_shock_threshold
IMU_BASELINE_SAMPLES = settings.imu_baseline_samples
DROP_FLAG_DURATION = settings.drop_flag_duration

NN_MODEL_NAME = settings.nn_model_name
NN_SHAVES = settings.nn_shaves
NN_CONFIDENCE = settings.nn_confidence

OAK_VOTES_NEEDED = settings.oak_votes_needed
OAK_DETECT_CONFIRM_N = settings.oak_detect_confirm_n
OAK_EMPTY_CONFIRM_N = settings.oak_empty_confirm_n
OAK_CHECK_INTERVAL = settings.oak_check_interval
MOTION_SPIKE_FACTOR = settings.motion_spike_factor
