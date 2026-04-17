"""
smartwaste/config.py — all runtime constants, sourced from the settings layer.

Existing code continues to import from here unchanged::

    from .config import MODEL_NAME

To override without editing source, set env vars (SMARTWASTE_* prefix) or add
entries to a .env file in the project root.  See smartwaste/settings.py.
"""

import os

from .settings import settings

# ── File / directory paths (not overridable; derived from install location) ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET_DIR = os.path.join(BASE_DIR, "waste_dataset")
DB_FILE = os.path.join(DATASET_DIR, "waste.db")

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
LOCATION = settings.location

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
