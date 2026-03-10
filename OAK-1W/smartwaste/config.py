import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR     = os.path.join(BASE_DIR, "logs")
DATASET_DIR = os.path.join(BASE_DIR, "waste_dataset")
META_FILE   = os.path.join(DATASET_DIR, "metadata.json")
EXCEL_FILE  = os.path.join(DATASET_DIR, "waste_log.xlsx")

MODEL_NAME    = "gemini-3-flash-preview"
VALID_CLASSES = ["Plastic", "Glass", "Paper", "Organic", "Aluminum", "Other", "Empty"]

WINDOW       = "Smart Waste AI (Dual OAK)"
DISPLAY_SIZE = (800, 800)
JPEG_QUALITY = 85
AUTO_INTERVAL = 6      # seconds between auto-classifications
CROP_PERCENT  = 0.20   # fraction to crop from each side before analysis
MAX_DT        = 0.25   # max timestamp delta (s) between camera frames

# ── Auto-gate mode (mainauto.py) ───────────────────────────────────────────────
MOTION_THRESHOLD = 12.0   # mean-abs pixel diff (0-255) to consider bin non-empty
DETECT_CONFIRM_N = 3      # consecutive above-threshold checks before triggering API
EMPTY_CONFIRM_N  = 6      # consecutive below-threshold checks before resetting to IDLE
BG_LEARNING_RATE = 0.03   # how fast background adapts when bin is empty
BG_WARMUP_FRAMES = 40     # local checks during warmup before detection is active
CHECK_INTERVAL   = 0.5    # seconds between local presence checks
