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
