import logging
import os
from datetime import datetime

from .config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"run_{RUN_ID}.log")
ERR_JSON_FILE = os.path.join(LOG_DIR, f"last_api_error_{RUN_ID}.json")


def get_logger() -> logging.Logger:
    logger = logging.getLogger("smartwaste")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
