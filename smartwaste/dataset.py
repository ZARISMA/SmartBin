import os
import random
from datetime import datetime

import cv2

from .config import BIN_ID, DATASET_DIR, EDGE_MODE, LOCATION, SERVER_URL
from .database import insert_entry
from .log_setup import get_logger

logger = get_logger()
os.makedirs(DATASET_DIR, exist_ok=True)


def _environment_data() -> dict:
    return {
        "simulated_temperature": round(random.uniform(15, 30), 2),
        "simulated_humidity": round(random.uniform(30, 70), 2),
        "simulated_vibration": round(random.uniform(0, 0.1), 3),
        "simulated_air_pollution": round(random.uniform(5, 50), 2),
        "simulated_smoke": round(random.uniform(0, 1), 2),
    }


def save_entry(label: str, img, description: str, brand_product: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{label}_{ts}.jpg"
    filepath = os.path.join(DATASET_DIR, filename)
    cv2.imwrite(filepath, img)

    entry = {
        "filename": filepath,
        "label": label,
        "description": description,
        "brand_product": brand_product,
        "location": LOCATION,
        "weight": "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bin_id": BIN_ID,
    }

    logger.info("Saved dataset entry: %s", filename)
    env = _environment_data()
    insert_entry(entry, env)

    # Report to central server when running as edge device
    if EDGE_MODE and SERVER_URL:
        try:
            from .edge_client import report_classification

            with open(filepath, "rb") as f:
                img_bytes = f.read()
            report_classification(entry, env, image_bytes=img_bytes)
        except Exception as exc:
            logger.warning("Edge report failed: %s", exc)
