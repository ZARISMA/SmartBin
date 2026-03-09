import json
import os
import random
from datetime import datetime

import cv2
import pandas as pd

from .config import DATASET_DIR, EXCEL_FILE, META_FILE
from .log_setup import get_logger

logger = get_logger()
os.makedirs(DATASET_DIR, exist_ok=True)


def load_metadata() -> list:
    if not os.path.exists(META_FILE):
        return []
    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


_metadata = load_metadata()


def _environment_data() -> dict:
    return {
        "temperature":   round(random.uniform(15, 30),  2),
        "humidity":      round(random.uniform(30, 70),  2),
        "vibration":     round(random.uniform(0, 0.1),  3),
        "air_pollution": round(random.uniform(5, 50),   2),
        "smoke":         round(random.uniform(0, 1),    2),
    }


def save_entry(label: str, img, description: str, brand_product: str) -> None:
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{label}_{ts}.jpg"
    filepath = os.path.join(DATASET_DIR, filename)
    cv2.imwrite(filepath, img)

    entry = {
        "filename":      filepath,
        "label":         label,
        "description":   description,
        "brand_product": brand_product,
        "location":      "Yerevan",
        "weight":        "",
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _metadata.append(entry)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(_metadata, f, ensure_ascii=False, indent=4)

    df = pd.DataFrame([{**entry, **_environment_data()}])
    try:
        if not os.path.exists(EXCEL_FILE):
            df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
        else:
            old = pd.read_excel(EXCEL_FILE, engine="openpyxl")
            pd.concat([old, df], ignore_index=True).to_excel(EXCEL_FILE, index=False, engine="openpyxl")
        logger.info("Saved dataset entry: %s", filename)
    except PermissionError:
        logger.warning("Excel file is open — close it and retry.")
    except Exception as e:
        logger.exception("Excel write failed: %s", e)
