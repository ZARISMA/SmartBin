"""
smartwaste/database.py — SQLite persistence layer.

Table schema mirrors the JSON/Excel entries plus the 5 simulated
environment sensor columns, so all three storage backends stay in sync.
"""

import os
import sqlite3

from .config import DB_FILE
from .log_setup import get_logger

logger = get_logger()

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS waste_entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT,
    label         TEXT,
    description   TEXT,
    brand_product TEXT,
    location      TEXT,
    weight        TEXT,
    timestamp     TEXT,
    temperature   REAL,
    humidity      REAL,
    vibration     REAL,
    air_pollution REAL,
    smoke         REAL
);
"""

_INSERT_ROW = """
INSERT INTO waste_entries
    (filename, label, description, brand_product, location, weight, timestamp,
     temperature, humidity, vibration, air_pollution, smoke)
VALUES
    (:filename, :label, :description, :brand_product, :location, :weight, :timestamp,
     :temperature, :humidity, :vibration, :air_pollution, :smoke);
"""


def init_db() -> None:
    """Create the database file and table if they don't exist yet."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(_CREATE_TABLE)
    logger.info("SQLite database ready: %s", DB_FILE)


def insert_entry(entry: dict, env: dict) -> None:
    """
    Insert one classification row.

    Args:
        entry: dict with keys filename, label, description, brand_product,
               location, weight, timestamp  (same dict built in dataset.save_entry)
        env:   dict with keys temperature, humidity, vibration,
               air_pollution, smoke  (from dataset._environment_data)
    """
    row = {**entry, **env}
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(_INSERT_ROW, row)
        logger.info("DB entry saved: label=%s ts=%s", entry.get("label"), entry.get("timestamp"))
    except Exception as e:
        logger.error("DB insert failed: %s", e)


# Initialise on first import
init_db()
