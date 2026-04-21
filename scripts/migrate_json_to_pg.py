#!/usr/bin/env python3
"""
One-time migration: metadata.json and/or waste.db (SQLite) -> PostgreSQL.

Usage:
    # Ensure SMARTWASTE_DB_* env vars are set (or .env exists), then:
    python scripts/migrate_json_to_pg.py

    # Or specify source explicitly:
    python scripts/migrate_json_to_pg.py --source sqlite
    python scripts/migrate_json_to_pg.py --source json
    python scripts/migrate_json_to_pg.py --source both

SQLite source is preferred (has all 13 columns including sensor data).
JSON source fills sensor fields with 0.0 (JSON never stored them).
"""

import argparse
import json
import os
import sqlite3
import sys

# Add project root to path so smartwaste can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2

from smartwaste.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "waste_dataset"
)
SQLITE_PATH = os.path.join(DATASET_DIR, "waste.db")
JSON_PATH = os.path.join(DATASET_DIR, "metadata.json")

INSERT_SQL = """
INSERT INTO waste_entries
    (filename, label, description, brand_product, location, weight, timestamp,
     simulated_temperature, simulated_humidity, simulated_vibration,
     simulated_air_pollution, simulated_smoke)
VALUES
    (%(filename)s, %(label)s, %(description)s, %(brand_product)s,
     %(location)s, %(weight)s, %(timestamp)s,
     %(simulated_temperature)s, %(simulated_humidity)s, %(simulated_vibration)s,
     %(simulated_air_pollution)s, %(simulated_smoke)s);
"""

CREATE_TABLE = """
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
    simulated_smoke         DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_waste_label ON waste_entries(label);
CREATE INDEX IF NOT EXISTS idx_waste_ts ON waste_entries(timestamp);
"""


def get_pg_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def migrate_sqlite(conn):
    if not os.path.exists(SQLITE_PATH):
        print(f"SQLite file not found: {SQLITE_PATH}")
        return 0

    src = sqlite3.connect(SQLITE_PATH)
    src.row_factory = sqlite3.Row
    rows = src.execute("SELECT * FROM waste_entries").fetchall()
    src.close()

    count = 0
    with conn.cursor() as cur:
        for row in rows:
            d = dict(row)
            d.pop("id", None)
            cur.execute(INSERT_SQL, d)
            count += 1
    conn.commit()
    return count


def migrate_json(conn):
    if not os.path.exists(JSON_PATH):
        print(f"JSON file not found: {JSON_PATH}")
        return 0

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    count = 0
    with conn.cursor() as cur:
        for entry in entries:
            d = {
                "filename": entry.get("filename", ""),
                "label": entry.get("label", "Other"),
                "description": entry.get("description", ""),
                "brand_product": entry.get("brand_product", ""),
                "location": entry.get("location", ""),
                "weight": entry.get("weight", ""),
                "timestamp": entry.get("timestamp", ""),
                "simulated_temperature": 0.0,
                "simulated_humidity": 0.0,
                "simulated_vibration": 0.0,
                "simulated_air_pollution": 0.0,
                "simulated_smoke": 0.0,
            }
            cur.execute(INSERT_SQL, d)
            count += 1
    conn.commit()
    return count


def main():
    parser = argparse.ArgumentParser(description="Migrate data to PostgreSQL")
    parser.add_argument(
        "--source",
        choices=["sqlite", "json", "both"],
        default="sqlite",
        help="Data source to migrate from (default: sqlite)",
    )
    args = parser.parse_args()

    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE)
        conn.commit()

        total = 0
        if args.source in ("sqlite", "both"):
            n = migrate_sqlite(conn)
            print(f"Migrated {n} entries from SQLite")
            total += n
        if args.source in ("json", "both"):
            n = migrate_json(conn)
            print(f"Migrated {n} entries from metadata.json")
            total += n

        print(f"Total: {total} entries migrated to PostgreSQL")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
