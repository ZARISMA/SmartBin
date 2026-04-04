"""
smartwaste/database.py — persistence layer with SQLite and PostgreSQL backends.

Backend is selected by the SMARTWASTE_DB_BACKEND setting ("sqlite" or "postgresql").
SQLite is the default for local development; PostgreSQL is used in Docker.
"""

import os
import sqlite3
import threading

from .config import DB_BACKEND, DB_FILE, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from .log_setup import get_logger

logger = get_logger()

try:
    import psycopg2
    import psycopg2.pool

    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

# ── SQL schemas ───────────────────────────────────────────────────────────────

_SQLITE_CREATE = """
CREATE TABLE IF NOT EXISTS waste_entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT,
    label         TEXT,
    description   TEXT,
    brand_product TEXT,
    location      TEXT,
    weight        TEXT,
    timestamp     TEXT,
    simulated_temperature   REAL,
    simulated_humidity      REAL,
    simulated_vibration     REAL,
    simulated_air_pollution REAL,
    simulated_smoke         REAL
);
"""

_PG_CREATE = """
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

_INSERT_COLS = (
    "filename",
    "label",
    "description",
    "brand_product",
    "location",
    "weight",
    "timestamp",
    "simulated_temperature",
    "simulated_humidity",
    "simulated_vibration",
    "simulated_air_pollution",
    "simulated_smoke",
)

_SQLITE_INSERT = (
    "INSERT INTO waste_entries"
    f" ({', '.join(_INSERT_COLS)})"
    f" VALUES ({', '.join(':' + c for c in _INSERT_COLS)});"
)

_PG_INSERT = (
    "INSERT INTO waste_entries"
    f" ({', '.join(_INSERT_COLS)})"
    f" VALUES ({', '.join('%(' + c + ')s' for c in _INSERT_COLS)});"
)

# ── State ─────────────────────────────────────────────────────────────────────

_init_lock = threading.Lock()
_initialized = False
_pg_pool: "psycopg2.pool.ThreadedConnectionPool | None" = None


def _use_pg() -> bool:
    return DB_BACKEND == "postgresql" and _PG_AVAILABLE


def _get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    return _pg_pool


def _ensure_init():
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        init_db()
        _initialized = True


# ── Public API ────────────────────────────────────────────────────────────────


def init_db() -> None:
    """Create the database table if it doesn't exist yet."""
    global _initialized
    if _use_pg():
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(_PG_CREATE)
            logger.info(
                "PostgreSQL database ready: %s@%s:%s/%s", DB_USER, DB_HOST, DB_PORT, DB_NAME
            )
        finally:
            pool.putconn(conn)
    else:
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(_SQLITE_CREATE)
        logger.info("SQLite database ready: %s", DB_FILE)
    _initialized = True


def insert_entry(entry: dict, env: dict) -> None:
    """Insert one classification row.

    Args:
        entry: dict with keys filename, label, description, brand_product,
               location, weight, timestamp
        env:   dict with keys simulated_temperature, simulated_humidity,
               simulated_vibration, simulated_air_pollution, simulated_smoke
    """
    _ensure_init()
    row = {**entry, **env}
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(_PG_INSERT, row)
                logger.info(
                    "DB entry saved (pg): label=%s ts=%s",
                    entry.get("label"),
                    entry.get("timestamp"),
                )
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute(_SQLITE_INSERT, row)
            logger.info(
                "DB entry saved: label=%s ts=%s", entry.get("label"), entry.get("timestamp")
            )
    except Exception as e:
        logger.error("DB insert failed: %s", e)


def get_entries(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return recent classification entries, newest first."""
    _ensure_init()
    cols = list(_INSERT_COLS) + ["id"]
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT "
                        + ", ".join(cols)
                        + " FROM waste_entries ORDER BY id DESC LIMIT %s OFFSET %s",
                        (limit, offset),
                    )
                    rows = cur.fetchall()
                return [dict(zip(cols, r)) for r in rows]
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT "
                    + ", ".join(cols)
                    + " FROM waste_entries ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []


def get_label_counts() -> dict[str, int]:
    """Return {label: count} for all categories."""
    _ensure_init()
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT label, COUNT(*) FROM waste_entries GROUP BY label")
                    return dict(cur.fetchall())
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                cur = conn.execute("SELECT label, COUNT(*) FROM waste_entries GROUP BY label")
                return dict(cur.fetchall())
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return {}


def get_entry_count() -> int:
    """Return total number of classification entries."""
    _ensure_init()
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM waste_entries")
                    return int(cur.fetchone()[0])
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                return int(conn.execute("SELECT COUNT(*) FROM waste_entries").fetchone()[0])
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return 0
