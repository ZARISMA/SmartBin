"""
smartwaste/database.py — persistence layer with SQLite and PostgreSQL backends.

Backend is selected by the SMARTWASTE_DB_BACKEND setting ("sqlite" or "postgresql").
SQLite is the default for local development; PostgreSQL is used in Docker.
"""

import os
import sqlite3
import threading

from .config import BIN_ID, DB_BACKEND, DB_FILE, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
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
    simulated_smoke         REAL,
    bin_id        TEXT DEFAULT 'bin-01'
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
    simulated_smoke         DOUBLE PRECISION,
    bin_id                  TEXT DEFAULT 'bin-01'
);
CREATE INDEX IF NOT EXISTS idx_waste_label ON waste_entries(label);
CREATE INDEX IF NOT EXISTS idx_waste_ts ON waste_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_waste_bin_id ON waste_entries(bin_id);
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
    "bin_id",
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


def _migrate_add_bin_id() -> None:
    """Add bin_id column to existing tables that lack it."""
    if _use_pg():
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "ALTER TABLE waste_entries ADD COLUMN IF NOT EXISTS bin_id TEXT DEFAULT 'bin-01'"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_waste_bin_id ON waste_entries(bin_id)"
                    )
        finally:
            pool.putconn(conn)
    else:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("ALTER TABLE waste_entries ADD COLUMN bin_id TEXT DEFAULT 'bin-01'")
        except sqlite3.OperationalError:
            pass  # column already exists


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
    _migrate_add_bin_id()
    _initialized = True


def insert_entry(entry: dict, env: dict) -> int | None:
    """Insert one classification row and return the new row ID.

    Args:
        entry: dict with keys filename, label, description, brand_product,
               location, weight, timestamp, bin_id
        env:   dict with keys simulated_temperature, simulated_humidity,
               simulated_vibration, simulated_air_pollution, simulated_smoke
    """
    _ensure_init()
    row = {**entry, **env}
    row.setdefault("bin_id", BIN_ID)
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(_PG_INSERT.rstrip(";") + " RETURNING id;", row)
                        row_id: int | None = cur.fetchone()[0]
                logger.info(
                    "DB entry saved (pg): id=%s label=%s ts=%s",
                    row_id,
                    entry.get("label"),
                    entry.get("timestamp"),
                )
                return row_id
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                cur = conn.execute(_SQLITE_INSERT, row)
                row_id = cur.lastrowid
            logger.info(
                "DB entry saved: id=%s label=%s ts=%s",
                row_id,
                entry.get("label"),
                entry.get("timestamp"),
            )
            return row_id
    except Exception as e:
        logger.error("DB insert failed: %s", e)
        return None


def get_entries(limit: int = 100, offset: int = 0, bin_id: str | None = None) -> list[dict]:
    """Return recent classification entries, newest first, optionally filtered by bin."""
    _ensure_init()
    cols = list(_INSERT_COLS) + ["id"]
    where = ""
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                params: list = []
                if bin_id:
                    where = " WHERE bin_id = %s"
                    params = [bin_id, limit, offset]
                else:
                    params = [limit, offset]
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT "
                        + ", ".join(cols)
                        + f" FROM waste_entries{where} ORDER BY id DESC LIMIT %s OFFSET %s",
                        params,
                    )
                    rows = cur.fetchall()
                return [dict(zip(cols, r)) for r in rows]
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                conn.row_factory = sqlite3.Row
                sql_params: tuple
                if bin_id:
                    where = " WHERE bin_id = ?"
                    sql_params = (bin_id, limit, offset)
                else:
                    sql_params = (limit, offset)
                cur = conn.execute(
                    "SELECT "
                    + ", ".join(cols)
                    + f" FROM waste_entries{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                    sql_params,
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []


def get_label_counts(bin_id: str | None = None) -> dict[str, int]:
    """Return {label: count} for all categories, optionally filtered by bin."""
    _ensure_init()
    where = ""
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                params: list = []
                if bin_id:
                    where = " WHERE bin_id = %s"
                    params = [bin_id]
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT label, COUNT(*) FROM waste_entries{where} GROUP BY label",
                        params,
                    )
                    return dict(cur.fetchall())
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                sql_params: tuple
                if bin_id:
                    where = " WHERE bin_id = ?"
                    sql_params = (bin_id,)
                else:
                    sql_params = ()
                cur = conn.execute(
                    f"SELECT label, COUNT(*) FROM waste_entries{where} GROUP BY label",
                    sql_params,
                )
                return dict(cur.fetchall())
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return {}


def get_entry_count(bin_id: str | None = None) -> int:
    """Return total number of classification entries, optionally filtered by bin."""
    _ensure_init()
    where = ""
    params: tuple | dict = ()
    if bin_id:
        if _use_pg():
            where = " WHERE bin_id = %(bin_id)s"
            params = {"bin_id": bin_id}
        else:
            where = " WHERE bin_id = :bin_id"
            params = {"bin_id": bin_id}
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM waste_entries{where}", params)
                    return int(cur.fetchone()[0])
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                return int(
                    conn.execute(f"SELECT COUNT(*) FROM waste_entries{where}", params).fetchone()[0]
                )
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return 0


def get_entries_by_bin(bin_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
    """Return recent entries for a specific bin."""
    return get_entries(limit=limit, offset=offset, bin_id=bin_id)


def get_label_counts_by_bin(bin_id: str) -> dict[str, int]:
    """Return {label: count} for a specific bin."""
    return get_label_counts(bin_id=bin_id)


def get_active_bins() -> list[dict]:
    """Return distinct bin_ids with their last timestamp and entry count."""
    _ensure_init()
    sql = (
        "SELECT bin_id, COUNT(*) AS total, MAX(timestamp) AS last_ts "
        "FROM waste_entries WHERE bin_id IS NOT NULL "
        "GROUP BY bin_id ORDER BY last_ts DESC"
    )
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return [
                        {"bin_id": r[0], "total": r[1], "last_timestamp": str(r[2]) if r[2] else ""}
                        for r in cur.fetchall()
                    ]
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                cur = conn.execute(sql)
                return [
                    {"bin_id": r[0], "total": r[1], "last_timestamp": r[2] or ""}
                    for r in cur.fetchall()
                ]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []
