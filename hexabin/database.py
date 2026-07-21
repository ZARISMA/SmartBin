"""
hexabin/database.py — persistence layer with SQLite and PostgreSQL backends.

Backend is selected by the HEXABIN_DB_BACKEND setting ("sqlite" or "postgresql").
SQLite is the default for local development; PostgreSQL is used in Docker.
"""

import os
import sqlite3
import threading
from datetime import datetime

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
    bin_id        TEXT DEFAULT 'bin-01',
    confidence    REAL,
    llm_backend   TEXT DEFAULT ''
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
    bin_id                  TEXT DEFAULT 'bin-01',
    confidence              DOUBLE PRECISION,
    llm_backend             TEXT DEFAULT ''
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
    "bin_id",
    "confidence",
    "llm_backend",
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

# ── Users (dashboard accounts) ─────────────────────────────────────────────────

_SQLITE_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT
);
"""

_PG_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
"""

# ── Camera geometry (per-bin, per-camera desired-state) ────────────────────────

_SQLITE_CREATE_CAMERA = """
CREATE TABLE IF NOT EXISTS camera_configs (
    bin_id     TEXT NOT NULL,
    cam_index  INTEGER NOT NULL,
    rotation   INTEGER DEFAULT 0,
    flip_h     INTEGER DEFAULT 0,
    flip_v     INTEGER DEFAULT 0,
    crop_x0    REAL DEFAULT 0.0,
    crop_y0    REAL DEFAULT 0.0,
    crop_x1    REAL DEFAULT 1.0,
    crop_y1    REAL DEFAULT 1.0,
    updated_at TEXT,
    PRIMARY KEY (bin_id, cam_index)
);
"""

_PG_CREATE_CAMERA = """
CREATE TABLE IF NOT EXISTS camera_configs (
    bin_id     TEXT NOT NULL,
    cam_index  INTEGER NOT NULL,
    rotation   INTEGER DEFAULT 0,
    flip_h     BOOLEAN DEFAULT FALSE,
    flip_v     BOOLEAN DEFAULT FALSE,
    crop_x0    DOUBLE PRECISION DEFAULT 0.0,
    crop_y0    DOUBLE PRECISION DEFAULT 0.0,
    crop_x1    DOUBLE PRECISION DEFAULT 1.0,
    crop_y1    DOUBLE PRECISION DEFAULT 1.0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (bin_id, cam_index)
);
"""

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


def _filters_sql(
    pg: bool,
    *,
    bin_id: str | None = None,
    label: str | None = None,
    q: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> tuple[str, dict]:
    """Build a ' WHERE ...' clause + named params for the active backend.

    pg=True uses %(name)s placeholders, otherwise :name.  since/until are
    'YYYY-MM-DD HH:MM:SS' strings; the range is half-open [since, until) —
    lexicographic comparison on the SQLite TEXT column matches chronological
    order because the stored format is fixed-width.
    """

    def ph(name: str) -> str:
        return f"%({name})s" if pg else f":{name}"

    clauses: list[str] = []
    params: dict = {}
    if bin_id:
        clauses.append(f"bin_id = {ph('bin_id')}")
        params["bin_id"] = bin_id
    if label:
        clauses.append(f"label = {ph('label')}")
        params["label"] = label
    if q:
        like = "ILIKE" if pg else "LIKE"
        clauses.append(f"(description {like} {ph('q')} OR brand_product {like} {ph('q')})")
        params["q"] = f"%{q}%"
    if since:
        clauses.append(f"timestamp >= {ph('since')}")
        params["since"] = since
    if until:
        clauses.append(f"timestamp < {ph('until')}")
        params["until"] = until
    if not clauses:
        return "", {}
    return " WHERE " + " AND ".join(clauses), params


def _fetch_rows(sql: str, params: dict | tuple = ()) -> list[tuple]:
    """Run a SELECT on the active backend and return all rows (raises on error)."""
    if _use_pg():
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows: list[tuple] = cur.fetchall()
                return rows
        finally:
            pool.putconn(conn)
    with sqlite3.connect(DB_FILE) as conn:
        return conn.execute(sql, params).fetchall()


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


def _migrate_add_llm_fields() -> None:
    """Add confidence / llm_backend columns to existing tables that lack them."""
    if _use_pg():
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "ALTER TABLE waste_entries"
                        " ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION"
                    )
                    cur.execute(
                        "ALTER TABLE waste_entries"
                        " ADD COLUMN IF NOT EXISTS llm_backend TEXT DEFAULT ''"
                    )
        finally:
            pool.putconn(conn)
    else:
        # One ALTER per column: an existing `confidence` must not skip `llm_backend`.
        for ddl in (
            "ALTER TABLE waste_entries ADD COLUMN confidence REAL",
            "ALTER TABLE waste_entries ADD COLUMN llm_backend TEXT DEFAULT ''",
        ):
            try:
                with sqlite3.connect(DB_FILE) as conn:
                    conn.execute(ddl)
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
                    cur.execute(_PG_CREATE_USERS)
                    cur.execute(_PG_CREATE_CAMERA)
            logger.info(
                "PostgreSQL database ready: %s@%s:%s/%s", DB_USER, DB_HOST, DB_PORT, DB_NAME
            )
        finally:
            pool.putconn(conn)
    else:
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(_SQLITE_CREATE)
            conn.execute(_SQLITE_CREATE_USERS)
            conn.execute(_SQLITE_CREATE_CAMERA)
            # PG creates these in its DDL; legacy tables may lack the columns.
            for ddl in (
                "CREATE INDEX IF NOT EXISTS idx_waste_label ON waste_entries(label)",
                "CREATE INDEX IF NOT EXISTS idx_waste_ts ON waste_entries(timestamp)",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass
        logger.info("SQLite database ready: %s", DB_FILE)
    _migrate_add_bin_id()
    _migrate_add_llm_fields()
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
    row.setdefault("confidence", None)
    row.setdefault("llm_backend", "")
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


def get_entries(
    limit: int = 100,
    offset: int = 0,
    bin_id: str | None = None,
    label: str | None = None,
    q: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    """Return recent classification entries, newest first.

    Optional filters: bin, exact label, q substring (description or brand),
    and a half-open [since, until) timestamp range.
    """
    _ensure_init()
    cols = list(_INSERT_COLS) + ["id"]
    pg = _use_pg()
    where, params = _filters_sql(pg, bin_id=bin_id, label=label, q=q, since=since, until=until)
    params.update({"limit": limit, "offset": offset})
    tail = (
        " ORDER BY id DESC LIMIT %(limit)s OFFSET %(offset)s"
        if pg
        else " ORDER BY id DESC LIMIT :limit OFFSET :offset"
    )
    sql = "SELECT " + ", ".join(cols) + f" FROM waste_entries{where}{tail}"
    try:
        return [dict(zip(cols, r)) for r in _fetch_rows(sql, params)]
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


def get_entry_count(
    bin_id: str | None = None,
    label: str | None = None,
    q: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> int:
    """Return the number of classification entries matching the given filters."""
    _ensure_init()
    where, params = _filters_sql(
        _use_pg(), bin_id=bin_id, label=label, q=q, since=since, until=until
    )
    try:
        rows = _fetch_rows(f"SELECT COUNT(*) FROM waste_entries{where}", params)
        return int(rows[0][0])
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


# ── Date-range analytics queries ──────────────────────────────────────────────

# granularity → (SQLite bucket expression, PG to_char format). The stored
# 'YYYY-MM-DD HH:MM:SS' format makes string prefixes exact bucket keys.
_BUCKET_SQL = {
    "hour": ("substr(timestamp, 1, 13)", "YYYY-MM-DD HH24"),
    "day": ("substr(timestamp, 1, 10)", "YYYY-MM-DD"),
}


def get_summary_in_range(since: str, until: str) -> dict:
    """Return {'total': int, 'avg_confidence': float | None} for [since, until)."""
    _ensure_init()
    where, params = _filters_sql(_use_pg(), since=since, until=until)
    try:
        row = _fetch_rows(f"SELECT COUNT(*), AVG(confidence) FROM waste_entries{where}", params)[0]
        return {
            "total": int(row[0]),
            "avg_confidence": float(row[1]) if row[1] is not None else None,
        }
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return {"total": 0, "avg_confidence": None}


def get_label_counts_in_range(since: str, until: str) -> dict[str, int]:
    """Return {label: count} for entries in [since, until)."""
    _ensure_init()
    where, params = _filters_sql(_use_pg(), since=since, until=until)
    try:
        rows = _fetch_rows(
            f"SELECT label, COUNT(*) FROM waste_entries{where} GROUP BY label", params
        )
        return {r[0]: int(r[1]) for r in rows}
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return {}


def get_timeseries_in_range(since: str, until: str, granularity: str = "day") -> list[dict]:
    """Return [{'bucket', 'label', 'count'}] for [since, until).

    Bucket keys are 'YYYY-MM-DD' (day) or 'YYYY-MM-DD HH' (hour) on both
    backends. granularity is whitelisted — it is interpolated into SQL.
    """
    if granularity not in _BUCKET_SQL:
        raise ValueError(f"granularity must be one of {sorted(_BUCKET_SQL)}, got {granularity!r}")
    _ensure_init()
    pg = _use_pg()
    sqlite_expr, pg_fmt = _BUCKET_SQL[granularity]
    bucket = f"to_char(timestamp, '{pg_fmt}')" if pg else sqlite_expr
    where, params = _filters_sql(pg, since=since, until=until)
    sql = f"SELECT {bucket} AS bucket, label, COUNT(*) FROM waste_entries{where} GROUP BY bucket, label"
    try:
        return [
            {"bucket": r[0], "label": r[1], "count": int(r[2])} for r in _fetch_rows(sql, params)
        ]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []


def get_bin_counts_in_range(since: str, until: str) -> dict[str, int]:
    """Return {bin_id: count} for entries in [since, until)."""
    _ensure_init()
    where, params = _filters_sql(_use_pg(), since=since, until=until)
    where += " AND bin_id IS NOT NULL" if where else " WHERE bin_id IS NOT NULL"
    try:
        rows = _fetch_rows(
            f"SELECT bin_id, COUNT(*) FROM waste_entries{where} GROUP BY bin_id", params
        )
        return {r[0]: int(r[1]) for r in rows}
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return {}


def get_backend_stats_in_range(since: str, until: str) -> list[dict]:
    """Return [{'backend', 'count', 'avg_confidence'}] for [since, until).

    Blank/NULL llm_backend values are folded into 'unknown'; busiest first.
    """
    _ensure_init()
    where, params = _filters_sql(_use_pg(), since=since, until=until)
    sql = (
        "SELECT COALESCE(NULLIF(llm_backend, ''), 'unknown') AS backend,"
        f" COUNT(*), AVG(confidence) FROM waste_entries{where}"
        " GROUP BY backend ORDER BY COUNT(*) DESC"
    )
    try:
        return [
            {
                "backend": r[0],
                "count": int(r[1]),
                "avg_confidence": float(r[2]) if r[2] is not None else None,
            }
            for r in _fetch_rows(sql, params)
        ]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []


def get_entry_by_id(entry_id: int) -> dict | None:
    """Return one classification entry by primary key, or None."""
    _ensure_init()
    cols = list(_INSERT_COLS) + ["id"]
    ph = "%(id)s" if _use_pg() else ":id"
    sql = "SELECT " + ", ".join(cols) + f" FROM waste_entries WHERE id = {ph}"
    try:
        rows = _fetch_rows(sql, {"id": entry_id})
        return dict(zip(cols, rows[0])) if rows else None
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return None


# ── User accounts ─────────────────────────────────────────────────────────────


def count_users() -> int:
    """Return the number of dashboard accounts (0 → the seed should run)."""
    _ensure_init()
    try:
        return int(_fetch_rows("SELECT COUNT(*) FROM users")[0][0])
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return 0


def get_user(username: str) -> dict | None:
    """Return {id, username, password_hash, created_at} or None."""
    _ensure_init()
    ph = "%(u)s" if _use_pg() else ":u"
    sql = f"SELECT id, username, password_hash, created_at FROM users WHERE username = {ph}"
    try:
        rows = _fetch_rows(sql, {"u": username})
        if not rows:
            return None
        r = rows[0]
        return {
            "id": r[0],
            "username": r[1],
            "password_hash": r[2],
            "created_at": str(r[3]) if r[3] else "",
        }
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return None


def list_users() -> list[dict]:
    """Return all accounts (id, username, created_at) — never the hash."""
    _ensure_init()
    try:
        rows = _fetch_rows("SELECT id, username, created_at FROM users ORDER BY username")
        return [
            {"id": r[0], "username": r[1], "created_at": str(r[2]) if r[2] else ""} for r in rows
        ]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []


def list_password_hashes() -> list[str]:
    """Return every stored password hash — used to match a bearer token."""
    _ensure_init()
    try:
        return [r[0] for r in _fetch_rows("SELECT password_hash FROM users")]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []


def create_user(username: str, password_hash: str) -> int | None:
    """Insert one account. Returns the new id, or None on failure (incl. a
    duplicate username, which violates the UNIQUE constraint)."""
    _ensure_init()
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO users (username, password_hash)"
                            " VALUES (%(u)s, %(p)s) RETURNING id",
                            {"u": username, "p": password_hash},
                        )
                        return int(cur.fetchone()[0])
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at)"
                    " VALUES (:u, :p, :c)",
                    {
                        "u": username,
                        "p": password_hash,
                        "c": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                return cur.lastrowid
    except Exception as e:
        logger.error("Create user failed: %s", e)
        return None


def set_password(username: str, password_hash: str) -> bool:
    """Update one account's password hash. Returns True if a row changed."""
    _ensure_init()
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE users SET password_hash = %(p)s WHERE username = %(u)s",
                            {"p": password_hash, "u": username},
                        )
                        return cur.rowcount > 0
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                cur = conn.execute(
                    "UPDATE users SET password_hash = :p WHERE username = :u",
                    {"p": password_hash, "u": username},
                )
                return cur.rowcount > 0
    except Exception as e:
        logger.error("Set password failed: %s", e)
        return False


def delete_user(username: str) -> bool:
    """Delete one account. Returns True if a row was removed."""
    _ensure_init()
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM users WHERE username = %(u)s", {"u": username}
                        )
                        return cur.rowcount > 0
            finally:
                pool.putconn(conn)
        else:
            with sqlite3.connect(DB_FILE) as conn:
                cur = conn.execute("DELETE FROM users WHERE username = :u", {"u": username})
                return cur.rowcount > 0
    except Exception as e:
        logger.error("Delete user failed: %s", e)
        return False


# ── Camera geometry configs ───────────────────────────────────────────────────


def get_camera_configs(bin_id: str) -> list[dict]:
    """Return saved per-camera transforms for a bin, ordered by camera index."""
    _ensure_init()
    ph = "%(b)s" if _use_pg() else ":b"
    sql = (
        "SELECT cam_index, rotation, flip_h, flip_v, crop_x0, crop_y0, crop_x1, crop_y1,"
        f" updated_at FROM camera_configs WHERE bin_id = {ph} ORDER BY cam_index"
    )
    try:
        rows = _fetch_rows(sql, {"b": bin_id})
        return [
            {
                "cam_index": r[0],
                "rotation": r[1],
                "flip_h": bool(r[2]),
                "flip_v": bool(r[3]),
                "crop": [r[4], r[5], r[6], r[7]],
                "updated_at": str(r[8]) if r[8] else "",
            }
            for r in rows
        ]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []


def upsert_camera_config(bin_id: str, cam_index: int, cfg: dict) -> bool:
    """Insert-or-update one camera's transform.

    cfg: ``{rotation, flip_h, flip_v, crop: [x0, y0, x1, y1]}``.
    """
    _ensure_init()
    crop = cfg.get("crop") or [0.0, 0.0, 1.0, 1.0]
    params = {
        "b": bin_id,
        "i": int(cam_index),
        "rot": int(cfg.get("rotation", 0)),
        "fh": bool(cfg.get("flip_h", False)),
        "fv": bool(cfg.get("flip_v", False)),
        "x0": float(crop[0]),
        "y0": float(crop[1]),
        "x1": float(crop[2]),
        "y1": float(crop[3]),
    }
    try:
        if _use_pg():
            pool = _get_pg_pool()
            conn = pool.getconn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO camera_configs"
                            " (bin_id, cam_index, rotation, flip_h, flip_v,"
                            "  crop_x0, crop_y0, crop_x1, crop_y1)"
                            " VALUES (%(b)s, %(i)s, %(rot)s, %(fh)s, %(fv)s,"
                            "  %(x0)s, %(y0)s, %(x1)s, %(y1)s)"
                            " ON CONFLICT (bin_id, cam_index) DO UPDATE SET"
                            "  rotation = EXCLUDED.rotation, flip_h = EXCLUDED.flip_h,"
                            "  flip_v = EXCLUDED.flip_v, crop_x0 = EXCLUDED.crop_x0,"
                            "  crop_y0 = EXCLUDED.crop_y0, crop_x1 = EXCLUDED.crop_x1,"
                            "  crop_y1 = EXCLUDED.crop_y1, updated_at = NOW()",
                            params,
                        )
                return True
            finally:
                pool.putconn(conn)
        else:
            params["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO camera_configs"
                    " (bin_id, cam_index, rotation, flip_h, flip_v,"
                    "  crop_x0, crop_y0, crop_x1, crop_y1, updated_at)"
                    " VALUES (:b, :i, :rot, :fh, :fv, :x0, :y0, :x1, :y1, :ts)",
                    params,
                )
            return True
    except Exception as e:
        logger.error("Upsert camera config failed: %s", e)
        return False
