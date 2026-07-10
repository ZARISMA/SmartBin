from smartwaste.database import get_entry_count, insert_entry, init_db, _ensure_init, DB_FILE
import sqlite3
import os
import time

os.environ["SMARTWASTE_DB_BACKEND"] = "sqlite"
init_db()

with sqlite3.connect(DB_FILE) as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS waste_entries (id INTEGER PRIMARY KEY, bin_id TEXT)")

_ensure_init()

print("Initial count:", get_entry_count())

start = time.time()
for _ in range(100):
    get_entry_count()
end = time.time()
print(f"100 calls without cache (or before optimization): {end - start:.4f}s")
