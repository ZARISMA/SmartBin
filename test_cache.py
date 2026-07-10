import time
from smartwaste.database import get_entry_count, DB_FILE, _ensure_init, init_db
import sqlite3
import os

os.environ["SMARTWASTE_DB_BACKEND"] = "sqlite"

init_db()

with sqlite3.connect(DB_FILE) as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS waste_entries (id INTEGER PRIMARY KEY, bin_id TEXT)")
    conn.execute("BEGIN TRANSACTION")
    for _ in range(100):
        conn.execute("INSERT INTO waste_entries (bin_id) VALUES ('bin_1')")
    conn.execute("COMMIT")

_ensure_init()

print("Initial call:", get_entry_count())
start = time.time()
for _ in range(1000):
    get_entry_count()
end = time.time()
print(f"Time for 1000 calls without cache: {end - start:.4f}s")
