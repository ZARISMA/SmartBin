import time
import os
import sqlite3
from smartwaste.database import get_entry_count, DB_FILE, _ensure_init, init_db

# Set up some dummy data
os.environ["SMARTWASTE_DB_BACKEND"] = "sqlite"

init_db()

with sqlite3.connect(DB_FILE) as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS waste_entries (id INTEGER PRIMARY KEY, bin_id TEXT)")
    # Insert a lot of rows
    print("Inserting rows...")
    conn.execute("BEGIN TRANSACTION")
    for _ in range(100000):
        conn.execute("INSERT INTO waste_entries (bin_id) VALUES ('bin_1')")
    conn.execute("COMMIT")

_ensure_init()

print("Benchmarking get_entry_count()...")
start_time = time.time()
for _ in range(100):
    get_entry_count()
end_time = time.time()

print(f"Time taken for 100 calls: {end_time - start_time:.4f} seconds")
