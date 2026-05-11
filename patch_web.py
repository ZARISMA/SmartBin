import sys

content = open("smartwaste/web.py").read()

# Add module-level variables
vars_code = """
_cached_active_bins = None
_cached_active_bins_ts = 0.0

@app.get("/api/dashboard")
"""

content = content.replace('@app.get("/api/dashboard")', vars_code)

# Replace the specific db_bins line
old_line = 'db_bins = {b["bin_id"]: b for b in get_active_bins()}'
new_line = """global _cached_active_bins, _cached_active_bins_ts
    if _cached_active_bins is None or time.time() - _cached_active_bins_ts > 5.0:
        _cached_active_bins = get_active_bins()
        _cached_active_bins_ts = time.time()

    db_bins = {b["bin_id"]: b for b in _cached_active_bins}"""

content = content.replace(old_line, new_line)

open("smartwaste/web.py", "w").write(content)
