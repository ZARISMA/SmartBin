"""Tests for smartwaste/database.py — SQLite backend and query functions."""

import sqlite3

import pytest


def _entry():
    return {
        "filename": "/path/file.jpg",
        "label": "Plastic",
        "description": "A bottle",
        "brand_product": "Coca-Cola",
        "location": "Yerevan",
        "weight": "",
        "timestamp": "2026-01-01 12:00:00",
    }


def _env():
    return {
        "simulated_temperature": 22.5,
        "simulated_humidity": 50.0,
        "simulated_vibration": 0.05,
        "simulated_air_pollution": 25.0,
        "simulated_smoke": 0.1,
    }


def _setup_db(tmp_path, monkeypatch):
    """Set up a fresh SQLite database for testing."""
    import smartwaste.database as db

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_FILE", db_path)
    monkeypatch.setattr(db, "_initialized", False)
    monkeypatch.setattr(db, "_pg_pool", None)
    # Force SQLite backend
    monkeypatch.setattr(db, "DB_BACKEND", "sqlite")
    db.init_db()
    return db, db_path


# ─────────────────────────────────────────────────────────────────────────────
# init_db
# ─────────────────────────────────────────────────────────────────────────────


class TestInitDb:
    def test_creates_waste_entries_table(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='waste_entries'"
            )
            assert cur.fetchone() is not None

    def test_idempotent_double_call(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        db.init_db()  # must not raise

    def test_table_has_simulated_temperature_column(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        with sqlite3.connect(db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        assert "simulated_temperature" in cols

    def test_table_has_all_simulated_columns(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        with sqlite3.connect(db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        for col in (
            "simulated_temperature",
            "simulated_humidity",
            "simulated_vibration",
            "simulated_air_pollution",
            "simulated_smoke",
        ):
            assert col in cols

    def test_table_has_core_columns(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        with sqlite3.connect(db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        for col in (
            "id",
            "filename",
            "label",
            "description",
            "brand_product",
            "location",
            "weight",
            "timestamp",
        ):
            assert col in cols

    def test_no_bare_sensor_columns(self, tmp_path, monkeypatch):
        """Old column names (without 'simulated_' prefix) must not exist."""
        db, db_path = _setup_db(tmp_path, monkeypatch)
        with sqlite3.connect(db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        for old in ("temperature", "humidity", "vibration", "air_pollution", "smoke"):
            assert old not in cols

    def test_id_column_is_primary_key(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        with sqlite3.connect(db_path) as conn:
            info = {r[1]: r for r in conn.execute("PRAGMA table_info(waste_entries)")}
        assert info["id"][5] == 1  # pk column


# ─────────────────────────────────────────────────────────────────────────────
# insert_entry
# ─────────────────────────────────────────────────────────────────────────────


class TestInsertEntry:
    def test_inserts_one_row(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM waste_entries").fetchone()[0] == 1

    def test_label_stored_correctly(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT label FROM waste_entries").fetchone()
        assert row[0] == "Plastic"

    def test_description_stored_correctly(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT description FROM waste_entries").fetchone()
        assert row[0] == "A bottle"

    def test_simulated_temperature_stored(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            val = conn.execute("SELECT simulated_temperature FROM waste_entries").fetchone()[0]
        assert val == pytest.approx(22.5)

    def test_all_simulated_values_stored(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT simulated_temperature, simulated_humidity, simulated_vibration,"
                "       simulated_air_pollution, simulated_smoke FROM waste_entries"
            ).fetchone()
        assert row[0] == pytest.approx(22.5)
        assert row[1] == pytest.approx(50.0)
        assert row[2] == pytest.approx(0.05)
        assert row[3] == pytest.approx(25.0)
        assert row[4] == pytest.approx(0.1)

    def test_multiple_inserts_accumulate(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        for i in range(5):
            e = _entry()
            e["label"] = f"Label{i}"
            db.insert_entry(e, _env())
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM waste_entries").fetchone()[0] == 5

    def test_unicode_brand_product_stored(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        e = _entry()
        e["brand_product"] = "Ջերմուկ"
        db.insert_entry(e, _env())
        with sqlite3.connect(db_path) as conn:
            val = conn.execute("SELECT brand_product FROM waste_entries").fetchone()[0]
        assert val == "Ջերմուկ"

    def test_handles_error_gracefully(self, tmp_path, monkeypatch):
        """insert_entry catches exceptions — should NOT raise."""
        db, _ = _setup_db(tmp_path, monkeypatch)
        # An empty dict will cause a missing-parameter error, caught internally
        db.insert_entry({}, {})  # must not raise

    def test_location_stored_correctly(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            val = conn.execute("SELECT location FROM waste_entries").fetchone()[0]
        assert val == "Yerevan"

    def test_timestamp_stored_correctly(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            val = conn.execute("SELECT timestamp FROM waste_entries").fetchone()[0]
        assert val == "2026-01-01 12:00:00"

    def test_autoincrement_ids(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM waste_entries")]
        assert ids == [1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Query functions
# ─────────────────────────────────────────────────────────────────────────────


class TestGetEntries:
    def test_returns_empty_list_when_no_data(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        assert db.get_entries() == []

    def test_returns_inserted_entries(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        entries = db.get_entries()
        assert len(entries) == 1
        assert entries[0]["label"] == "Plastic"

    def test_limit_works(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        for i in range(5):
            e = _entry()
            e["label"] = f"Label{i}"
            db.insert_entry(e, _env())
        entries = db.get_entries(limit=3)
        assert len(entries) == 3

    def test_newest_first(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        for label in ["First", "Second", "Third"]:
            e = _entry()
            e["label"] = label
            db.insert_entry(e, _env())
        entries = db.get_entries()
        assert entries[0]["label"] == "Third"
        assert entries[2]["label"] == "First"


class TestGetLabelCounts:
    def test_returns_empty_dict_when_no_data(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        assert db.get_label_counts() == {}

    def test_counts_labels(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        for label in ["Plastic", "Plastic", "Glass"]:
            e = _entry()
            e["label"] = label
            db.insert_entry(e, _env())
        counts = db.get_label_counts()
        assert counts["Plastic"] == 2
        assert counts["Glass"] == 1


class TestGetEntryCount:
    def test_returns_zero_when_empty(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        assert db.get_entry_count() == 0

    def test_counts_all_entries(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        for _ in range(3):
            db.insert_entry(_entry(), _env())
        assert db.get_entry_count() == 3


# ─────────────────────────────────────────────────────────────────────────────
# confidence / llm_backend columns (server-side LLM classification)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMFieldsMigration:
    def test_new_columns_exist_on_fresh_db(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        with sqlite3.connect(db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        assert "confidence" in cols
        assert "llm_backend" in cols

    def test_migration_upgrades_old_table(self, tmp_path, monkeypatch):
        """A pre-existing table without the new columns gains them on init."""
        import smartwaste.database as db

        db_path = str(tmp_path / "old.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE waste_entries (id INTEGER PRIMARY KEY, label TEXT)")
        monkeypatch.setattr(db, "DB_FILE", db_path)
        monkeypatch.setattr(db, "_initialized", False)
        monkeypatch.setattr(db, "DB_BACKEND", "sqlite")
        db.init_db()
        with sqlite3.connect(db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        assert "confidence" in cols
        assert "llm_backend" in cols

    def test_insert_without_confidence_stores_null(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT confidence, llm_backend FROM waste_entries").fetchone()
        assert row[0] is None
        assert row[1] == ""

    def test_insert_with_confidence_persists(self, tmp_path, monkeypatch):
        db, db_path = _setup_db(tmp_path, monkeypatch)
        e = _entry()
        e["confidence"] = 0.87
        e["llm_backend"] = "lmstudio"
        db.insert_entry(e, _env())
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT confidence, llm_backend FROM waste_entries").fetchone()
        assert row[0] == pytest.approx(0.87)
        assert row[1] == "lmstudio"

    def test_get_entries_includes_new_fields(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        e = _entry()
        e["confidence"] = 0.5
        db.insert_entry(e, _env())
        entry = db.get_entries()[0]
        assert "confidence" in entry
        assert "llm_backend" in entry


# ─────────────────────────────────────────────────────────────────────────────
# Date-range analytics queries + entry filters
# ─────────────────────────────────────────────────────────────────────────────


def _seed_range_data(db):
    """Four entries across two days, mixed bins/backends/confidence."""
    rows = [
        ("Plastic", "2026-01-01 10:00:00", "bin-01", 0.9, "gemini", "A bottle", "Coca-Cola"),
        ("Glass", "2026-01-01 11:30:00", "bin-01", None, "", "Green jar", ""),
        ("Plastic", "2026-01-02 09:00:00", "bin-02", 0.7, "lmstudio", "Bag", "Jermuk wrap"),
        ("Empty", "2026-01-02 12:00:00", "bin-02", 0.5, "gemini", "", ""),
    ]
    for label, ts, bin_id, conf, backend, desc, brand in rows:
        e = _entry()
        e.update(
            {
                "label": label,
                "timestamp": ts,
                "bin_id": bin_id,
                "confidence": conf,
                "llm_backend": backend,
                "description": desc,
                "brand_product": brand,
            }
        )
        db.insert_entry(e, _env())


class TestSummaryInRange:
    def test_counts_and_averages_in_window(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        s = db.get_summary_in_range("2026-01-01 00:00:00", "2026-01-02 00:00:00")
        assert s["total"] == 2
        assert s["avg_confidence"] == pytest.approx(0.9)  # NULL confidence excluded

    def test_range_is_half_open(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        # until == exact timestamp of the 09:00 entry → excluded
        s = db.get_summary_in_range("2026-01-01 00:00:00", "2026-01-02 09:00:00")
        assert s["total"] == 2

    def test_empty_window(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        s = db.get_summary_in_range("2025-01-01 00:00:00", "2025-02-01 00:00:00")
        assert s == {"total": 0, "avg_confidence": None}


class TestLabelCountsInRange:
    def test_scopes_to_window(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        counts = db.get_label_counts_in_range("2026-01-02 00:00:00", "2026-01-03 00:00:00")
        assert counts == {"Plastic": 1, "Empty": 1}


class TestTimeseriesInRange:
    def test_day_buckets(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        rows = db.get_timeseries_in_range(
            "2026-01-01 00:00:00", "2026-01-03 00:00:00", granularity="day"
        )
        assert {"bucket": "2026-01-01", "label": "Plastic", "count": 1} in rows
        assert {"bucket": "2026-01-02", "label": "Empty", "count": 1} in rows

    def test_hour_buckets(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        rows = db.get_timeseries_in_range(
            "2026-01-01 00:00:00", "2026-01-02 00:00:00", granularity="hour"
        )
        buckets = {r["bucket"] for r in rows}
        assert buckets == {"2026-01-01 10", "2026-01-01 11"}

    def test_rejects_bad_granularity(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            db.get_timeseries_in_range("2026-01-01 00:00:00", "2026-01-02 00:00:00", "week")


class TestBinCountsInRange:
    def test_groups_by_bin(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        counts = db.get_bin_counts_in_range("2026-01-01 00:00:00", "2026-01-03 00:00:00")
        assert counts == {"bin-01": 2, "bin-02": 2}


class TestBackendStatsInRange:
    def test_blank_backend_folds_to_unknown(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        stats = db.get_backend_stats_in_range("2026-01-01 00:00:00", "2026-01-03 00:00:00")
        by_name = {s["backend"]: s for s in stats}
        assert by_name["gemini"]["count"] == 2
        assert by_name["gemini"]["avg_confidence"] == pytest.approx(0.7)
        assert by_name["unknown"]["count"] == 1
        assert by_name["unknown"]["avg_confidence"] is None


class TestEntryFilters:
    def test_label_filter(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        entries = db.get_entries(label="Plastic")
        assert len(entries) == 2
        assert all(e["label"] == "Plastic" for e in entries)

    def test_q_matches_description_or_brand_case_insensitive(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        assert len(db.get_entries(q="bottle")) == 1  # description, lowercased
        assert len(db.get_entries(q="jermuk")) == 1  # brand_product

    def test_range_filter_on_entries(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        entries = db.get_entries(since="2026-01-02 00:00:00", until="2026-01-03 00:00:00")
        assert len(entries) == 2

    def test_combined_filters_and_count(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        assert db.get_entry_count(bin_id="bin-02", label="Plastic") == 1
        assert db.get_entry_count(label="Plastic", q="bag") == 1
        assert db.get_entry_count() == 4


class TestGetEntryById:
    def test_returns_row(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        _seed_range_data(db)
        first = db.get_entries(limit=1, offset=3)[0]
        row = db.get_entry_by_id(first["id"])
        assert row is not None
        assert row["label"] == first["label"]

    def test_missing_id_returns_none(self, tmp_path, monkeypatch):
        db, _ = _setup_db(tmp_path, monkeypatch)
        assert db.get_entry_by_id(999) is None
