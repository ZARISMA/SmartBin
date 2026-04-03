"""Tests for smartwaste/database.py — SQLite init and insertion."""

import sqlite3

import pytest


def _entry():
    return {
        "filename":      "/path/file.jpg",
        "label":         "Plastic",
        "description":   "A bottle",
        "brand_product": "Coca-Cola",
        "location":      "Yerevan",
        "weight":        "",
        "timestamp":     "2026-01-01 12:00:00",
    }


def _env():
    return {
        "simulated_temperature":   22.5,
        "simulated_humidity":      50.0,
        "simulated_vibration":     0.05,
        "simulated_air_pollution": 25.0,
        "simulated_smoke":         0.1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# init_db
# ─────────────────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_waste_entries_table(self, tmp_path, monkeypatch):
        import smartwaste.database as db
        monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
        db.init_db()
        with sqlite3.connect(str(tmp_path / "test.db")) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='waste_entries'"
            )
            assert cur.fetchone() is not None

    def test_idempotent_double_call(self, tmp_path, monkeypatch):
        import smartwaste.database as db
        monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
        db.init_db()
        db.init_db()  # must not raise

    def test_table_has_simulated_temperature_column(self, tmp_path, monkeypatch):
        import smartwaste.database as db
        monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
        db.init_db()
        with sqlite3.connect(str(tmp_path / "test.db")) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        assert "simulated_temperature" in cols

    def test_table_has_all_simulated_columns(self, tmp_path, monkeypatch):
        import smartwaste.database as db
        monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
        db.init_db()
        with sqlite3.connect(str(tmp_path / "test.db")) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        for col in ("simulated_temperature", "simulated_humidity",
                    "simulated_vibration", "simulated_air_pollution", "simulated_smoke"):
            assert col in cols

    def test_table_has_core_columns(self, tmp_path, monkeypatch):
        import smartwaste.database as db
        monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
        db.init_db()
        with sqlite3.connect(str(tmp_path / "test.db")) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        for col in ("id", "filename", "label", "description",
                    "brand_product", "location", "weight", "timestamp"):
            assert col in cols

    def test_no_bare_sensor_columns(self, tmp_path, monkeypatch):
        """Old column names (without 'simulated_' prefix) must not exist."""
        import smartwaste.database as db
        monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
        db.init_db()
        with sqlite3.connect(str(tmp_path / "test.db")) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(waste_entries)")]
        for old in ("temperature", "humidity", "vibration", "air_pollution", "smoke"):
            assert old not in cols

    def test_id_column_is_primary_key(self, tmp_path, monkeypatch):
        import smartwaste.database as db
        monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
        db.init_db()
        with sqlite3.connect(str(tmp_path / "test.db")) as conn:
            info = {r[1]: r for r in conn.execute("PRAGMA table_info(waste_entries)")}
        assert info["id"][5] == 1  # pk column


# ─────────────────────────────────────────────────────────────────────────────
# insert_entry
# ─────────────────────────────────────────────────────────────────────────────

class TestInsertEntry:
    def _setup(self, tmp_path, monkeypatch):
        import smartwaste.database as db
        db_path = str(tmp_path / "test.db")
        monkeypatch.setattr(db, "DB_FILE", db_path)
        db.init_db()
        return db, db_path

    def test_inserts_one_row(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM waste_entries").fetchone()[0] == 1

    def test_label_stored_correctly(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT label FROM waste_entries").fetchone()
        assert row[0] == "Plastic"

    def test_description_stored_correctly(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT description FROM waste_entries").fetchone()
        assert row[0] == "A bottle"

    def test_simulated_temperature_stored(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            val = conn.execute("SELECT simulated_temperature FROM waste_entries").fetchone()[0]
        assert val == pytest.approx(22.5)

    def test_all_simulated_values_stored(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
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
        db, db_path = self._setup(tmp_path, monkeypatch)
        for i in range(5):
            e = _entry()
            e["label"] = f"Label{i}"
            db.insert_entry(e, _env())
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM waste_entries").fetchone()[0] == 5

    def test_unicode_brand_product_stored(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
        e = _entry()
        e["brand_product"] = "Ջերմուկ"
        db.insert_entry(e, _env())
        with sqlite3.connect(db_path) as conn:
            val = conn.execute("SELECT brand_product FROM waste_entries").fetchone()[0]
        assert val == "Ջերմուկ"

    def test_handles_error_gracefully(self, tmp_path, monkeypatch):
        """insert_entry catches exceptions — should NOT raise."""
        import smartwaste.database as db
        db_path = str(tmp_path / "test.db")
        monkeypatch.setattr(db, "DB_FILE", db_path)
        db.init_db()
        # An empty dict will cause a missing-parameter error in sqlite, caught internally
        db.insert_entry({}, {})  # must not raise

    def test_location_stored_correctly(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            val = conn.execute("SELECT location FROM waste_entries").fetchone()[0]
        assert val == "Yerevan"

    def test_timestamp_stored_correctly(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            val = conn.execute("SELECT timestamp FROM waste_entries").fetchone()[0]
        assert val == "2026-01-01 12:00:00"

    def test_autoincrement_ids(self, tmp_path, monkeypatch):
        db, db_path = self._setup(tmp_path, monkeypatch)
        db.insert_entry(_entry(), _env())
        db.insert_entry(_entry(), _env())
        with sqlite3.connect(db_path) as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM waste_entries")]
        assert ids == [1, 2]
