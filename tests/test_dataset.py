"""Tests for smartwaste/dataset.py — metadata, environment data, save_entry."""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolate_metadata(monkeypatch):
    """Reset module-level _metadata list before/after each test."""
    import smartwaste.dataset as ds
    original = list(ds._metadata)
    monkeypatch.setattr(ds, "_metadata", [])
    yield
    # Restore
    monkeypatch.setattr(ds, "_metadata", original)


# ─────────────────────────────────────────────────────────────────────────────
# load_metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadMetadata:
    def test_returns_empty_list_for_missing_file(self, tmp_path):
        import smartwaste.dataset as ds
        with patch.object(ds, "META_FILE", str(tmp_path / "nonexistent.json")):
            assert ds.load_metadata() == []

    def test_returns_empty_list_for_invalid_json(self, tmp_path):
        import smartwaste.dataset as ds
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{}")
        with patch.object(ds, "META_FILE", str(f)):
            assert ds.load_metadata() == []

    def test_returns_empty_list_for_empty_file(self, tmp_path):
        import smartwaste.dataset as ds
        f = tmp_path / "empty.json"
        f.write_text("")
        with patch.object(ds, "META_FILE", str(f)):
            assert ds.load_metadata() == []

    def test_loads_valid_list(self, tmp_path):
        import smartwaste.dataset as ds
        data = [{"label": "Plastic", "filename": "x.jpg"}]
        f = tmp_path / "meta.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(ds, "META_FILE", str(f)):
            assert ds.load_metadata() == data

    def test_loads_multiple_entries(self, tmp_path):
        import smartwaste.dataset as ds
        data = [{"label": f"L{i}"} for i in range(5)]
        f = tmp_path / "meta.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(ds, "META_FILE", str(f)):
            result = ds.load_metadata()
        assert len(result) == 5

    def test_loads_unicode_content(self, tmp_path):
        import smartwaste.dataset as ds
        data = [{"label": "Organic", "brand_product": "Ջերմուկ"}]
        f = tmp_path / "meta.json"
        f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with patch.object(ds, "META_FILE", str(f)):
            result = ds.load_metadata()
        assert result[0]["brand_product"] == "Ջերմուկ"


# ─────────────────────────────────────────────────────────────────────────────
# _environment_data
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentData:
    def test_returns_exactly_five_keys(self):
        import smartwaste.dataset as ds
        assert len(ds._environment_data()) == 5

    def test_has_all_simulated_keys(self):
        import smartwaste.dataset as ds
        expected = {
            "simulated_temperature", "simulated_humidity",
            "simulated_vibration", "simulated_air_pollution", "simulated_smoke",
        }
        assert set(ds._environment_data().keys()) == expected

    def test_no_bare_keys(self):
        import smartwaste.dataset as ds
        env = ds._environment_data()
        for key in ("temperature", "humidity", "vibration", "air_pollution", "smoke"):
            assert key not in env

    def test_temperature_in_range(self):
        import smartwaste.dataset as ds
        for _ in range(30):
            v = ds._environment_data()["simulated_temperature"]
            assert 15 <= v <= 30

    def test_humidity_in_range(self):
        import smartwaste.dataset as ds
        for _ in range(30):
            v = ds._environment_data()["simulated_humidity"]
            assert 30 <= v <= 70

    def test_vibration_in_range(self):
        import smartwaste.dataset as ds
        for _ in range(30):
            v = ds._environment_data()["simulated_vibration"]
            assert 0 <= v <= 0.1

    def test_air_pollution_in_range(self):
        import smartwaste.dataset as ds
        for _ in range(30):
            v = ds._environment_data()["simulated_air_pollution"]
            assert 5 <= v <= 50

    def test_smoke_in_range(self):
        import smartwaste.dataset as ds
        for _ in range(30):
            v = ds._environment_data()["simulated_smoke"]
            assert 0 <= v <= 1

    def test_all_values_are_floats(self):
        import smartwaste.dataset as ds
        for v in ds._environment_data().values():
            assert isinstance(v, float)

    def test_values_differ_across_calls(self):
        import smartwaste.dataset as ds
        # Very unlikely all 5 values match twice in a row
        first  = list(ds._environment_data().values())
        second = list(ds._environment_data().values())
        assert first != second


# ─────────────────────────────────────────────────────────────────────────────
# save_entry
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveEntry:
    def _run(self, tmp_path, **kwargs):
        import smartwaste.dataset as ds
        meta_file = tmp_path / "metadata.json"
        label        = kwargs.get("label",        "Plastic")
        description  = kwargs.get("description",  "A bottle")
        brand_product = kwargs.get("brand_product", "Coca-Cola")
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(ds, "DATASET_DIR", str(tmp_path)), \
             patch.object(ds, "META_FILE",   str(meta_file)), \
             patch("smartwaste.dataset.insert_entry") as mock_insert, \
             patch("cv2.imwrite", return_value=True):
            ds.save_entry(label, frame, description, brand_product)
        return meta_file, mock_insert

    def test_writes_metadata_json(self, tmp_path):
        meta_file, _ = self._run(tmp_path)
        assert meta_file.exists()

    def test_metadata_contains_one_entry(self, tmp_path):
        meta_file, _ = self._run(tmp_path)
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert len(data) == 1

    def test_entry_label_correct(self, tmp_path):
        meta_file, _ = self._run(tmp_path, label="Glass")
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert data[0]["label"] == "Glass"

    def test_entry_description_correct(self, tmp_path):
        meta_file, _ = self._run(tmp_path, description="Green jar")
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert data[0]["description"] == "Green jar"

    def test_entry_brand_product_correct(self, tmp_path):
        meta_file, _ = self._run(tmp_path, brand_product="Jermuk")
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert data[0]["brand_product"] == "Jermuk"

    def test_entry_location_is_yerevan(self, tmp_path):
        meta_file, _ = self._run(tmp_path)
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert data[0]["location"] == "Yerevan"

    def test_entry_has_timestamp(self, tmp_path):
        meta_file, _ = self._run(tmp_path)
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert "timestamp" in data[0] and data[0]["timestamp"]

    def test_entry_has_filename(self, tmp_path):
        meta_file, _ = self._run(tmp_path)
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert "filename" in data[0] and data[0]["filename"]

    def test_entry_has_all_required_fields(self, tmp_path):
        meta_file, _ = self._run(tmp_path)
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        for key in ("filename", "label", "description", "brand_product",
                    "location", "weight", "timestamp"):
            assert key in data[0]

    def test_calls_insert_entry(self, tmp_path):
        _, mock_insert = self._run(tmp_path)
        assert mock_insert.called

    def test_insert_entry_receives_env_with_simulated_keys(self, tmp_path):
        import smartwaste.dataset as ds
        meta_file = tmp_path / "metadata.json"
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        captured_env = {}

        def capture_insert(entry, env):
            captured_env.update(env)

        with patch.object(ds, "DATASET_DIR", str(tmp_path)), \
             patch.object(ds, "META_FILE",   str(meta_file)), \
             patch("smartwaste.dataset.insert_entry", side_effect=capture_insert), \
             patch("cv2.imwrite", return_value=True):
            ds.save_entry("Paper", frame, "sheet", "N/A")

        for key in ("simulated_temperature", "simulated_humidity",
                    "simulated_vibration", "simulated_air_pollution", "simulated_smoke"):
            assert key in captured_env

    def test_second_save_appends_to_existing_json(self, tmp_path):
        import smartwaste.dataset as ds
        meta_file = tmp_path / "metadata.json"
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch.object(ds, "DATASET_DIR", str(tmp_path)), \
             patch.object(ds, "META_FILE",   str(meta_file)), \
             patch("smartwaste.dataset.insert_entry"), \
             patch("cv2.imwrite", return_value=True):
            ds.save_entry("Plastic", frame, "bottle", "Coke")
            ds.save_entry("Glass",   frame, "jar",    "Bjni")

        data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["label"] == "Plastic"
        assert data[1]["label"] == "Glass"

    def test_label_in_filename(self, tmp_path):
        import smartwaste.dataset as ds
        meta_file = tmp_path / "metadata.json"
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        written_paths = []

        def capture_write(path, _img):
            written_paths.append(path)
            return True

        with patch.object(ds, "DATASET_DIR", str(tmp_path)), \
             patch.object(ds, "META_FILE",   str(meta_file)), \
             patch("smartwaste.dataset.insert_entry"), \
             patch("cv2.imwrite", side_effect=capture_write):
            ds.save_entry("Aluminum", frame, "can", "BOOM")

        assert written_paths and "Aluminum" in written_paths[0]
