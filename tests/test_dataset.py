"""Tests for smartwaste/dataset.py — environment data and save_entry."""

from unittest.mock import patch

import numpy as np
import pytest


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
            "simulated_temperature",
            "simulated_humidity",
            "simulated_vibration",
            "simulated_air_pollution",
            "simulated_smoke",
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
        first = list(ds._environment_data().values())
        second = list(ds._environment_data().values())
        assert first != second


# ─────────────────────────────────────────────────────────────────────────────
# save_entry
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveEntry:
    def _run(self, tmp_path, **kwargs):
        import smartwaste.dataset as ds

        label = kwargs.get("label", "Plastic")
        description = kwargs.get("description", "A bottle")
        brand_product = kwargs.get("brand_product", "Coca-Cola")
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        with (
            patch.object(ds, "DATASET_DIR", str(tmp_path)),
            patch("smartwaste.dataset.insert_entry") as mock_insert,
            patch("cv2.imwrite", return_value=True),
        ):
            ds.save_entry(label, frame, description, brand_product)
        return mock_insert

    def test_calls_insert_entry(self, tmp_path):
        mock_insert = self._run(tmp_path)
        assert mock_insert.called

    def test_insert_entry_receives_entry_dict(self, tmp_path):
        mock_insert = self._run(tmp_path, label="Glass", description="Green jar")
        entry = mock_insert.call_args[0][0]
        assert entry["label"] == "Glass"
        assert entry["description"] == "Green jar"

    def test_insert_entry_receives_env_with_simulated_keys(self, tmp_path):
        import smartwaste.dataset as ds

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        captured_env = {}

        def capture_insert(entry, env):
            captured_env.update(env)

        with (
            patch.object(ds, "DATASET_DIR", str(tmp_path)),
            patch("smartwaste.dataset.insert_entry", side_effect=capture_insert),
            patch("cv2.imwrite", return_value=True),
        ):
            ds.save_entry("Paper", frame, "sheet", "N/A")

        for key in (
            "simulated_temperature",
            "simulated_humidity",
            "simulated_vibration",
            "simulated_air_pollution",
            "simulated_smoke",
        ):
            assert key in captured_env

    def test_label_in_filename(self, tmp_path):
        import smartwaste.dataset as ds

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        written_paths = []

        def capture_write(path, _img):
            written_paths.append(path)
            return True

        with (
            patch.object(ds, "DATASET_DIR", str(tmp_path)),
            patch("smartwaste.dataset.insert_entry"),
            patch("cv2.imwrite", side_effect=capture_write),
        ):
            ds.save_entry("Aluminum", frame, "can", "BOOM")

        assert written_paths and "Aluminum" in written_paths[0]

    def test_entry_has_all_required_fields(self, tmp_path):
        mock_insert = self._run(tmp_path)
        entry = mock_insert.call_args[0][0]
        for key in (
            "filename",
            "label",
            "description",
            "brand_product",
            "location",
            "weight",
            "timestamp",
        ):
            assert key in entry

    def test_entry_location_is_yerevan(self, tmp_path):
        mock_insert = self._run(tmp_path)
        entry = mock_insert.call_args[0][0]
        assert entry["location"] == "Yerevan"
