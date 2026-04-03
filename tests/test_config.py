"""Tests for smartwaste/config.py — validate constant types and ranges."""

import os
import pytest

from smartwaste.config import (
    AUTO_INTERVAL,
    BASE_DIR,
    BG_LEARNING_RATE,
    BG_WARMUP_FRAMES,
    CHECK_INTERVAL,
    CROP_PERCENT,
    DATASET_DIR,
    DB_FILE,
    DETECT_CONFIRM_N,
    DISPLAY_SIZE,
    EMPTY_CONFIRM_N,
    JPEG_QUALITY,
    LOG_DIR,
    MAX_DT,
    META_FILE,
    MODEL_NAME,
    MOTION_THRESHOLD,
    VALID_CLASSES,
    WINDOW,
)


class TestValidClasses:
    def test_exactly_seven_categories(self):
        assert len(VALID_CLASSES) == 7

    def test_contains_all_expected_categories(self):
        expected = {"Plastic", "Glass", "Paper", "Organic", "Aluminum", "Other", "Empty"}
        assert set(VALID_CLASSES) == expected

    def test_no_duplicates(self):
        assert len(VALID_CLASSES) == len(set(VALID_CLASSES))

    def test_all_strings(self):
        assert all(isinstance(c, str) for c in VALID_CLASSES)

    def test_all_capitalized(self):
        assert all(c[0].isupper() for c in VALID_CLASSES)

    def test_empty_category_present(self):
        assert "Empty" in VALID_CLASSES


class TestCameraConstants:
    def test_jpeg_quality_in_valid_range(self):
        assert 1 <= JPEG_QUALITY <= 100

    def test_crop_percent_is_fraction(self):
        assert 0 < CROP_PERCENT < 0.5  # must crop less than half from each side

    def test_max_dt_positive(self):
        assert MAX_DT > 0

    def test_display_size_is_two_tuple(self):
        assert len(DISPLAY_SIZE) == 2
        assert all(isinstance(v, int) and v > 0 for v in DISPLAY_SIZE)

    def test_window_is_nonempty_string(self):
        assert isinstance(WINDOW, str) and len(WINDOW) > 0


class TestTimingConstants:
    def test_auto_interval_positive(self):
        assert AUTO_INTERVAL > 0

    def test_check_interval_positive(self):
        assert CHECK_INTERVAL > 0


class TestPresenceDetectorConstants:
    def test_motion_threshold_positive(self):
        assert MOTION_THRESHOLD > 0

    def test_detect_confirm_n_at_least_one(self):
        assert DETECT_CONFIRM_N >= 1

    def test_empty_confirm_n_at_least_one(self):
        assert EMPTY_CONFIRM_N >= 1

    def test_bg_learning_rate_valid_alpha(self):
        # cv2.accumulateWeighted requires alpha in (0, 1)
        assert 0 < BG_LEARNING_RATE < 1

    def test_bg_warmup_frames_positive(self):
        assert BG_WARMUP_FRAMES > 0

    def test_empty_confirm_n_greater_than_detect(self):
        # Should take longer to confirm "empty" than "occupied" to avoid
        # premature resets; this is a design constraint
        assert EMPTY_CONFIRM_N >= DETECT_CONFIRM_N


class TestPaths:
    def test_base_dir_exists(self):
        assert os.path.isdir(BASE_DIR)

    def test_dataset_dir_under_base(self):
        assert DATASET_DIR.startswith(BASE_DIR)

    def test_log_dir_under_base(self):
        assert LOG_DIR.startswith(BASE_DIR)

    def test_meta_file_under_dataset_dir(self):
        assert META_FILE.startswith(DATASET_DIR)

    def test_db_file_under_dataset_dir(self):
        assert DB_FILE.startswith(DATASET_DIR)

    def test_meta_file_is_json(self):
        assert META_FILE.endswith(".json")

    def test_db_file_is_sqlite(self):
        assert DB_FILE.endswith(".db")

    def test_model_name_nonempty(self):
        assert isinstance(MODEL_NAME, str) and len(MODEL_NAME) > 0
