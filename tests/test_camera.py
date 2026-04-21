"""Tests for crop_sides() in smartwaste/cameraOak.py."""

import numpy as np
import pytest

from smartwaste.cameraOak import crop_sides


class TestNoCrop:
    def test_zero_percent_returns_same_shape(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        assert crop_sides(frame, 0.0).shape == frame.shape

    def test_negative_percent_returns_same_shape(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        assert crop_sides(frame, -0.5).shape == frame.shape

    def test_zero_percent_data_unchanged(self):
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        np.testing.assert_array_equal(crop_sides(frame, 0.0), frame)

    def test_negative_percent_data_unchanged(self):
        frame = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
        np.testing.assert_array_equal(crop_sides(frame, -1.0), frame)


class TestCropGeometry:
    def test_20_percent_width(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = crop_sides(frame, 0.20)
        left, right = int(200 * 0.20), int(200 * 0.80)
        assert result.shape[1] == right - left  # 120

    def test_height_is_preserved(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert crop_sides(frame, 0.20).shape[0] == 480

    def test_channels_preserved(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        assert crop_sides(frame, 0.10).shape[2] == 3

    def test_10_percent_on_100px_wide(self):
        frame = np.zeros((10, 100, 3), dtype=np.uint8)
        result = crop_sides(frame, 0.10)
        assert result.shape[1] == 80  # remove 10 from each side

    def test_25_percent_on_200px_wide(self):
        frame = np.zeros((10, 200, 3), dtype=np.uint8)
        result = crop_sides(frame, 0.25)
        assert result.shape[1] == 100

    def test_50_percent_yields_zero_width(self):
        frame = np.zeros((10, 100, 3), dtype=np.uint8)
        result = crop_sides(frame, 0.50)
        # left=50, right=50 → frame[:,50:50] has 0 columns
        assert result.shape[1] == 0

    def test_odd_width_floor_truncation(self):
        frame = np.zeros((10, 101, 3), dtype=np.uint8)
        left = int(101 * 0.20)  # 20
        right = int(101 * 0.80)  # 80
        assert crop_sides(frame, 0.20).shape[1] == right - left

    def test_large_hd_frame(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        left = int(1920 * 0.20)
        right = int(1920 * 0.80)
        assert crop_sides(frame, 0.20).shape == (1080, right - left, 3)


class TestCropContent:
    def test_removes_left_columns(self):
        frame = np.zeros((10, 100, 3), dtype=np.uint8)
        frame[:, :10] = 255  # leftmost 10 cols = white
        result = crop_sides(frame, 0.10)
        assert result[:, 0].max() == 0  # first remaining col should be black

    def test_removes_right_columns(self):
        frame = np.zeros((10, 100, 3), dtype=np.uint8)
        frame[:, -10:] = 255  # rightmost 10 cols = white
        result = crop_sides(frame, 0.10)
        assert result[:, -1].max() == 0  # last remaining col should be black

    def test_preserves_center_content(self):
        frame = np.zeros((10, 100, 3), dtype=np.uint8)
        frame[:, 50] = 128  # mark column 50 (center)
        result = crop_sides(frame, 0.10)
        # col 50 is at index 40 in cropped result (50 - left=10)
        assert result[:, 40, 0].max() == 128

    def test_both_extremes_removed(self):
        frame = np.zeros((10, 100, 3), dtype=np.uint8)
        frame[:, :10] = 200  # left edge
        frame[:, -10:] = 200  # right edge
        result = crop_sides(frame, 0.10)
        assert result.max() == 0

    def test_result_is_view_or_copy_with_correct_values(self):
        frame = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
        result = crop_sides(frame, 0.10)
        # left=1, right=9 -> columns 1..8
        np.testing.assert_array_equal(result, frame[:, 1:9])


class TestGrayscaleFrames:
    def test_grayscale_preserves_ndim(self):
        frame = np.zeros((100, 200), dtype=np.uint8)
        result = crop_sides(frame, 0.20)
        assert result.ndim == 2

    def test_grayscale_width_reduced(self):
        frame = np.zeros((100, 200), dtype=np.uint8)
        result = crop_sides(frame, 0.20)
        left, right = int(200 * 0.20), int(200 * 0.80)
        assert result.shape[1] == right - left

    def test_grayscale_height_preserved(self):
        frame = np.zeros((100, 200), dtype=np.uint8)
        assert crop_sides(frame, 0.20).shape[0] == 100

    def test_grayscale_zero_crop_unchanged(self):
        frame = np.arange(200, dtype=np.uint8).reshape(10, 20)
        np.testing.assert_array_equal(crop_sides(frame, 0.0), frame)
