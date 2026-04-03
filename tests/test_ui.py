"""Tests for smartwaste/ui.py — draw_overlay()."""

import numpy as np
import pytest

from smartwaste.ui import draw_overlay


def _img(h=200, w=1600):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDrawOverlayBasic:
    def test_does_not_raise_on_normal_input(self):
        draw_overlay(_img(), "Ready", "Press c to classify", False)

    def test_does_not_raise_with_auto_on(self):
        draw_overlay(_img(), "Ready", "detail", True)

    def test_does_not_raise_with_empty_strings(self):
        draw_overlay(_img(), "", "", False)

    def test_does_not_raise_with_long_strings(self):
        draw_overlay(_img(), "X" * 200, "Y" * 200, True)

    def test_does_not_raise_for_every_valid_category(self):
        from smartwaste.config import VALID_CLASSES
        for cat in VALID_CLASSES:
            draw_overlay(_img(), cat, "detail", False)


class TestDrawOverlayModifiesImage:
    def test_image_is_modified_in_place(self):
        img = _img()
        original_sum = img.sum()
        draw_overlay(img, "Plastic", "Some detail", False)
        # The overlay draws a rectangle and text, so pixel sum must change
        assert img.sum() != original_sum

    def test_image_shape_preserved(self):
        img = _img()
        shape_before = img.shape
        draw_overlay(img, "Glass", "detail", True)
        assert img.shape == shape_before

    def test_image_dtype_preserved(self):
        img = _img()
        draw_overlay(img, "Paper", "detail", False)
        assert img.dtype == np.uint8

    def test_auto_mode_produces_different_pixels_than_manual(self):
        img_auto   = _img()
        img_manual = _img()
        draw_overlay(img_auto,   "Ready", "detail", True)
        draw_overlay(img_manual, "Ready", "detail", False)
        # Text differs ("AUTO" vs "MANUAL") so pixel sums should differ
        assert img_auto.sum() != img_manual.sum()


class TestDrawOverlayReturnValue:
    def test_returns_none(self):
        result = draw_overlay(_img(), "Ready", "detail", False)
        assert result is None


class TestDrawOverlaySmallImage:
    def test_works_on_minimal_image(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        draw_overlay(img, "Ready", "d", False)

    def test_works_on_square_image(self):
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        draw_overlay(img, "Plastic", "bottle", True)
