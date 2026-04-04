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


class TestDrawOverlayHistory:
    def test_history_none_does_not_raise(self):
        draw_overlay(_img(), "Plastic", "detail", False, None)

    def test_history_empty_does_not_raise(self):
        draw_overlay(_img(), "Plastic", "detail", False, [])

    def test_history_renders_without_raise(self):
        history = [("14:23", "Plastic"), ("14:22", "Glass"), ("14:20", "Paper")]
        draw_overlay(_img(), "Organic", "detail", True, history)

    def test_history_five_items_without_raise(self):
        history = [
            ("14:25", "Aluminum"),
            ("14:24", "Plastic"),
            ("14:23", "Glass"),
            ("14:22", "Organic"),
            ("14:21", "Paper"),
        ]
        draw_overlay(_img(), "Aluminum", "can", False, history)

    def test_history_modifies_image(self):
        img_with    = _img()
        img_without = _img()
        history = [("14:23", "Plastic")]
        draw_overlay(img_with,    "Plastic", "detail", False, history)
        draw_overlay(img_without, "Plastic", "detail", False, None)
        assert img_with.sum() != img_without.sum()
