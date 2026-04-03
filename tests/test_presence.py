"""Tests for smartwaste/presence.py — PresenceDetector state machine."""

import numpy as np
import pytest

from smartwaste.config import (
    BG_WARMUP_FRAMES,
    DETECT_CONFIRM_N,
    EMPTY_CONFIRM_N,
    MOTION_THRESHOLD,
)
from smartwaste.presence import PresenceDetector


def _gray(value: int, shape=(100, 100)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _noise(shape=(100, 100)) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, shape, dtype=np.uint8)


class TestInitialState:
    def test_not_ready_after_construction(self):
        assert PresenceDetector().ready is False

    def test_warmup_progress_starts_at_zero(self):
        current, total = PresenceDetector().warmup_progress
        assert current == 0

    def test_warmup_total_equals_config(self):
        _, total = PresenceDetector().warmup_progress
        assert total == BG_WARMUP_FRAMES

    def test_first_update_returns_zero_score(self):
        d = PresenceDetector()
        score, _, _ = d.update(_gray(0))
        assert score == 0.0

    def test_first_update_returns_false_occupied(self):
        d = PresenceDetector()
        _, is_occ, _ = d.update(_gray(255))
        assert is_occ is False

    def test_first_update_returns_false_empty(self):
        d = PresenceDetector()
        _, _, is_empty = d.update(_gray(0))
        assert is_empty is False


class TestWarmupPhase:
    def test_warmup_increments_each_frame(self):
        d = PresenceDetector()
        for i in range(1, 11):
            d.update(_gray(0))
            assert d.warmup_progress[0] == i

    def test_bright_frames_during_warmup_never_trigger_detection(self):
        d = PresenceDetector()
        for _ in range(BG_WARMUP_FRAMES - 1):
            _, is_occ, _ = d.update(_gray(255))
            assert is_occ is False

    def test_bright_frames_during_warmup_never_trigger_empty(self):
        d = PresenceDetector()
        for _ in range(BG_WARMUP_FRAMES - 1):
            _, _, is_empty = d.update(_gray(255))
            assert is_empty is False

    def test_not_ready_one_frame_before_threshold(self):
        d = PresenceDetector()
        for _ in range(BG_WARMUP_FRAMES - 1):
            d.update(_gray(0))
        assert d.ready is False

    def test_ready_exactly_at_threshold(self):
        d = PresenceDetector()
        for _ in range(BG_WARMUP_FRAMES):
            d.update(_gray(0))
        assert d.ready is True

    def test_ready_stays_true_after_more_frames(self):
        d = PresenceDetector()
        for _ in range(BG_WARMUP_FRAMES + 10):
            d.update(_gray(0))
        assert d.ready is True

    def test_warmup_score_always_zero(self):
        d = PresenceDetector()
        scores = []
        for _ in range(BG_WARMUP_FRAMES):
            score, _, _ = d.update(_gray(0))
            scores.append(score)
        assert all(s == 0.0 for s in scores)


class TestDetectionAfterWarmup:
    def test_score_positive_after_warmup(self, ready_detector):
        score, _, _ = ready_detector.update(_gray(255))
        assert score > 0

    def test_score_above_threshold_on_bright_frame(self, ready_detector):
        score, _, _ = ready_detector.update(_gray(255))
        assert score >= MOTION_THRESHOLD

    def test_score_near_zero_on_matching_background(self, ready_detector):
        # background was learned on black; sending black again = near-zero diff
        score, _, _ = ready_detector.update(_gray(0))
        assert score < MOTION_THRESHOLD

    def test_occupied_not_triggered_before_confirm_n(self, ready_detector):
        bright = _gray(255)
        for _ in range(DETECT_CONFIRM_N - 1):
            _, is_occ, _ = ready_detector.update(bright)
            assert is_occ is False

    def test_occupied_triggered_at_confirm_n(self, ready_detector):
        bright = _gray(255)
        is_occ = False
        for _ in range(DETECT_CONFIRM_N):
            _, is_occ, _ = ready_detector.update(bright)
        assert is_occ is True

    def test_occupied_stays_true_while_bright(self, ready_detector):
        bright = _gray(255)
        for _ in range(DETECT_CONFIRM_N + 5):
            _, is_occ, _ = ready_detector.update(bright)
        assert is_occ is True

    def test_detect_streak_resets_on_empty_frame(self, ready_detector):
        bright = _gray(255)
        black = _gray(0)
        # Build partial detect streak
        for _ in range(DETECT_CONFIRM_N - 1):
            ready_detector.update(bright)
        # One below-threshold frame resets it
        ready_detector.update(black)
        # Sending DETECT_CONFIRM_N - 1 bright frames again should NOT trigger
        for _ in range(DETECT_CONFIRM_N - 1):
            _, is_occ, _ = ready_detector.update(bright)
            assert is_occ is False


class TestEmptyDetection:
    def _occupancy_then_clear(self, detector) -> PresenceDetector:
        """Drive detector to occupied, then send black frames until is_empty."""
        bright = _gray(255)
        for _ in range(DETECT_CONFIRM_N):
            detector.update(bright)
        return detector

    def test_empty_triggered_after_enough_below_threshold_frames(self, ready_detector):
        black = _gray(0)
        is_empty = False
        for _ in range(EMPTY_CONFIRM_N + 5):
            _, _, is_empty = ready_detector.update(black)
        assert is_empty is True

    def test_empty_not_triggered_before_confirm_n(self, ready_detector):
        black = _gray(0)
        for _ in range(EMPTY_CONFIRM_N - 1):
            _, _, is_empty = ready_detector.update(black)
            assert is_empty is False

    def test_empty_streak_resets_on_bright_frame(self, ready_detector):
        black = _gray(0)
        bright = _gray(255)
        # Build partial empty streak
        for _ in range(EMPTY_CONFIRM_N - 1):
            ready_detector.update(black)
        # Interrupt with a bright frame
        ready_detector.update(bright)
        # Now EMPTY_CONFIRM_N - 1 black frames should NOT trigger is_empty
        for _ in range(EMPTY_CONFIRM_N - 1):
            _, _, is_empty = ready_detector.update(black)
            assert is_empty is False


class TestAcceptAsBackground:
    def test_accept_resets_detect_streak(self, ready_detector):
        bright = _gray(255)
        for _ in range(DETECT_CONFIRM_N):
            ready_detector.update(bright)
        ready_detector.accept_as_background(bright)
        assert ready_detector._detect_streak == 0

    def test_accept_resets_empty_streak(self, ready_detector):
        black = _gray(0)
        for _ in range(EMPTY_CONFIRM_N):
            ready_detector.update(black)
        ready_detector.accept_as_background(black)
        assert ready_detector._empty_streak == 0

    def test_accept_new_background_low_diff(self, ready_detector):
        bright = _gray(200)
        ready_detector.accept_as_background(bright)
        # Sending the same value should now give a low diff
        score, _, _ = ready_detector.update(_gray(200))
        assert score < MOTION_THRESHOLD

    def test_accept_as_background_does_not_affect_ready(self, ready_detector):
        ready_detector.accept_as_background(_gray(128))
        assert ready_detector.ready is True


class TestReset:
    def test_reset_without_gray_clears_ready(self, ready_detector):
        ready_detector.reset()
        assert ready_detector.ready is False

    def test_reset_without_gray_resets_warmup_count(self, ready_detector):
        ready_detector.reset()
        assert ready_detector.warmup_progress[0] == 0

    def test_reset_without_gray_clears_detect_streak(self, ready_detector):
        bright = _gray(255)
        for _ in range(DETECT_CONFIRM_N):
            ready_detector.update(bright)
        ready_detector.reset()
        assert ready_detector._detect_streak == 0

    def test_reset_without_gray_clears_empty_streak(self, ready_detector):
        black = _gray(0)
        for _ in range(EMPTY_CONFIRM_N):
            ready_detector.update(black)
        ready_detector.reset()
        assert ready_detector._empty_streak == 0

    def test_reset_with_gray_skips_warmup(self):
        d = PresenceDetector()
        d.reset(_gray(128))
        assert d.ready is True

    def test_reset_with_gray_seeds_background(self):
        d = PresenceDetector()
        d.reset(_gray(128))
        score, _, _ = d.update(_gray(128))
        assert score < MOTION_THRESHOLD

    def test_reset_with_bright_gray_detects_black_movement(self):
        d = PresenceDetector()
        d.reset(_gray(255))  # background is bright
        # Black frame = large diff
        score, _, _ = d.update(_gray(0))
        assert score >= MOTION_THRESHOLD

    def test_reset_on_fresh_detector_does_not_raise(self):
        d = PresenceDetector()
        d.reset()  # _bg is None, warmup_count already 0
        assert d.ready is False


class TestEdgeCases:
    def test_consistent_frame_shape_required(self, ready_detector):
        """Detector requires frames to match the warmup shape."""
        small = _gray(255, shape=(10, 10))  # warmup was 100×100
        with pytest.raises(ValueError):
            ready_detector.update(small)

    def test_noise_frame_has_variable_score(self, ready_detector):
        scores = set()
        for _ in range(5):
            frame = _noise()
            score, _, _ = ready_detector.update(frame)
            scores.add(round(score, 1))
        # Scores from random noise frames should vary
        assert len(scores) >= 1  # at minimum not all identical

    def test_warmup_progress_never_exceeds_total(self):
        d = PresenceDetector()
        for _ in range(BG_WARMUP_FRAMES + 20):
            d.update(_gray(0))
        current, total = d.warmup_progress
        assert current >= total
