"""Tests for smartwaste/strategies.py — ManualStrategy and PresenceGateStrategy."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smartwaste.state import AppState
from smartwaste.strategies import ManualStrategy, PresenceGateStrategy


def _combined() -> np.ndarray:
    """Minimal valid BGR frame (two DISPLAY_SIZE frames side-by-side)."""
    return np.zeros((480, 1280, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# ManualStrategy
# ─────────────────────────────────────────────────────────────────────────────


class TestManualStrategySetup:
    def test_setup_leaves_auto_classify_false(self):
        state = AppState()
        ManualStrategy().setup(state)
        assert state.auto_classify is False

    def test_setup_does_not_change_label(self):
        state = AppState()
        ManualStrategy().setup(state)
        label, _, _ = state.get_display()
        assert label == "Ready"


class TestManualStrategyOnCombinedFrame:
    def test_no_trigger_when_auto_off(self):
        state = AppState()
        state.auto_classify = False
        with patch("smartwaste.strategies.launch_classify") as mock:
            ManualStrategy().on_combined_frame(_combined(), state)
        assert not mock.called

    def test_triggers_when_auto_on_and_interval_passed(self):
        state = AppState()
        state.auto_classify = True
        state.last_capture_time = 0.0  # far in the past
        with (
            patch("smartwaste.strategies.launch_classify") as mock,
            patch("smartwaste.strategies.encode_frame", return_value=b"x"),
        ):
            ManualStrategy().on_combined_frame(_combined(), state)
        assert mock.called

    def test_does_not_trigger_within_interval(self):
        state = AppState()
        state.auto_classify = True
        state.last_capture_time = time.time()  # just now
        with patch("smartwaste.strategies.launch_classify") as mock:
            ManualStrategy().on_combined_frame(_combined(), state)
        assert not mock.called

    def test_does_not_double_trigger_when_busy(self):
        state = AppState()
        state.auto_classify = True
        state.last_capture_time = 0.0
        state.start_classify()  # mark as busy
        with patch("smartwaste.strategies.launch_classify") as mock:
            ManualStrategy().on_combined_frame(_combined(), state)
        assert not mock.called

    def test_updates_last_capture_time_on_trigger(self):
        state = AppState()
        state.auto_classify = True
        state.last_capture_time = 0.0
        before = time.time()
        with (
            patch("smartwaste.strategies.launch_classify"),
            patch("smartwaste.strategies.encode_frame", return_value=b"x"),
        ):
            ManualStrategy().on_combined_frame(_combined(), state)
        assert state.last_capture_time >= before


class TestManualStrategyOnKey:
    def test_c_key_triggers_classify(self):
        state = AppState()
        with (
            patch("smartwaste.strategies.launch_classify") as mock,
            patch("smartwaste.strategies.encode_frame", return_value=b"x"),
        ):
            ManualStrategy().on_key(ord("c"), _combined(), state)
        assert mock.called

    def test_c_key_does_nothing_when_busy(self):
        state = AppState()
        state.start_classify()  # mark busy
        with patch("smartwaste.strategies.launch_classify") as mock:
            ManualStrategy().on_key(ord("c"), _combined(), state)
        assert not mock.called

    def test_c_key_does_nothing_when_combined_is_none(self):
        state = AppState()
        with patch("smartwaste.strategies.launch_classify") as mock:
            ManualStrategy().on_key(ord("c"), None, state)
        assert not mock.called

    def test_a_key_toggles_auto_on(self):
        state = AppState()
        ManualStrategy().on_key(ord("a"), None, state)
        assert state.auto_classify is True

    def test_a_key_toggles_auto_off(self):
        state = AppState()
        state.auto_classify = True
        ManualStrategy().on_key(ord("a"), None, state)
        assert state.auto_classify is False

    def test_a_key_updates_status(self):
        state = AppState()
        ManualStrategy().on_key(ord("a"), None, state)
        _, detail, _ = state.get_display()
        assert "Auto ON" in detail

    def test_unknown_key_is_ignored(self):
        state = AppState()
        ManualStrategy().on_key(ord("z"), _combined(), state)  # must not raise

    def test_r_key_ignored_in_manual_strategy(self):
        state = AppState()
        ManualStrategy().on_key(ord("r"), _combined(), state)  # no-op, must not raise


# ─────────────────────────────────────────────────────────────────────────────
# PresenceGateStrategy
# ─────────────────────────────────────────────────────────────────────────────


class TestPresenceGateStrategySetup:
    def test_setup_sets_auto_classify_true(self):
        state = AppState()
        PresenceGateStrategy().setup(state)
        assert state.auto_classify is True

    def test_initial_bin_occupied_false(self):
        s = PresenceGateStrategy()
        assert s._bin_occupied is False

    def test_initial_item_classified_false(self):
        s = PresenceGateStrategy()
        assert s._item_classified is False


class TestPresenceGateStrategyOnKey:
    def test_c_key_triggers_classify(self):
        state = AppState()
        with (
            patch("smartwaste.strategies.launch_classify") as mock,
            patch("smartwaste.strategies.encode_frame", return_value=b"x"),
        ):
            PresenceGateStrategy().on_key(ord("c"), _combined(), state)
        assert mock.called

    def test_c_key_sets_item_classified(self):
        state = AppState()
        s = PresenceGateStrategy()
        with (
            patch("smartwaste.strategies.launch_classify"),
            patch("smartwaste.strategies.encode_frame", return_value=b"x"),
        ):
            s.on_key(ord("c"), _combined(), state)
        assert s._item_classified is True

    def test_c_key_does_nothing_when_combined_none(self):
        state = AppState()
        with patch("smartwaste.strategies.launch_classify") as mock:
            PresenceGateStrategy().on_key(ord("c"), None, state)
        assert not mock.called

    def test_c_key_does_nothing_when_busy(self):
        state = AppState()
        state.start_classify()
        with patch("smartwaste.strategies.launch_classify") as mock:
            PresenceGateStrategy().on_key(ord("c"), _combined(), state)
        assert not mock.called

    def test_r_key_resets_bin_occupied(self):
        state = AppState()
        s = PresenceGateStrategy()
        s._bin_occupied = True
        s.on_key(ord("r"), _combined(), state)
        assert s._bin_occupied is False

    def test_r_key_resets_item_classified(self):
        state = AppState()
        s = PresenceGateStrategy()
        s._item_classified = True
        s.on_key(ord("r"), _combined(), state)
        assert s._item_classified is False

    def test_r_key_updates_status(self):
        state = AppState()
        PresenceGateStrategy().on_key(ord("r"), _combined(), state)
        label, _, _ = state.get_display()
        assert label == "Ready"

    def test_r_key_does_nothing_when_combined_none(self):
        state = AppState()
        s = PresenceGateStrategy()
        s._bin_occupied = True
        s.on_key(ord("r"), None, state)
        assert s._bin_occupied is True  # unchanged


class TestPresenceGateStrategyOnCombinedFrame:
    def test_respects_check_interval(self):
        state = AppState()
        s = PresenceGateStrategy()
        s._last_check_time = time.time()  # just now — interval not passed
        with patch("smartwaste.strategies.launch_classify") as mock:
            s.on_combined_frame(_combined(), state)
        assert not mock.called

    def test_sets_calibrating_status_during_warmup(self):
        state = AppState()
        s = PresenceGateStrategy()
        s._last_check_time = 0.0  # force check
        # Detector starts in warmup — first update returns (0, False, False)
        s.on_combined_frame(_combined(), state)
        label, _, _ = state.get_display()
        assert label == "Calibrating"

    def test_calibrating_detail_shows_progress(self):
        state = AppState()
        s = PresenceGateStrategy()
        s._last_check_time = 0.0
        s.on_combined_frame(_combined(), state)
        _, detail, _ = state.get_display()
        assert "/" in detail  # e.g. "1/40"

    def test_does_not_trigger_api_during_warmup(self):
        state = AppState()
        s = PresenceGateStrategy()
        s._last_check_time = 0.0
        with patch("smartwaste.strategies.launch_classify") as mock:
            s.on_combined_frame(_combined(), state)
        assert not mock.called

    def test_fires_api_when_occupied_and_not_classified(self):
        """Force detector to ready state and simulate occupied bin."""
        state = AppState()
        s = PresenceGateStrategy()
        s._last_check_time = 0.0

        # Skip warmup by seeding detector with a black background
        gray_seed = np.zeros((480, 1280), dtype=np.uint8)
        s._detector.reset(gray_seed)  # ready=True, background=black

        # Send a bright frame → large diff → occupied
        bright = np.full((480, 1280, 3), 200, dtype=np.uint8)

        from smartwaste.config import DETECT_CONFIRM_N

        with (
            patch("smartwaste.strategies.launch_classify") as mock,
            patch("smartwaste.strategies.encode_frame", return_value=b"x"),
        ):
            for _ in range(DETECT_CONFIRM_N + 1):
                s._last_check_time = 0.0
                s.on_combined_frame(bright, state)

        assert mock.called

    def test_fires_api_only_once_per_item(self):
        """A second call while bin_occupied should not re-fire."""
        state = AppState()
        s = PresenceGateStrategy()
        s._last_check_time = 0.0
        gray_seed = np.zeros((480, 1280), dtype=np.uint8)
        s._detector.reset(gray_seed)
        bright = np.full((480, 1280, 3), 200, dtype=np.uint8)

        from smartwaste.config import DETECT_CONFIRM_N

        call_count = []
        with (
            patch(
                "smartwaste.strategies.launch_classify",
                side_effect=lambda *a, **kw: call_count.append(1),
            ),
            patch("smartwaste.strategies.encode_frame", return_value=b"x"),
        ):
            for _ in range(DETECT_CONFIRM_N + 5):
                s._last_check_time = 0.0
                # Simulate finish_classify so start_classify can succeed again
                state.finish_classify()
                s.on_combined_frame(bright, state)

        assert len(call_count) == 1

    def test_bin_cleared_resets_flags(self):
        """After is_empty fires, bin_occupied and item_classified reset."""
        state = AppState()
        s = PresenceGateStrategy()
        s._bin_occupied = True
        s._item_classified = True
        s._last_check_time = 0.0

        gray_seed = np.zeros((480, 1280), dtype=np.uint8)
        s._detector.reset(gray_seed)

        from smartwaste.config import EMPTY_CONFIRM_N

        black = np.zeros((480, 1280, 3), dtype=np.uint8)
        with (
            patch("smartwaste.strategies.launch_classify"),
            patch("smartwaste.strategies.encode_frame", return_value=b"x"),
        ):
            for _ in range(EMPTY_CONFIRM_N + 1):
                s._last_check_time = 0.0
                s.on_combined_frame(black, state)

        assert s._bin_occupied is False
        assert s._item_classified is False
