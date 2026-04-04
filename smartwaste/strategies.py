"""
smartwaste/strategies.py — classification-trigger strategies for run_loop().

ManualStrategy        — 'c' key + optional auto-classify timer ('a' to toggle).
PresenceGateStrategy  — pixel-diff presence detection gates every API call.
"""

import time

import cv2
import numpy as np

from .app import Strategy
from .config import AUTO_INTERVAL, CHECK_INTERVAL
from .log_setup import get_logger
from .presence import PresenceDetector
from .state import AppState
from .utils import encode_frame, launch_classify

logger = get_logger()


class ManualStrategy(Strategy):
    """
    Original manual-mode behaviour.

    Keys
    ----
    c — classify current frame once.
    a — toggle auto-classify (fires every AUTO_INTERVAL seconds).
    """

    def on_combined_frame(self, combined: np.ndarray, state: AppState) -> None:
        now = time.time()
        if (
            state.auto_classify
            and now - state.last_capture_time >= AUTO_INTERVAL
            and state.start_classify()
        ):
            state.last_capture_time = now
            launch_classify(encode_frame(combined), combined.copy(), state)

    def on_key(self, key: int, combined: np.ndarray | None, state: AppState) -> None:
        if key == ord("c") and combined is not None and state.start_classify():
            launch_classify(encode_frame(combined), combined.copy(), state)

        elif key == ord("a"):
            auto_on = state.toggle_auto()
            state.set_status("Ready", "Auto ON" if auto_on else "Auto OFF (manual: press 'c')")
            logger.info("AUTO_CLASSIFY=%s", auto_on)


class PresenceGateStrategy(Strategy):
    """
    Auto-gate mode: local pixel-diff detection gates every Gemini API call.

    State machine
    -------------
    Calibrating → Ready/IDLE → Detected → Classified → Ready/IDLE …

    Keys
    ----
    c — force-classify current frame (manual override).
    r — reset the background model from the current frame.
    """

    def __init__(self) -> None:
        self._detector = PresenceDetector()
        self._last_check_time = 0.0
        self._bin_occupied = False
        self._item_classified = False

    def setup(self, state: AppState) -> None:
        state.auto_classify = True  # always on in auto-gate mode

    def on_combined_frame(self, combined: np.ndarray, state: AppState) -> None:
        now = time.time()
        if now - self._last_check_time < CHECK_INTERVAL:
            return
        self._last_check_time = now

        gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
        score, is_occupied, is_empty = self._detector.update(gray)

        if not self._detector.ready:
            done, total = self._detector.warmup_progress
            state.set_status(
                "Calibrating",
                f"Learning empty-bin background... ({done}/{total})",
            )

        elif is_empty and self._bin_occupied:
            # Bin cleared → reset for next item
            self._bin_occupied = False
            self._item_classified = False
            self._detector.accept_as_background(gray)
            state.set_status("Ready", "Bin is empty — waiting for item.")
            logger.info("Bin cleared. Background updated. score=%.1f", score)

        elif is_occupied and not self._bin_occupied:
            # Object just entered bin
            self._bin_occupied = True
            self._item_classified = False
            state.set_status(
                "Detected",
                f"Object detected (score={score:.1f}) — classifying...",
            )
            logger.info("Object detected. score=%.1f → API call queued", score)

        # Fire exactly one API call per item arrival
        if self._bin_occupied and not self._item_classified and state.start_classify():
            self._item_classified = True
            logger.info("Gemini API call triggered. score=%.1f", score)
            launch_classify(encode_frame(combined), combined.copy(), state)

    def on_key(self, key: int, combined: np.ndarray | None, state: AppState) -> None:
        if key == ord("c") and combined is not None and state.start_classify():
            self._item_classified = True
            logger.info("Manual classify triggered.")
            launch_classify(encode_frame(combined), combined.copy(), state)

        elif key == ord("r") and combined is not None:
            gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
            self._detector.reset(gray)
            self._bin_occupied = False
            self._item_classified = False
            state.set_status("Ready", "Background reset from current frame.")
            logger.info("Background manually reset.")
