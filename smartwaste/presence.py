"""
smartwaste/presence.py — Local bin-occupancy detector (no API calls).

Uses a rolling background model (cv2.accumulateWeighted) to decide whether
an object is present in the bin purely from pixel-diff scoring.
"""

import cv2
import numpy as np

from .config import (
    BG_LEARNING_RATE,
    BG_WARMUP_FRAMES,
    DETECT_CONFIRM_N,
    EMPTY_CONFIRM_N,
    MOTION_THRESHOLD,
)


class PresenceDetector:
    """
    Maintains a rolling background model and reports whether an object
    is present in the bin based on per-frame pixel-diff scoring.

    Typical usage::

        detector = PresenceDetector()
        ...
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, is_occupied, is_empty = detector.update(gray)
    """

    def __init__(self) -> None:
        self._bg: np.ndarray | None = None
        self._warmup_count  = 0
        self._detect_streak = 0   # consecutive checks above threshold
        self._empty_streak  = 0   # consecutive checks below threshold

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """True once the warmup phase is complete."""
        return self._warmup_count >= BG_WARMUP_FRAMES

    @property
    def warmup_progress(self) -> tuple[int, int]:
        """(current, total) warmup frame counts."""
        return self._warmup_count, BG_WARMUP_FRAMES

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, gray: np.ndarray) -> tuple[float, bool, bool]:
        """
        Feed a new grayscale frame.

        Returns:
            score       — mean absolute diff vs background (0–255)
            is_occupied — True when detect streak reaches DETECT_CONFIRM_N
            is_empty    — True when empty streak reaches EMPTY_CONFIRM_N
        """
        gray_f = gray.astype(np.float32)

        if self._bg is None:
            self._bg = gray_f.copy()

        # Warmup: fast-learn background, never fire detections
        if not self.ready:
            cv2.accumulateWeighted(gray_f, self._bg, 0.15)
            self._warmup_count += 1
            return 0.0, False, False

        score = float(np.abs(gray_f - self._bg).mean())

        if score >= MOTION_THRESHOLD:
            self._detect_streak += 1
            self._empty_streak   = 0
        else:
            self._empty_streak  += 1
            self._detect_streak  = 0
            # Slowly drift background toward current frame only when bin is empty
            cv2.accumulateWeighted(gray_f, self._bg, BG_LEARNING_RATE)

        is_occupied = self._detect_streak >= DETECT_CONFIRM_N
        is_empty    = self._empty_streak  >= EMPTY_CONFIRM_N
        return score, is_occupied, is_empty

    def accept_as_background(self, gray: np.ndarray) -> None:
        """Hard-snap background to current frame (call when bin confirmed empty)."""
        self._bg = gray.astype(np.float32)
        self._detect_streak = 0
        self._empty_streak  = 0

    def reset(self, gray: np.ndarray | None = None) -> None:
        """
        Full reset of the detector.

        If *gray* is provided the background is seeded from it and warmup is
        skipped; otherwise a fresh warmup cycle starts from scratch.
        """
        self._bg = gray.astype(np.float32) if gray is not None else None
        self._warmup_count  = BG_WARMUP_FRAMES if gray is not None else 0
        self._detect_streak = 0
        self._empty_streak  = 0
