"""
smartwaste/utils.py — Shared frame helpers used by main.py and mainauto.py.
"""

import threading

import cv2
import numpy as np

from .classifier import classify
from .config import JPEG_QUALITY
from .log_setup import get_logger
from .state import AppState

logger = get_logger()


def encode_frame(frame) -> bytes | None:
    """JPEG-encode a BGR frame. Returns bytes or None on failure."""
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return enc.tobytes() if ok else None


def launch_classify(img_bytes: bytes | None, frame_copy, state: AppState) -> None:
    """Start a daemon thread that calls Gemini with *img_bytes*."""
    if img_bytes:
        threading.Thread(
            target=classify,
            args=(img_bytes, frame_copy, state),
            daemon=True,
        ).start()
    else:
        state.set_status("Error", "JPEG encode failed.")
        state.finish_classify()


def crop_sides(frame: np.ndarray, crop_percent: float) -> np.ndarray:
    """Remove crop_percent fraction from both left and right sides."""
    if crop_percent <= 0:
        return frame
    h, w = frame.shape[:2]
    left = int(w * crop_percent)
    right = int(w * (1 - crop_percent))
    return frame[:, left:right]
