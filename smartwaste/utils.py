"""
smartwaste/utils.py — Shared frame helpers used by main.py and mainauto.py.
"""

import threading

import cv2

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
