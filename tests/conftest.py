"""
conftest.py — shared fixtures and environment setup.

Must run before any smartwaste module is imported, because several modules
execute side-effects at import time (DB init, GEMINI_API_KEY check, etc.).
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── 1. Satisfy GEMINI_API_KEY check in classifier.py ─────────────────────────
os.environ.setdefault("GEMINI_API_KEY", "test-key-for-pytest")

# ── 1b. Default to SQLite for tests (no PostgreSQL required) ─────────────────
os.environ.setdefault("SMARTWASTE_DB_BACKEND", "sqlite")

# ── 2. Mock hardware-specific packages that may not be installed ──────────────
for _mod in ("depthai", "picamera2"):
    sys.modules.setdefault(_mod, MagicMock())


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def blank_bgr():
    """480×640 black BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_bgr():
    """480×640 white BGR frame."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def blank_gray():
    """480×640 black grayscale frame."""
    return np.zeros((480, 640), dtype=np.uint8)


@pytest.fixture
def white_gray():
    """480×640 white grayscale frame."""
    return np.full((480, 640), 255, dtype=np.uint8)


@pytest.fixture
def app_state():
    from smartwaste.state import AppState

    return AppState()


@pytest.fixture
def ready_detector():
    """PresenceDetector that has completed warmup on a black frame."""
    from smartwaste.presence import PresenceDetector
    from smartwaste.config import BG_WARMUP_FRAMES
    import numpy as np

    d = PresenceDetector()
    black = np.zeros((100, 100), dtype=np.uint8)
    for _ in range(BG_WARMUP_FRAMES):
        d.update(black)
    return d
