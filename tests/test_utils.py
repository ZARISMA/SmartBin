"""Tests for smartwaste/utils.py — encode_frame() and launch_classify()."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smartwaste.state import AppState
from smartwaste.utils import encode_frame, launch_classify


# ─────────────────────────────────────────────────────────────────────────────
# encode_frame
# ─────────────────────────────────────────────────────────────────────────────


class TestEncodeFrame:
    def test_returns_bytes_for_valid_bgr_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = encode_frame(frame)
        assert isinstance(result, bytes)

    def test_returns_nonempty_bytes(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert len(encode_frame(frame)) > 0

    def test_jpeg_magic_bytes(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = encode_frame(frame)
        assert result[:2] == b"\xff\xd8"  # JPEG SOI marker

    def test_white_frame_encodes(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert encode_frame(frame) is not None

    def test_gradient_frame_encodes(self):
        frame = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
        frame = np.stack([frame, frame, frame], axis=-1)[:100, :100]
        assert encode_frame(frame) is not None

    def test_large_frame_encodes(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert encode_frame(frame) is not None

    def test_tiny_frame_encodes(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        assert encode_frame(frame) is not None

    def test_returns_none_when_imencode_fails(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch("cv2.imencode", return_value=(False, None)):
            assert encode_frame(frame) is None

    def test_higher_quality_produces_larger_output(self):
        """Verify encode_frame uses the configured JPEG_QUALITY."""
        import cv2
        from smartwaste.config import JPEG_QUALITY

        frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = encode_frame(frame)
        # Encode the same frame at quality 1 manually
        _, low_q = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 1])
        # Our configured quality (85) should produce larger file than quality 1
        if JPEG_QUALITY > 1:
            assert len(result) >= len(low_q.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# launch_classify
# ─────────────────────────────────────────────────────────────────────────────


class TestLaunchClassify:
    def test_none_bytes_sets_error_label(self):
        state = AppState()
        launch_classify(None, np.zeros((10, 10, 3), dtype=np.uint8), state)
        label, _, _ = state.get_display()
        assert label == "Error"

    def test_none_bytes_sets_encode_failed_detail(self):
        state = AppState()
        launch_classify(None, np.zeros((10, 10, 3), dtype=np.uint8), state)
        _, detail, _ = state.get_display()
        assert "encode" in detail.lower() or "failed" in detail.lower()

    def test_none_bytes_calls_finish_classify(self):
        state = AppState()
        state.start_classify()
        launch_classify(None, np.zeros((10, 10, 3), dtype=np.uint8), state)
        # finish_classify was called, so start_classify should work again
        assert state.start_classify() is True

    def test_valid_bytes_starts_daemon_thread(self):
        state = AppState()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        img_bytes = b"\xff\xd8\xff" + b"\x00" * 50

        started = []
        with patch("smartwaste.utils.threading.Thread") as mock_thread_cls:
            mock_instance = MagicMock()
            mock_thread_cls.return_value = mock_instance
            launch_classify(img_bytes, frame, state)
            assert mock_thread_cls.called
            assert mock_instance.start.called

    def test_valid_bytes_creates_daemon_thread(self):
        state = AppState()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        img_bytes = b"\xff\xd8\xff" + b"\x00" * 50

        with patch("smartwaste.utils.threading.Thread") as mock_thread_cls:
            mock_instance = MagicMock()
            mock_thread_cls.return_value = mock_instance
            launch_classify(img_bytes, frame, state)
            _, kwargs = mock_thread_cls.call_args
            assert kwargs.get("daemon") is True

    def test_valid_bytes_targets_classify(self):
        from smartwaste.classifier import classify

        state = AppState()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        img_bytes = b"\xff\xd8\xff" + b"\x00" * 50

        with patch("smartwaste.utils.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            launch_classify(img_bytes, frame, state)
            _, kwargs = mock_thread_cls.call_args
            assert kwargs.get("target") is classify

    def test_empty_bytes_treated_same_as_none(self):
        """b'' is falsy so it takes the error path, same as None."""
        state = AppState()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch("smartwaste.utils.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            launch_classify(b"", frame, state)
            assert not mock_thread_cls.called
            label, _, _ = state.get_display()
            assert label == "Error"
