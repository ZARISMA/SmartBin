"""
smartwaste/oak_native.py — OAK multi-sensor occupancy detector.

Builds a depthai 3.x pipeline that runs on a single OAK-1 W / OAK-1 Lite
device (no stereo depth, no IMU):

  * Presence   — pixel-diff background model detects when an item enters the bin
  * Motion     — sudden score spike detects the moment an item is dropped
  * MobileNetSSD — object detection running on the Myriad X VPU (on-device)

Public API::

    detector = OAKOccupancyDetector(device)
    done = detector.calibrate()   # call each loop tick; returns True when ready
    votes = detector.update()     # drain queues, return SensorVotes
    detector.reset()              # restart calibration
    detector.stop()               # clean up pipeline
"""

from __future__ import annotations

import time
from typing import NamedTuple

import cv2
import depthai as dai
import numpy as np

from .config import (
    DROP_FLAG_DURATION,
    MOTION_SPIKE_FACTOR,
    MOTION_THRESHOLD,
    NN_CONFIDENCE,
    NN_MODEL_NAME,
    NN_SHAVES,
)
from .log_setup import get_logger
from .presence import PresenceDetector

logger = get_logger()


# ── Result types ───────────────────────────────────────────────────────────────


class Detection(NamedTuple):
    """Single NN bounding-box detection (normalised 0-1 coordinates)."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    label_idx: int


class SensorVotes(NamedTuple):
    presence_occupied: bool
    motion_spike: bool
    nn_occupied: bool
    votes: int  # sum of the three booleans above
    rgb_frame: np.ndarray | None  # latest RGB frame (getCvFrame), or None
    presence_score: float  # pixel-diff score (0-255)
    motion_delta: float  # score jump magnitude
    nn_count: int  # number of NN detections above threshold
    nn_detections: list[Detection]  # bounding boxes for display


# ── Pipeline construction ──────────────────────────────────────────────────────


def _try_get_blob() -> tuple[str | None, bool]:
    """Download MobileNetSSD blob via blobconverter. Returns (path, available)."""
    try:
        import blobconverter  # type: ignore[import]

        path = blobconverter.from_zoo(
            name=NN_MODEL_NAME,
            shaves=NN_SHAVES,
            version="2022.1",
        )
        logger.info("NN blob ready: %s", path)
        return path, True
    except Exception as exc:
        logger.warning("MobileNetSSD blob unavailable (%s); NN sensor disabled.", exc)
        return None, False


def build_oak_pipeline(device: dai.Device) -> tuple:
    """
    Create and start a depthai 3.x pipeline on *device*.

    Designed for OAK-1 W / OAK-1 Lite (single RGB camera + Myriad X VPU).
    No stereo depth or IMU.

    Returns
    -------
    (pipeline, rgb_q, nn_q, nn_available)

    *nn_q* may be None when the NN could not be initialised.
    """
    blob_path, nn_available = _try_get_blob()

    pipeline = dai.Pipeline(device)

    # ── RGB camera (CAM_A) ─────────────────────────────────────────────────────
    cam_a = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_q = cam_a.requestFullResolutionOutput(type=dai.ImgFrame.Type.BGR888p).createOutputQueue(
        maxSize=4, blocking=False
    )

    # ── Neural network (MobileNetSSD on Myriad X) ──────────────────────────────
    nn_q = None
    if nn_available and blob_path is not None:
        try:
            # Resize RGB preview to 300×300 for MobileNetSSD input
            nn_in = cam_a.requestOutput((300, 300), type=dai.ImgFrame.Type.BGR888p)
            det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
            det_nn.setConfidenceThreshold(NN_CONFIDENCE)
            det_nn.setBlobPath(blob_path)
            det_nn.setNumInferenceThreads(2)
            det_nn.input.setBlocking(False)
            nn_in.link(det_nn.input)
            nn_q = det_nn.out.createOutputQueue(maxSize=4, blocking=False)
            logger.info("NN pipeline ready (MobileNetSSD).")
        except Exception as exc:
            logger.warning("NN pipeline failed (%s); NN sensor disabled.", exc)
            nn_available = False

    pipeline.start()
    logger.info(
        "OAK pipeline started — NN=%s  (presence + motion via software)",
        nn_q is not None,
    )
    return pipeline, rgb_q, nn_q, nn_available


# ── Detector ───────────────────────────────────────────────────────────────────


class OAKOccupancyDetector:
    """
    Multi-sensor bin occupancy detector using pixel-diff presence, motion
    spike detection, and MobileNetSSD NN.

    Designed for OAK-1 W / OAK-1 Lite (no stereo depth, no IMU).

    Usage::

        detector = OAKOccupancyDetector(device)
        while not detector.calibrate():
            pass                        # warmup
        votes = detector.update()       # call every loop iteration
    """

    def __init__(self, device: dai.Device) -> None:
        (
            self._pipeline,
            self._rgb_q,
            self._nn_q,
            self._nn_available,
        ) = build_oak_pipeline(device)

        # Presence detector (pixel-diff background model)
        self._presence = PresenceDetector()

        # Motion spike state (replaces IMU drop detection)
        self._last_presence_score: float = 0.0
        self._prev_presence_score: float = 0.0
        self._motion_spike: bool = False
        self._motion_spike_expiry: float = 0.0

        # Latest cached NN values
        self._last_nn_count: int = 0
        self._last_nn_detections: list[Detection] = []

        # Cache the last RGB frame during calibration for use after warmup
        self._last_rgb: np.ndarray | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def calibrate(self) -> bool:
        """
        Feed frames into the presence detector during warmup.

        Returns True once the background model is ready; False while still
        warming up.  Call once per main-loop tick until it returns True.
        """
        rgb = self._drain_rgb()
        if rgb is not None:
            self._last_rgb = rgb
            if not self._presence.ready:
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                self._presence.update(gray)

        return self._presence.ready

    def update(self) -> SensorVotes:
        """
        Drain all queues and return the current sensor votes.

        Should be called on every main-loop iteration (non-blocking).
        """
        rgb_frame = self._drain_rgb()

        presence_occupied = False
        if rgb_frame is not None:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            score, is_occupied, _ = self._presence.update(gray)
            self._prev_presence_score = self._last_presence_score
            self._last_presence_score = score
            presence_occupied = is_occupied

            # Motion spike detection (replaces IMU drop)
            score_jump = score - self._prev_presence_score
            spike_threshold = MOTION_THRESHOLD * MOTION_SPIKE_FACTOR
            if score_jump > spike_threshold:
                self._motion_spike = True
                self._motion_spike_expiry = time.time() + DROP_FLAG_DURATION
                logger.debug("Motion spike: jump=%.1f (threshold=%.1f)", score_jump, spike_threshold)

        # Auto-expire motion spike
        if self._motion_spike and time.time() >= self._motion_spike_expiry:
            self._motion_spike = False

        nn_count, nn_detections = self._update_nn()

        votes = int(presence_occupied) + int(self._motion_spike) + int(nn_count > 0)

        return SensorVotes(
            presence_occupied=presence_occupied,
            motion_spike=self._motion_spike,
            nn_occupied=nn_count > 0,
            votes=votes,
            rgb_frame=rgb_frame,
            presence_score=self._last_presence_score,
            motion_delta=self._last_presence_score - self._prev_presence_score,
            nn_count=nn_count,
            nn_detections=nn_detections,
        )

    def calibration_progress(self) -> int:
        """Return estimated calibration progress as a percentage (0-100)."""
        current, total = self._presence.warmup_progress
        return min(100, current * 100 // max(total, 1))

    def reset(self) -> None:
        """Clear all calibration state so calibrate() starts fresh."""
        self._presence.reset()
        self._last_presence_score = 0.0
        self._prev_presence_score = 0.0
        self._motion_spike = False
        self._motion_spike_expiry = 0.0
        self._last_nn_count = 0
        self._last_nn_detections = []
        logger.info("OAKOccupancyDetector reset — recalibrating.")

    def stop(self) -> None:
        """Stop the depthai pipeline cleanly."""
        try:
            self._pipeline.stop()
        except Exception:
            pass

    @property
    def presence_ready(self) -> bool:
        return self._presence.ready

    @property
    def nn_available(self) -> bool:
        return bool(self._nn_available)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _drain_rgb(self) -> np.ndarray | None:
        """Return most-recent RGB frame, or None if no frame available."""
        frame = None
        while self._rgb_q.has():
            frame = self._rgb_q.get().getCvFrame()
        return frame

    def _update_nn(self) -> tuple[int, list[Detection]]:
        """Drain NN queue; return count and bounding boxes of detections."""
        if self._nn_q is None:
            return self._last_nn_count, self._last_nn_detections

        count = None
        detections: list[Detection] = []
        while self._nn_q.has():
            dets = self._nn_q.get().detections
            count = len(dets)
            detections = [
                Detection(d.xmin, d.ymin, d.xmax, d.ymax, d.confidence, d.label)
                for d in dets
            ]

        if count is not None:
            self._last_nn_count = count
            self._last_nn_detections = detections
        return self._last_nn_count, self._last_nn_detections
