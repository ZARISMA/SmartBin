"""
smartwaste/oak_native.py — OAK-D multi-sensor occupancy detector.

Builds a depthai 3.x pipeline that runs three sensors concurrently on a
single OAK-D device:

  * Stereo depth  — measures the distance change when an item enters the bin
  * IMU           — detects the physical shock when an item is dropped
  * MobileNetSSD  — object detection running on the Myriad X VPU (on-device)

Public API::

    detector = OAKOccupancyDetector(device)
    done = detector.calibrate()   # call each loop tick; returns True when ready
    votes = detector.update()     # drain queues, return SensorVotes
    detector.reset()              # restart calibration
    detector.stop()               # clean up pipeline
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple

import depthai as dai
import numpy as np

from .config import (
    DEPTH_CHANGE_THRESHOLD,
    DEPTH_ROI_FRACTION,
    DROP_FLAG_DURATION,
    IMU_BASELINE_SAMPLES,
    IMU_SAMPLE_RATE_HZ,
    IMU_SHOCK_THRESHOLD,
    NN_CONFIDENCE,
    NN_MODEL_NAME,
    NN_SHAVES,
    OAK_CALIB_FRAMES,
)
from .log_setup import get_logger

logger = get_logger()


# ── Result type ────────────────────────────────────────────────────────────────


class SensorVotes(NamedTuple):
    depth_occupied: bool
    drop_flag: bool
    nn_occupied: bool
    votes: int  # sum of the three booleans above
    rgb_frame: np.ndarray | None  # latest RGB frame (getCvFrame), or None
    depth_mm_delta: float  # baseline − current median depth (for display)
    imu_delta: float  # accel magnitude above quiet baseline (for display)
    nn_count: int  # number of NN detections above threshold


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
    Create and start an extended depthai 3.x pipeline on *device*.

    Returns
    -------
    (pipeline, rgb_q, depth_q, imu_q, nn_q, imu_available, nn_available)

    *depth_q*, *imu_q*, *nn_q* may be None when the corresponding sensor could
    not be initialised (graceful degradation).
    """
    blob_path, nn_available = _try_get_blob()

    pipeline = dai.Pipeline(device)

    # ── RGB camera (CAM_A) ─────────────────────────────────────────────────────
    cam_a = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_q = cam_a.requestFullResolutionOutput(type=dai.ImgFrame.Type.BGR888p).createOutputQueue(
        maxSize=4, blocking=False
    )

    # ── Stereo cameras (CAM_B / CAM_C) → StereoDepth ──────────────────────────
    depth_q = None
    try:
        cam_b = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_c = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        left_out = cam_b.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)
        right_out = cam_c.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetType.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setExtendedDisparity(True)  # halves minimum range (~0.4m → ~0.2m)
        stereo.setLeftRightCheck(True)  # filter invalid pixels
        stereo.setSubpixel(False)  # integer depth saves bandwidth

        left_out.link(stereo.left)
        right_out.link(stereo.right)

        depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        logger.info("Stereo depth pipeline ready.")
    except Exception as exc:
        logger.warning("Stereo depth unavailable (%s); depth sensor disabled.", exc)

    # ── IMU ────────────────────────────────────────────────────────────────────
    imu_q = None
    imu_available = False
    try:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, IMU_SAMPLE_RATE_HZ)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, IMU_SAMPLE_RATE_HZ)
        imu.setBatchReportThreshold(5)
        imu.setMaxBatchReports(20)
        imu_q = imu.out.createOutputQueue(maxSize=20, blocking=False)
        imu_available = True
        logger.info("IMU pipeline ready.")
    except Exception as exc:
        logger.warning("IMU unavailable (%s); IMU sensor disabled.", exc)

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
        "OAK-D pipeline started — depth=%s  IMU=%s  NN=%s",
        depth_q is not None,
        imu_available,
        nn_q is not None,
    )
    return pipeline, rgb_q, depth_q, imu_q, nn_q, imu_available, nn_available


# ── Detector ───────────────────────────────────────────────────────────────────


class OAKOccupancyDetector:
    """
    Multi-sensor bin occupancy detector using OAK-D stereo depth, IMU, and NN.

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
            self._depth_q,
            self._imu_q,
            self._nn_q,
            self._imu_available,
            self._nn_available,
        ) = build_oak_pipeline(device)

        # Calibration accumulators
        self._calib_depth_samples: list[float] = []
        self._calib_imu_samples: list[float] = []

        # Baselines (set after calibration)
        self._depth_baseline: float | None = None
        self._imu_quiet_magnitude: float | None = None

        # IMU drop flag
        self._drop_flag: bool = False
        self._drop_flag_expiry: float = 0.0

        # Latest cached values (used for display even between sensor updates)
        self._last_depth_delta: float = 0.0
        self._last_imu_delta: float = 0.0
        self._last_nn_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def calibrate(self) -> bool:
        """
        Collect calibration samples from depth and IMU queues.

        Returns True once both baselines are established; False while still
        warming up.  Call once per main-loop tick until it returns True.
        """
        # Drain depth queue for baseline samples
        if self._depth_q is not None and self._depth_baseline is None:
            while self._depth_q.has():
                raw = self._depth_q.get().getFrame()
                roi_median = self._roi_median(raw)
                if roi_median > 0:
                    self._calib_depth_samples.append(roi_median)

        # Drain IMU queue for baseline samples
        if self._imu_available and self._imu_q is not None and self._imu_quiet_magnitude is None:
            while self._imu_q.has():
                pkt = self._imu_q.get()
                for p in pkt.packets:
                    mag = _accel_magnitude(p.acceleroMeter)
                    self._calib_imu_samples.append(mag)

        # Finalise baselines when enough samples collected
        if self._depth_baseline is None and len(self._calib_depth_samples) >= OAK_CALIB_FRAMES:
            self._depth_baseline = float(np.median(self._calib_depth_samples))
            logger.info("Depth baseline: %.0f mm", self._depth_baseline)

        if (
            self._imu_available
            and self._imu_quiet_magnitude is None
            and len(self._calib_imu_samples) >= IMU_BASELINE_SAMPLES
        ):
            self._imu_quiet_magnitude = float(np.mean(self._calib_imu_samples))
            logger.info("IMU quiet magnitude: %.3f m/s²", self._imu_quiet_magnitude)

        # Depth is the primary calibration gate; IMU is optional
        depth_ready = (self._depth_q is None) or (self._depth_baseline is not None)
        imu_ready = (not self._imu_available) or (self._imu_quiet_magnitude is not None)
        return depth_ready and imu_ready

    def update(self) -> SensorVotes:
        """
        Drain all queues and return the current sensor votes.

        Should be called on every main-loop iteration (non-blocking).
        """
        rgb_frame = self._drain_rgb()
        depth_occupied = self._update_depth()
        self._update_imu()
        nn_count = self._update_nn()

        # Auto-expire drop flag
        if self._drop_flag and time.time() >= self._drop_flag_expiry:
            self._drop_flag = False

        votes = int(depth_occupied) + int(self._drop_flag) + int(nn_count > 0)

        return SensorVotes(
            depth_occupied=depth_occupied,
            drop_flag=self._drop_flag,
            nn_occupied=nn_count > 0,
            votes=votes,
            rgb_frame=rgb_frame,
            depth_mm_delta=self._last_depth_delta,
            imu_delta=self._last_imu_delta,
            nn_count=nn_count,
        )

    def calibration_progress(self) -> int:
        """Return estimated calibration progress as a percentage (0–100)."""
        from .config import IMU_BASELINE_SAMPLES  # avoid circular at module level

        d_pct = min(100, len(self._calib_depth_samples) * 100 // OAK_CALIB_FRAMES)
        if self._imu_available:
            i_pct = min(100, len(self._calib_imu_samples) * 100 // IMU_BASELINE_SAMPLES)
            return (d_pct + i_pct) // 2
        return d_pct

    def reset(self) -> None:
        """Clear all calibration state so calibrate() starts fresh."""
        self._calib_depth_samples.clear()
        self._calib_imu_samples.clear()
        self._depth_baseline = None
        self._imu_quiet_magnitude = None
        self._drop_flag = False
        self._drop_flag_expiry = 0.0
        self._last_depth_delta = 0.0
        self._last_imu_delta = 0.0
        self._last_nn_count = 0
        logger.info("OAKOccupancyDetector reset — recalibrating.")

    def stop(self) -> None:
        """Stop the depthai pipeline cleanly."""
        try:
            self._pipeline.stop()
        except Exception:
            pass

    @property
    def imu_available(self) -> bool:
        return bool(self._imu_available)

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

    def _roi_median(self, depth_frame: np.ndarray) -> float:
        """Return the median of valid (non-zero) depth pixels in the centre ROI."""
        h, w = depth_frame.shape[:2]
        margin_h = int(h * (1 - DEPTH_ROI_FRACTION) / 2)
        margin_w = int(w * (1 - DEPTH_ROI_FRACTION) / 2)
        roi = depth_frame[margin_h : h - margin_h, margin_w : w - margin_w]
        valid = roi[roi > 0]
        if valid.size < roi.size * 0.5:
            return 0.0  # too many invalid pixels — skip this frame
        return float(np.median(valid))

    def _update_depth(self) -> bool:
        """Drain depth queue; return True if ROI depth indicates occupied bin."""
        if self._depth_q is None or self._depth_baseline is None:
            return False

        current_median = None
        while self._depth_q.has():
            raw = self._depth_q.get().getFrame()
            m = self._roi_median(raw)
            if m > 0:
                current_median = m

        if current_median is None:
            return False

        delta = self._depth_baseline - current_median
        self._last_depth_delta = delta
        return delta > DEPTH_CHANGE_THRESHOLD

    def _update_imu(self) -> None:
        """Drain IMU queue; set drop flag on shock above threshold."""
        if not self._imu_available or self._imu_q is None:
            self._last_imu_delta = 0.0
            return
        if self._imu_quiet_magnitude is None:
            return

        while self._imu_q.has():
            pkt = self._imu_q.get()
            for p in pkt.packets:
                mag = _accel_magnitude(p.acceleroMeter)
                delta = abs(mag - self._imu_quiet_magnitude)
                self._last_imu_delta = max(self._last_imu_delta, delta)
                if delta > IMU_SHOCK_THRESHOLD:
                    self._drop_flag = True
                    self._drop_flag_expiry = time.time() + DROP_FLAG_DURATION
                    logger.debug("IMU drop event: delta=%.3f m/s²", delta)

        # Decay display delta toward zero when no active shock
        if not self._drop_flag:
            self._last_imu_delta = 0.0

    def _update_nn(self) -> int:
        """Drain NN queue; return count of detections above confidence threshold."""
        if self._nn_q is None:
            return self._last_nn_count

        count = None
        while self._nn_q.has():
            dets = self._nn_q.get().detections
            count = len(dets)  # already filtered by setConfidenceThreshold

        if count is not None:
            self._last_nn_count = count
        return self._last_nn_count


# ── Utility ────────────────────────────────────────────────────────────────────


def _accel_magnitude(a) -> float:
    """Return vector magnitude of an IMU accelerometer reading in m/s²."""
    return math.sqrt(a.x**2 + a.y**2 + a.z**2)
