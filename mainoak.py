"""
mainoak.py — Smart Waste AI (OAK-D Native Mode)

Supports 1 or 2 OAK-D cameras:

  2 cameras → dual view (sensor fusion on camera 1, RGB from camera 2,
              frames concatenated side-by-side for Gemini)
  1 camera  → single view with sensor fusion only

Uses three OAK-D hardware sensors on the primary camera to decide when the
bin is occupied, then fires one Gemini API call for waste classification:

  Depth   — stereo IR measures distance change in bin ROI (lighting-independent)
  IMU     — accelerometer detects the physical shock when an item is dropped
  NN      — MobileNetSSD runs on the Myriad X VPU for on-device object detection

All three sensors vote; Gemini is triggered when ≥ 2 agree the bin is occupied.

State machine
─────────────
  Calibrating → Ready → Detected → Classifying → Classified → Ready

Controls
────────
  q — quit
  c — force-classify current frame
  r — reset / recalibrate

CLI overrides (all also settable via env vars or .env):
  --model NAME              Gemini model  (SMARTWASTE_MODEL_NAME)
  --depth-threshold MM      Depth change threshold  (SMARTWASTE_DEPTH_CHANGE_THRESHOLD)
  --imu-threshold FLOAT     IMU shock threshold  (SMARTWASTE_IMU_SHOCK_THRESHOLD)
  --votes N                 Sensor votes needed  (SMARTWASTE_OAK_VOTES_NEEDED)
  --location NAME           Deployment location tag  (SMARTWASTE_LOCATION)
"""

from __future__ import annotations

import argparse
import enum
import os
import sys
import time


# ── Early CLI parsing — sets env vars BEFORE smartwaste modules are imported ──
# This ensures Settings() sees CLI values when config constants are assigned.
def _export_cli_overrides() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model")
    p.add_argument("--depth-threshold", type=int)
    p.add_argument("--imu-threshold", type=float)
    p.add_argument("--votes", type=int)
    p.add_argument("--location")
    known, _ = p.parse_known_args()
    if known.model:
        os.environ["SMARTWASTE_MODEL_NAME"] = known.model
    if known.depth_threshold is not None:
        os.environ["SMARTWASTE_DEPTH_CHANGE_THRESHOLD"] = str(known.depth_threshold)
    if known.imu_threshold is not None:
        os.environ["SMARTWASTE_IMU_SHOCK_THRESHOLD"] = str(known.imu_threshold)
    if known.votes is not None:
        os.environ["SMARTWASTE_OAK_VOTES_NEEDED"] = str(known.votes)
    if known.location:
        os.environ["SMARTWASTE_LOCATION"] = known.location


_export_cli_overrides()  # must run before the smartwaste imports below

import contextlib

import cv2
import depthai as dai
import numpy as np

from smartwaste.cameraOak import crop_sides, make_pipeline
from smartwaste.config import (
    CROP_PERCENT,
    DISPLAY_SIZE,
    MAX_DT,
    OAK_CHECK_INTERVAL,
    OAK_DETECT_CONFIRM_N,
    OAK_DISPLAY_H,
    OAK_DISPLAY_W,
    OAK_EMPTY_CONFIRM_N,
    OAK_VOTES_NEEDED,
    OAK_WINDOW,
)
from smartwaste.log_setup import get_logger
from smartwaste.oak_native import OAKOccupancyDetector, SensorVotes
from smartwaste.state import AppState
from smartwaste.ui import draw_nn_detections
from smartwaste.utils import encode_frame, launch_classify

logger = get_logger()


# ── State machine ──────────────────────────────────────────────────────────────

class OakState(enum.Enum):
    CALIBRATING = "Calibrating"
    READY       = "Ready"
    DETECTED    = "Detected"
    CLASSIFYING = "Classifying"
    CLASSIFIED  = "Classified"


# ── Overlay ────────────────────────────────────────────────────────────────────

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_WHITE     = (255, 255, 255)
_GREEN     = (0, 220, 0)
_YELLOW    = (0, 220, 220)
_BAR_H     = 150   # height of the status bar in pixels


def _draw_overlay(
    frame: np.ndarray,
    oak_state: OakState,
    votes: SensorVotes,
    app_state: AppState,
    detector: OAKOccupancyDetector,
    calib_pct: int,
    *,
    title: str = "OAK-D Native",
) -> None:
    """Draw a three-line status bar onto *frame* (in-place)."""
    h, w = frame.shape[:2]

    # Semi-transparent black background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, _BAR_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    label, detail, _ = app_state.get_display()

    # Line 1 — title + state
    state_str = oak_state.value
    if oak_state == OakState.CALIBRATING:
        state_str = f"Calibrating … {calib_pct}%"
    cv2.putText(frame,
                f"Smart Waste AI ({title})   |   {state_str}",
                (14, 34), _FONT, 0.70, _WHITE, 1, cv2.LINE_AA)

    # Line 2 — votes + sensor availability
    depth_s = "Depth"
    imu_s   = "IMU" if detector.imu_available else "IMU(N/A)"
    nn_s    = "NN"  if detector.nn_available  else "NN(N/A)"
    voted   = []
    if votes.depth_occupied: voted.append(depth_s)
    if votes.drop_flag:      voted.append(imu_s)
    if votes.nn_occupied:    voted.append(nn_s)
    vote_str = f"Votes: {votes.votes}/{_active_count(detector)}  →  {', '.join(voted) or 'none'}"
    cv2.putText(frame, vote_str, (14, 72), _FONT, 0.60, _GREEN, 1, cv2.LINE_AA)

    # Line 3 — per-sensor readings
    depth_val = f"{votes.depth_mm_delta:+.0f}mm"
    imu_val   = "DROP" if votes.drop_flag else f"{votes.imu_delta:.2f}m/s²"
    nn_val    = f"{votes.nn_count} obj"
    reading_str = f"Depth: {depth_val}   |   IMU: {imu_val}   |   NN: {nn_val}"
    cv2.putText(frame, reading_str, (14, 108), _FONT, 0.56, _YELLOW, 1, cv2.LINE_AA)

    # Line 4 — Gemini result when available
    if label not in ("Ready", "Calibrating"):
        cv2.putText(frame, f"{label}  {detail}", (14, 140),
                    _FONT, 0.52, _WHITE, 1, cv2.LINE_AA)


def _active_count(detector: OAKOccupancyDetector) -> int:
    """Number of sensors that are actually available."""
    return 1 + int(detector.imu_available) + int(detector.nn_available)


# ── State-machine tick (called at OAK_CHECK_INTERVAL) ─────────────────────────

def _tick(
    state: OakState,
    votes: SensorVotes,
    app_state: AppState,
    detector: OAKOccupancyDetector,
    detect_streak: int,
    empty_streak: int,
    calib_pct_ref: list[int],
    classify_frame: np.ndarray | None = None,
) -> tuple[OakState, int, int]:
    """Advance the state machine by one check interval. Returns updated state."""

    if state == OakState.CALIBRATING:
        done = detector.calibrate()
        calib_pct_ref[0] = detector.calibration_progress()
        if done:
            app_state.set_status("Ready", "Bin empty — waiting for item.")
            logger.info("Calibration complete.")
            return OakState.READY, 0, 0
        return state, detect_streak, empty_streak

    if state == OakState.READY:
        if votes.votes >= OAK_VOTES_NEEDED:
            detect_streak += 1
            app_state.set_status(
                "Detecting",
                f"Object signals: {votes.votes} vote(s) ({detect_streak}/{OAK_DETECT_CONFIRM_N})",
            )
            if detect_streak >= OAK_DETECT_CONFIRM_N:
                logger.info("Occupancy confirmed (%d votes). Triggering classify.", votes.votes)
                return OakState.DETECTED, 0, 0
        else:
            if detect_streak:
                app_state.set_status("Ready", "Bin empty — waiting for item.")
            detect_streak = 0
        return state, detect_streak, empty_streak

    if state == OakState.DETECTED:
        frame = classify_frame if classify_frame is not None else votes.rgb_frame
        if frame is not None and app_state.start_classify():
            img_bytes = encode_frame(frame)
            launch_classify(img_bytes, frame.copy(), app_state)
            app_state.set_status("Classifying…", "Sending to Gemini AI…")
            return OakState.CLASSIFYING, 0, 0
        return state, detect_streak, empty_streak

    if state == OakState.CLASSIFYING:
        if not app_state.is_classifying:
            logger.info("Classification complete.")
            return OakState.CLASSIFIED, 0, 0
        return state, detect_streak, empty_streak

    if state == OakState.CLASSIFIED:
        if votes.votes == 0:
            empty_streak += 1
            if empty_streak >= OAK_EMPTY_CONFIRM_N:
                app_state.set_status("Ready", "Bin cleared — waiting for next item.")
                logger.info("Bin empty confirmed — returning to Ready.")
                return OakState.READY, 0, 0
        else:
            empty_streak = 0
        return state, detect_streak, empty_streak

    return state, detect_streak, empty_streak


# ── Keyboard handler ───────────────────────────────────────────────────────────

def _handle_key(
    key: int,
    oak_state: OakState,
    votes: SensorVotes,
    app_state: AppState,
    detector: OAKOccupancyDetector,
    classify_frame: np.ndarray | None = None,
) -> OakState:
    """Handle a non-quit keypress. Returns (possibly changed) state."""

    if key == ord("c"):
        # Force-classify current frame regardless of sensor state
        frame = classify_frame if classify_frame is not None else votes.rgb_frame
        if frame is not None and app_state.start_classify():
            img_bytes = encode_frame(frame)
            launch_classify(img_bytes, frame.copy(), app_state)
            app_state.set_status("Classifying…", "Manual trigger → Gemini…")
            return OakState.CLASSIFYING
        else:
            logger.info("Force-classify ignored: no frame or already classifying.")

    elif key == ord("r"):
        detector.reset()
        app_state.set_status("Calibrating", "Resetting sensors…")
        logger.info("Manual recalibration triggered.")
        return OakState.CALIBRATING

    return oak_state


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Full parser — for --help and validation only (values already in env vars)
    p = argparse.ArgumentParser(
        description="SmartWaste AI — OAK-D Native mode (1 or 2 cameras)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", metavar="NAME", help="Gemini model name")
    p.add_argument("--depth-threshold", type=int, metavar="MM",
                   help="Depth change (mm) to declare bin occupied")
    p.add_argument("--imu-threshold", type=float, metavar="FLOAT",
                   help="IMU acceleration delta (m/s²) to flag a drop event")
    p.add_argument("--votes", type=int, metavar="N",
                   help="Sensor votes needed to trigger classification")
    p.add_argument("--location", metavar="NAME",
                   help="Deployment location written to dataset")
    p.parse_args()   # triggers --help / validation; actual values are in env already

    # ── Device discovery ───────────────────────────────────────────────────────
    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        msg = "No OAK-D device found. Check USB3 connection."
        if sys.platform != "win32":
            msg += (
                "\nOn Linux/Raspberry Pi, udev rules may be missing. Run once:\n"
                "  echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"03e7\", MODE=\"0666\"'"
                " | sudo tee /etc/udev/rules.d/80-movidius.rules\n"
                "  sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                "Then reconnect the camera."
            )
        raise RuntimeError(msg)

    dual_mode = len(infos) >= 2
    if dual_mode:
        logger.info("Dual-camera mode: devices %s, %s",
                     infos[0].getDeviceId(), infos[1].getDeviceId())
    else:
        logger.info("Single-camera mode: device %s", infos[0].getDeviceId())

    # ── Display window ─────────────────────────────────────────────────────────
    display_w = DISPLAY_SIZE[0] * 2 if dual_mode else OAK_DISPLAY_W
    display_h = DISPLAY_SIZE[1] if dual_mode else OAK_DISPLAY_H
    mode_title = "Dual OAK-D Native" if dual_mode else "OAK-D Native"

    try:
        cv2.namedWindow(OAK_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(OAK_WINDOW, display_w, display_h)
    except cv2.error as exc:
        raise RuntimeError(
            f"Cannot open display window: {exc}\n"
            "On Linux/Raspberry Pi: a monitor must be connected and a desktop "
            "session active. If using SSH, run: export DISPLAY=:0"
        ) from exc

    app_state = AppState()
    app_state.set_status("Calibrating", "Warming up OAK-D sensors…")

    with contextlib.ExitStack() as exit_stack:
        # Primary camera — full sensor pipeline (depth, IMU, NN)
        device1 = exit_stack.enter_context(dai.Device(infos[0]))
        detector = OAKOccupancyDetector(device1)

        # Optional second camera — simple RGB pipeline
        cam2_pipeline = None
        cam2_queue = None
        if dual_mode:
            device2 = exit_stack.enter_context(dai.Device(infos[1]))
            cam2_pipeline, cam2_queue = make_pipeline(device2)

        oak_state     = OakState.CALIBRATING
        detect_streak = 0
        empty_streak  = 0
        last_check    = 0.0
        calib_pct     = [0]         # mutable container so _tick can write to it
        last_votes    = SensorVotes(
            False, False, False, 0, None, 0.0, 0.0, 0, []
        )

        # Dual-mode frame state
        last_cam1_ts      = 0.0
        last_cam1_cropped = None    # cam1 frame after crop + resize
        last_cam2_frame   = None
        last_cam2_ts      = 0.0

        logger.info("Starting Smart Waste AI (%s)", mode_title)

        try:
            while True:
                # ── Drain all sensor queues (camera 1) ────────────────────────
                votes = detector.update()
                if votes.rgb_frame is not None:
                    last_votes = votes
                    last_cam1_ts = time.time()
                    if dual_mode:
                        cropped = crop_sides(votes.rgb_frame, CROP_PERCENT)
                        last_cam1_cropped = cv2.resize(
                            cropped, DISPLAY_SIZE,
                            interpolation=cv2.INTER_AREA,
                        )

                # ── Second camera capture (dual mode) ─────────────────────────
                if dual_mode and cam2_queue is not None:
                    while cam2_queue.has():
                        raw2 = cam2_queue.get().getCvFrame()
                        raw2 = crop_sides(raw2, CROP_PERCENT)
                        last_cam2_frame = cv2.resize(
                            raw2, DISPLAY_SIZE,
                            interpolation=cv2.INTER_AREA,
                        )
                        last_cam2_ts = time.time()

                # ── Build classify frame ──────────────────────────────────────
                classify_frame = None
                if dual_mode:
                    if (last_cam1_cropped is not None
                            and last_cam2_frame is not None
                            and abs(last_cam1_ts - last_cam2_ts) <= MAX_DT):
                        classify_frame = cv2.hconcat(
                            [last_cam1_cropped, last_cam2_frame]
                        )
                elif last_votes.rgb_frame is not None:
                    classify_frame = last_votes.rgb_frame

                # ── Display ───────────────────────────────────────────────────
                if classify_frame is not None:
                    disp = cv2.resize(
                        classify_frame, (display_w, display_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    draw_nn_detections(disp, last_votes.nn_detections)
                    _draw_overlay(disp, oak_state, last_votes, app_state,
                                  detector, calib_pct[0], title=mode_title)
                    cv2.imshow(OAK_WINDOW, disp)

                # ── State-machine tick at CHECK_INTERVAL ──────────────────────
                now = time.time()
                if now - last_check >= OAK_CHECK_INTERVAL:
                    last_check = now
                    oak_state, detect_streak, empty_streak = _tick(
                        oak_state, last_votes, app_state, detector,
                        detect_streak, empty_streak, calib_pct,
                        classify_frame,
                    )

                # ── Keyboard ──────────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit.")
                    break
                if key != 0xFF:
                    oak_state = _handle_key(
                        key, oak_state, last_votes, app_state, detector,
                        classify_frame,
                    )

        except KeyboardInterrupt:
            logger.info("Stopping (Ctrl+C)…")
        finally:
            detector.stop()
            if cam2_pipeline is not None:
                try:
                    cam2_pipeline.stop()
                except Exception:
                    pass
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(0.2)
            logger.info("Stopped cleanly.")


if __name__ == "__main__":
    main()
