"""
mainauto.py — Smart Waste AI (Automatic Mode)

Local motion/presence detection gates all Gemini API calls:
  - Warm-up phase: learns an empty-bin background model (no API calls)
  - IDLE: bin looks empty → no API calls
  - DETECTED: object presence confirmed by local diff → one API call
  - CLASSIFIED: waiting for bin to clear → no API calls
  - CLEARED: bin back to empty → reset to IDLE, ready for next item

Controls:
  q — quit
  c — force-classify current frame immediately (manual override)
  r — reset background model (re-calibrate)
"""

import contextlib
import sys
import threading
import time

import cv2
import depthai as dai
import numpy as np

from smartwaste.camera import crop_sides, make_pipeline
from smartwaste.classifier import classify
from smartwaste.config import CROP_PERCENT, DISPLAY_SIZE, JPEG_QUALITY, MAX_DT, WINDOW
from smartwaste.log_setup import get_logger
from smartwaste.state import AppState
from smartwaste.ui import draw_overlay

logger = get_logger()
logger.info("Starting Smart Waste AI (Auto Gate Mode)")

# ── Local detection tuning ─────────────────────────────────────────────────────
MOTION_THRESHOLD  = 12.0   # mean-abs pixel diff (0-255) to consider bin non-empty
DETECT_CONFIRM_N  = 3      # consecutive above-threshold checks before triggering API
EMPTY_CONFIRM_N   = 6      # consecutive below-threshold checks before resetting to IDLE
BG_LEARNING_RATE  = 0.03   # how fast background model adapts when bin is empty
BG_WARMUP_FRAMES  = 40     # local checks during warmup before detection is active
CHECK_INTERVAL    = 0.5    # seconds between local presence checks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _encode(frame) -> bytes | None:
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return enc.tobytes() if ok else None


def _launch_classify(img_bytes: bytes, frame_copy, state: AppState) -> None:
    if img_bytes:
        threading.Thread(target=classify, args=(img_bytes, frame_copy, state), daemon=True).start()
    else:
        state.set_status("Error", "JPEG encode failed.")
        state.finish_classify()


# ── Background / presence detector ────────────────────────────────────────────

class PresenceDetector:
    """
    Maintains a rolling background model and reports whether an object
    is present in the bin based on per-frame pixel-diff scoring.
    """

    def __init__(self):
        self._bg: np.ndarray | None = None
        self._warmup_count = 0
        self._detect_streak = 0   # consecutive frames above threshold
        self._empty_streak  = 0   # consecutive frames below threshold

    @property
    def ready(self) -> bool:
        return self._warmup_count >= BG_WARMUP_FRAMES

    @property
    def warmup_progress(self) -> tuple[int, int]:
        return self._warmup_count, BG_WARMUP_FRAMES

    def reset(self, gray: np.ndarray | None = None) -> None:
        """Force-reset the background model (e.g. on user request or bin-clear)."""
        self._bg = gray.astype(np.float32) if gray is not None else None
        self._warmup_count  = 0 if gray is None else BG_WARMUP_FRAMES
        self._detect_streak = 0
        self._empty_streak  = 0

    def update(self, gray: np.ndarray) -> tuple[float, bool, bool]:
        """
        Feed a new grayscale frame.

        Returns:
            score       — mean absolute diff vs background
            is_occupied — True when detect streak confirms object present
            is_empty    — True when empty streak confirms bin has cleared
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
        """Hard-update background to current frame (called when bin confirmed empty)."""
        self._bg = gray.astype(np.float32)
        self._detect_streak = 0
        self._empty_streak  = 0


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    state    = AppState()
    detector = PresenceDetector()

    # In auto-gate mode the toggle is always on from the user perspective
    state.auto_classify = True

    # Main-thread-only tracking (no locking needed)
    last_check_time  = 0.0
    bin_occupied     = False   # True while we believe something is in the bin
    item_classified  = False   # True once we've triggered the API for the current item

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1600, 800)
    except cv2.error as e:
        raise RuntimeError(
            f"Cannot open display window: {e}\n"
            "On Linux/Raspberry Pi: a monitor must be connected and a desktop session active.\n"
            "If using SSH, run: export DISPLAY=:0  before starting the app."
        ) from e

    with contextlib.ExitStack() as stack:
        infos = dai.Device.getAllAvailableDevices()
        if len(infos) < 2:
            msg = f"Need 2 OAK devices connected. Found: {len(infos)}"
            if sys.platform != "win32":
                msg += (
                    "\nOn Linux/Raspberry Pi: udev rules may be missing. Run once:\n"
                    "  echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"03e7\", MODE=\"0666\"'"
                    " | sudo tee /etc/udev/rules.d/80-movidius.rules\n"
                    "  sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                    "Then reconnect the cameras."
                )
            raise RuntimeError(msg)

        logger.info("Using devices: %s", [i.getDeviceId() for i in infos[:2]])
        devices = [stack.enter_context(dai.Device(info)) for info in infos[:2]]

        pipelines, queues = [], []
        for dev in devices:
            p, q = make_pipeline(dev)
            pipelines.append(p)
            queues.append(q)

        last_frames = [None, None]
        last_ts     = [0.0, 0.0]

        try:
            while True:
                # ── Capture frames ─────────────────────────────────────────
                for i, q in enumerate(queues):
                    if q.has():
                        frame      = q.get().getCvFrame()
                        last_ts[i] = time.time()
                        frame      = crop_sides(frame, CROP_PERCENT)
                        frame      = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
                        last_frames[i] = frame

                combined = None
                if last_frames[0] is not None and last_frames[1] is not None:
                    if abs(last_ts[0] - last_ts[1]) <= MAX_DT:
                        combined = cv2.hconcat([last_frames[0], last_frames[1]])
                        label, detail, auto_on = state.get_display()
                        draw_overlay(combined, label, detail, auto_on)
                        cv2.imshow(WINDOW, combined)

                now = time.time()

                # ── Local presence check (cheap, no API) ───────────────────
                if combined is not None and now - last_check_time >= CHECK_INTERVAL:
                    last_check_time = now
                    gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
                    score, is_occupied, is_empty = detector.update(gray)

                    if not detector.ready:
                        done, total = detector.warmup_progress
                        state.set_status(
                            "Calibrating",
                            f"Learning empty-bin background... ({done}/{total})"
                        )

                    elif is_empty and bin_occupied:
                        # ── Bin cleared → reset ────────────────────────────
                        bin_occupied    = False
                        item_classified = False
                        detector.accept_as_background(gray)
                        state.set_status("Ready", "Bin is empty — waiting for item.")
                        logger.info("Bin cleared. Background updated. score=%.1f", score)

                    elif is_occupied and not bin_occupied:
                        # ── Object just entered bin ────────────────────────
                        bin_occupied    = True
                        item_classified = False
                        state.set_status("Detected", f"Object detected (score={score:.1f}) — classifying...")
                        logger.info("Object detected. score=%.1f → API call queued", score)

                    # ── API gate: fire exactly once per item ───────────────
                    # (state.start_classify is atomic; returns False if already running)
                    if bin_occupied and not item_classified and state.start_classify():
                        item_classified = True
                        img_bytes = _encode(combined)
                        logger.info("Gemini API call triggered. score=%.1f", score)
                        _launch_classify(img_bytes, combined.copy(), state)

                # ── Keyboard ───────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info("Quit.")
                    break

                if key == ord("c") and combined is not None and state.start_classify():
                    # Manual force-classify (bypasses local gate)
                    item_classified = True
                    logger.info("Manual classify triggered.")
                    _launch_classify(_encode(combined), combined.copy(), state)

                if key == ord("r") and combined is not None:
                    # Re-calibrate background from current frame
                    gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
                    detector.reset(gray)
                    bin_occupied    = False
                    item_classified = False
                    state.set_status("Ready", "Background reset from current frame.")
                    logger.info("Background manually reset.")

        except KeyboardInterrupt:
            logger.info("Stopping (Ctrl+C)...")

        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(0.2)
            for p in pipelines:
                try:
                    p.stop()
                except Exception:
                    pass
            logger.info("Stopped cleanly.")


if __name__ == "__main__":
    main()
