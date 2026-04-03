"""
mainauto.py — Smart Waste AI (Automatic Gate Mode)

Local presence detection gates all Gemini API calls:
  - Calibrating : learning empty-bin background  (no API)
  - Ready/IDLE  : bin looks empty               (no API)
  - Detected    : object confirmed by pixel-diff → one API call
  - Classified  : waiting for bin to clear       (no API)

Controls:
  q — quit
  c — force-classify current frame (manual override)
  r — reset background model from current frame
"""

import contextlib
import sys
import time

import cv2
import depthai as dai

from smartwaste.cameraOak import crop_sides, make_pipeline
from smartwaste.config import CHECK_INTERVAL, CROP_PERCENT, DISPLAY_SIZE, MAX_DT, WINDOW
from smartwaste.log_setup import get_logger
from smartwaste.presence import PresenceDetector
from smartwaste.state import AppState
from smartwaste.ui import draw_overlay
from smartwaste.utils import encode_frame, launch_classify

logger = get_logger()
logger.info("Starting Smart Waste AI (Auto Gate Mode)")


def main() -> None:
    state    = AppState()
    detector = PresenceDetector()

    state.auto_classify = True   # always on in this entry point

    # Main-thread-only state (no locking needed)
    last_check_time = 0.0
    bin_occupied    = False   # True while object is detected in bin
    item_classified = False   # True once API has been triggered for current item

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
                # ── Capture ────────────────────────────────────────────────
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
                            f"Learning empty-bin background... ({done}/{total})",
                        )

                    elif is_empty and bin_occupied:
                        # Bin cleared → reset
                        bin_occupied    = False
                        item_classified = False
                        detector.accept_as_background(gray)
                        state.set_status("Ready", "Bin is empty — waiting for item.")
                        logger.info("Bin cleared. Background updated. score=%.1f", score)

                    elif is_occupied and not bin_occupied:
                        # Object just entered bin
                        bin_occupied    = True
                        item_classified = False
                        state.set_status(
                            "Detected",
                            f"Object detected (score={score:.1f}) — classifying...",
                        )
                        logger.info("Object detected. score=%.1f → API call queued", score)

                    # Fire exactly one API call per item arrival
                    if bin_occupied and not item_classified and state.start_classify():
                        item_classified = True
                        logger.info("Gemini API call triggered. score=%.1f", score)
                        launch_classify(encode_frame(combined), combined.copy(), state)

                # ── Keyboard ───────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info("Quit.")
                    break

                if key == ord("c") and combined is not None and state.start_classify():
                    item_classified = True
                    logger.info("Manual classify triggered.")
                    launch_classify(encode_frame(combined), combined.copy(), state)

                if key == ord("r") and combined is not None:
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
