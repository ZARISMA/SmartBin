import sys
import time

import cv2
from picamera2 import Picamera2

from smartwaste.cameraraspberry import crop_sides, grab_frame, make_cameras, stop_cameras
from smartwaste.config import AUTO_INTERVAL, CROP_PERCENT, DISPLAY_SIZE, WINDOW
from smartwaste.log_setup import get_logger
from smartwaste.state import AppState
from smartwaste.ui import draw_overlay
from smartwaste.utils import encode_frame, launch_classify

logger = get_logger()
logger.info("Starting Smart Waste AI (Raspberry Pi)")


def main() -> None:
    state = AppState()

    num_cameras = len(Picamera2.global_camera_info())
    if num_cameras < 2:
        msg = f"Need 2 Pi cameras connected. Found: {num_cameras}"
        if sys.platform != "win32":
            msg += (
                "\nOn Raspberry Pi 5 (dual native ports): camera_auto_detect=1 must be set in"
                " /boot/firmware/config.txt (default on Pi OS Bookworm)."
                "\nFor a camera multiplexer add: dtoverlay=camera-mux-4port"
            )
        raise RuntimeError(msg)

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1600, 800)
    except cv2.error as e:
        raise RuntimeError(
            f"Cannot open display window: {e}\n"
            "On Linux/Raspberry Pi: a monitor must be connected and a desktop session active.\n"
            "If using SSH, run: export DISPLAY=:0  before starting the app."
        ) from e

    cameras = make_cameras(2)
    logger.info("Opened %d Pi cameras.", len(cameras))

    last_frames = [None, None]
    last_ts     = [0.0, 0.0]

    try:
        while True:
            for i, cam in enumerate(cameras):
                frame      = grab_frame(cam)
                last_ts[i] = time.time()
                frame      = crop_sides(frame, CROP_PERCENT)
                frame      = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
                last_frames[i] = frame

            combined = None
            if last_frames[0] is not None and last_frames[1] is not None:
                combined = cv2.hconcat([last_frames[0], last_frames[1]])
                label, detail, auto_on = state.get_display()
                draw_overlay(combined, label, detail, auto_on)
                cv2.imshow(WINDOW, combined)

            now = time.time()
            if (state.auto_classify and combined is not None
                    and now - state.last_capture_time >= AUTO_INTERVAL
                    and state.start_classify()):
                state.last_capture_time = now
                launch_classify(encode_frame(combined), combined.copy(), state)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quit.")
                break

            if key == ord("c") and combined is not None and state.start_classify():
                launch_classify(encode_frame(combined), combined.copy(), state)

            if key == ord("a"):
                auto_on = state.toggle_auto()
                state.set_status("Ready", "Auto ON" if auto_on else "Auto OFF (manual: press 'c')")
                logger.info("AUTO_CLASSIFY=%s", auto_on)

    except KeyboardInterrupt:
        logger.info("Stopping (Ctrl+C)...")

    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        time.sleep(0.2)
        stop_cameras(cameras)
        logger.info("Stopped cleanly.")


if __name__ == "__main__":
    main()
