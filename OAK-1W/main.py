import contextlib
import sys
import time

import cv2
import depthai as dai

from smartwaste.camera import crop_sides, make_pipeline
from smartwaste.config import AUTO_INTERVAL, CROP_PERCENT, DISPLAY_SIZE, MAX_DT, WINDOW
from smartwaste.log_setup import get_logger
from smartwaste.state import AppState
from smartwaste.ui import draw_overlay
from smartwaste.utils import encode_frame, launch_classify

logger = get_logger()
logger.info("Starting Smart Waste AI")


def main() -> None:
    state = AppState()

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
            for p in pipelines:
                try:
                    p.stop()
                except Exception:
                    pass
            logger.info("Stopped cleanly.")


if __name__ == "__main__":
    main()
