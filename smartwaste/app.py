"""
smartwaste/app.py — shared OAK-camera run loop.

Usage::

    from smartwaste.app import run_loop
    from smartwaste.strategies import ManualStrategy
    run_loop(ManualStrategy())
"""

import contextlib
import sys
import time
from abc import ABC, abstractmethod

import cv2
import depthai as dai
import numpy as np

from .cameraOak import crop_sides, make_pipeline
from .config import CROP_PERCENT, DISPLAY_SIZE, MAX_DT, WINDOW
from .log_setup import get_logger
from .state import AppState
from .ui import draw_overlay

logger = get_logger()


class Strategy(ABC):
    """
    Defines the classification-trigger behaviour for one run-loop variant.

    Subclasses implement:

    * ``on_combined_frame`` — called every iteration when a synced combined
      frame is available; responsible for deciding when to fire the API.
    * ``on_key`` (optional) — called for every non-quit keypress; handle
      strategy-specific bindings here.

    Override ``setup`` to initialise per-strategy state before the loop.
    """

    def setup(self, state: AppState) -> None:
        """Called once with the fresh AppState before the loop starts."""

    @abstractmethod
    def on_combined_frame(self, combined: np.ndarray, state: AppState) -> None:
        """Handle one loop iteration while a synced combined frame is available."""

    def on_key(self, key: int, combined: np.ndarray | None, state: AppState) -> None:
        """Handle a non-quit keypress. Default: no-op."""


def run_loop(strategy: Strategy) -> None:
    """
    Initialise dual OAK cameras and run the capture / display / classify loop.

    The *strategy* controls when classifications are triggered and which
    additional key bindings are active.  Only 'q' (quit) is handled here.
    """
    state = AppState()
    strategy.setup(state)

    logger.info("Starting Smart Waste AI (%s)", type(strategy).__name__)

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
                    '  echo \'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"\''
                    " | sudo tee /etc/udev/rules.d/80-movidius.rules\n"
                    "  sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                    "Then reconnect the cameras."
                )
            raise RuntimeError(msg)

        logger.info("Using devices: %s", [i.getDeviceId() for i in infos[:2]])
        devices = [stack.enter_context(dai.Device(info)) for info in infos[:2]]

        pipelines: list = []
        queues: list = []
        for dev in devices:
            p, q = make_pipeline(dev)
            pipelines.append(p)
            queues.append(q)

        last_frames: list = [None, None]
        last_ts: list = [0.0, 0.0]

        try:
            while True:
                # ── Capture ────────────────────────────────────────────────────
                for i, q in enumerate(queues):
                    if q.has():
                        frame = q.get().getCvFrame()
                        last_ts[i] = time.time()
                        frame = crop_sides(frame, CROP_PERCENT)
                        frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
                        last_frames[i] = frame

                # ── Sync + display ─────────────────────────────────────────────
                combined = None
                if last_frames[0] is not None and last_frames[1] is not None:
                    if abs(last_ts[0] - last_ts[1]) <= MAX_DT:
                        combined = cv2.hconcat([last_frames[0], last_frames[1]])
                        label, detail, auto_on = state.get_display()
                        draw_overlay(combined, label, detail, auto_on, state.get_history())
                        cv2.imshow(WINDOW, combined)

                # ── Strategy logic ─────────────────────────────────────────────
                if combined is not None:
                    strategy.on_combined_frame(combined, state)

                # ── Keyboard ───────────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit.")
                    break
                if key != 0xFF:
                    strategy.on_key(key, combined, state)

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
