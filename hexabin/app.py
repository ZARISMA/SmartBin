"""
hexabin/app.py — shared OAK-camera run loop.

Usage::

    from hexabin.app import run_loop
    from hexabin.strategies import ManualStrategy
    run_loop(ManualStrategy())
"""

import contextlib
import os
import sys
import time
from abc import ABC, abstractmethod

import cv2
import depthai as dai
import numpy as np

from .camera_config import CameraConfigStore, apply_transform
from .cameraOak import make_pipeline
from .config import CAMERA_CONFIG_FILE, DISPLAY_SIZE, EDGE_MODE, MAX_DT, WINDOW
from .log_setup import get_logger
from .state import AppState
from .ui import draw_overlay

logger = get_logger()


def _is_headless() -> bool:
    """True when no display is available — edge containers, SSH, etc."""
    if os.environ.get("HEXABIN_HEADLESS", "").lower() in ("1", "true", "yes"):
        return True
    if EDGE_MODE:
        return True
    if sys.platform.startswith(("linux", "darwin")) and not os.environ.get("DISPLAY"):
        return True
    return False


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


def _set_strategy_name(state: AppState, strategy: "Strategy") -> None:
    """Map a Strategy subclass to the canonical name stored in AppState."""
    cls = type(strategy).__name__.lower()
    if "presence" in cls or "auto" in cls:
        state.set_strategy("auto")
    else:
        state.set_strategy("manual")


def run_loop(strategy: Strategy, state: AppState | None = None) -> None:
    """
    Initialise dual OAK cameras and run the capture / display / classify loop.

    The *strategy* controls when classifications are triggered and which
    additional key bindings are active.  Only 'q' (quit) is handled here.
    If ``state`` is None a fresh ``AppState`` is created.
    """
    if state is None:
        state = AppState()
    state.set_pipeline("oak")
    strategy.setup(state)

    # Record the initial strategy so the dashboard knows what's active.
    _set_strategy_name(state, strategy)

    logger.info("Starting Smart Waste AI (%s)", type(strategy).__name__)

    headless = _is_headless()

    # Per-camera geometry (rotate/flip/crop) — reloaded from disk so a saved
    # edit survives a restart; shared with the edge server so the dashboard can
    # read raw snapshots and push new configs while the loop runs.
    camera_store = CameraConfigStore()
    camera_store.load_json(CAMERA_CONFIG_FILE)

    frame_buf = None
    if EDGE_MODE:
        from .edge_client import start_heartbeat_thread
        from .edge_server import FrameBuffer, start_edge_server

        start_heartbeat_thread(state)
        frame_buf = FrameBuffer()
        start_edge_server(state, frame_buf, camera_store)

    if headless:
        logger.info("Headless mode — skipping OpenCV GUI.")
    else:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1600, 800)
        except cv2.error as e:
            raise RuntimeError(
                f"Cannot open display window: {e}\n"
                "On Linux/Raspberry Pi: a monitor must be connected and a desktop session active.\n"
                "If using SSH, run: export DISPLAY=:0  before starting the app.\n"
                "For headless edge deployments set HEXABIN_HEADLESS=1 "
                "or HEXABIN_EDGE_MODE=true."
            ) from e

    with contextlib.ExitStack() as stack:
        infos = dai.Device.getAllAvailableDevices()
        state.set_camera_count(len(infos))
        if len(infos) < 2:
            state.warnings.add(
                "CAMERA_COUNT_LOW",
                f"Only {len(infos)} OAK device(s) detected — 2 required for dual-camera mode.",
                severity="error",
            )
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
        state.warnings.clear("CAMERA_COUNT_LOW")

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
                # ── Admin shutdown / restart request ───────────────────────────
                if state.shutdown_requested:
                    logger.info("Shutdown requested via admin — stopping loop.")
                    break

                # ── Hot strategy swap (manual ↔ auto) ─────────────────────────
                pending = state.take_pending_strategy_swap()
                if pending:
                    from .strategies import build_strategy  # local import to avoid cycle

                    logger.info("Swapping strategy to %r", pending)
                    strategy = build_strategy(pending)
                    strategy.setup(state)
                    _set_strategy_name(state, strategy)

                # ── Capture ────────────────────────────────────────────────────
                for i, q in enumerate(queues):
                    if q.has():
                        frame = q.get().getCvFrame()
                        last_ts[i] = time.time()
                        camera_store.set_raw(i, frame)
                        frame = apply_transform(frame, camera_store.get(i))
                        frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
                        last_frames[i] = frame

                # ── Sync + display ─────────────────────────────────────────────
                combined = None
                if last_frames[0] is not None and last_frames[1] is not None:
                    if abs(last_ts[0] - last_ts[1]) <= MAX_DT:
                        combined = cv2.hconcat([last_frames[0], last_frames[1]])
                        if frame_buf is not None:
                            frame_buf.set(combined)
                        label, detail, auto_on = state.get_display()
                        draw_overlay(combined, label, detail, auto_on, state.get_history())
                        if not headless:
                            cv2.imshow(WINDOW, combined)

                # ── Strategy logic ─────────────────────────────────────────────
                if combined is not None:
                    strategy.on_combined_frame(combined, state)

                # ── Keyboard / pacing ──────────────────────────────────────────
                if headless:
                    time.sleep(0.01)
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit.")
                        break
                    if key != 0xFF:
                        strategy.on_key(key, combined, state)

        except KeyboardInterrupt:
            logger.info("Stopping (Ctrl+C)...")

        finally:
            if not headless:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            time.sleep(0.2)
            for p in pipelines:
                try:
                    p.stop()
                except Exception:
                    pass
            logger.info("Stopped cleanly.")
