"""
smartwaste/edge_server.py — sidecar HTTP server for edge devices.

Exposes the local camera stream and control endpoints so the central
server can proxy them onto its dashboard. Started from any of the
main*.py entry points when EDGE_MODE is enabled.

Endpoints (all require Bearer EDGE_API_KEY):

  GET  /stream         MJPEG stream of the latest frame
  POST /classify       Force-classify the current frame
  POST /toggle-auto    Toggle auto-classify
  GET  /state          Current classification label / history
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .config import EDGE_API_KEY, WEB_HOST, WEB_PORT
from .log_setup import get_logger
from .state import AppState
from .utils import encode_frame, launch_classify

logger = get_logger()


class FrameBuffer:
    """Thread-safe single-slot frame buffer shared with the camera loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def set(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        with self._lock:
            self._frame = frame.copy()

    def get(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()


def _placeholder_frame(msg: str = "Waiting for camera…") -> bytes:
    img = np.zeros((240, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2, cv2.LINE_AA)
    ok, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes() if ok else b""


def _gen_frames(buf: FrameBuffer):
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame = buf.get()
        if frame is not None:
            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                yield boundary + jpeg.tobytes() + b"\r\n"
        else:
            yield boundary + _placeholder_frame() + b"\r\n"
        time.sleep(0.033)  # ~30 FPS cap


def _check_auth(auth: str) -> bool:
    if not auth or not auth.startswith("Bearer "):
        return False
    token = auth[7:]
    return bool(EDGE_API_KEY) and token == EDGE_API_KEY


def _build_app(state: AppState, buf: FrameBuffer):
    from fastapi import FastAPI, Header
    from fastapi.responses import JSONResponse
    from starlette.responses import StreamingResponse

    from .schemas import BinCommand

    app = FastAPI(title="SmartBin Edge")

    def _unauthorized():
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    @app.get("/stream")
    def stream(authorization: str = Header("")):
        if not _check_auth(authorization):
            return _unauthorized()
        return StreamingResponse(
            _gen_frames(buf),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/classify")
    def classify(authorization: str = Header("")):
        if not _check_auth(authorization):
            return _unauthorized()
        frame = buf.get()
        if frame is None:
            return JSONResponse({"error": "No camera frame available"}, status_code=503)
        if not state.start_classify():
            return JSONResponse({"error": "Classification already in progress"}, status_code=409)
        img_bytes = encode_frame(frame)
        launch_classify(img_bytes, frame.copy(), state)
        return {"status": "classifying"}

    @app.post("/toggle-auto")
    def toggle(authorization: str = Header("")):
        if not _check_auth(authorization):
            return _unauthorized()
        return {"auto_classify": state.toggle_auto()}

    @app.get("/state")
    def get_state(authorization: str = Header("")):
        if not _check_auth(authorization):
            return _unauthorized()
        label, detail, auto_on = state.get_display()
        return {
            "label": label,
            "detail": detail,
            "auto_on": auto_on,
            "is_classifying": state.is_classifying,
            "history": [{"time": ts, "label": lbl} for ts, lbl in state.get_history()],
        }

    @app.get("/diagnostics")
    def diagnostics(authorization: str = Header("")):
        if not _check_auth(authorization):
            return _unauthorized()
        label, detail, auto_on = state.get_display()
        return {
            "strategy": state.get_strategy(),
            "pipeline": state.get_pipeline(),
            "camera_count": state.get_camera_count(),
            "running": state.running,
            "is_classifying": state.is_classifying,
            "auto_classify": auto_on,
            "label": label,
            "detail": detail,
            "warnings": state.warnings.list(),
        }

    @app.post("/command")
    def command(cmd: BinCommand, authorization: str = Header("")):
        if not _check_auth(authorization):
            return _unauthorized()

        action = cmd.action
        value = (cmd.value or "").strip().lower()

        if action == "stop":
            state.set_running(False)
            state.set_status("Stopped", "Classifications paused by admin.")
            logger.info("Admin command: stop")
            return {"status": "ok", "message": "classifications paused"}

        if action == "start":
            state.set_running(True)
            state.set_status("Ready", "Resumed by admin.")
            logger.info("Admin command: start")
            return {"status": "ok", "message": "resumed"}

        if action == "restart":
            logger.info("Admin command: restart — requesting supervisor respawn")
            state.request_restart()
            return {"status": "ok", "message": "restart requested"}

        if action == "set_strategy":
            if value not in ("manual", "auto"):
                return JSONResponse(
                    {"status": "error", "message": f"invalid strategy: {value!r}"},
                    status_code=400,
                )
            # Hot-swap within the dual-OAK pipeline only. For oak-native,
            # the admin must use set_pipeline to change to the dual pipeline.
            if state.get_pipeline() != "oak":
                return JSONResponse(
                    {
                        "status": "error",
                        "message": "strategy swap only supported on the dual-OAK pipeline",
                    },
                    status_code=409,
                )
            state.request_strategy_swap(value)
            logger.info("Admin command: set_strategy=%s", value)
            return {"status": "ok", "message": f"strategy will swap to {value}"}

        if action == "set_pipeline":
            if value not in ("oak", "oak-native"):
                return JSONResponse(
                    {"status": "error", "message": f"invalid pipeline: {value!r}"},
                    status_code=400,
                )
            # Pipeline change requires a process restart. Persist the new
            # mode to env so control.py picks it up on respawn.
            import os as _os

            _os.environ["SMARTWASTE_CAMERA_MODE"] = value
            logger.info("Admin command: set_pipeline=%s (restart required)", value)
            state.request_restart()
            return {"status": "ok", "message": f"pipeline will change to {value} on restart"}

        if action == "classify":
            frame = buf.get()
            if frame is None:
                return JSONResponse(
                    {"status": "error", "message": "no frame available"}, status_code=503
                )
            if not state.start_classify():
                return JSONResponse(
                    {"status": "error", "message": "already classifying"}, status_code=409
                )
            img_bytes = encode_frame(frame)
            launch_classify(img_bytes, frame.copy(), state)
            return {"status": "ok", "message": "classifying"}

        if action == "toggle_auto":
            new = state.toggle_auto()
            return {"status": "ok", "message": f"auto_classify={new}"}

        if action == "clear_warnings":
            state.warnings.clear_all()
            return {"status": "ok", "message": "warnings cleared"}

        return JSONResponse(
            {"status": "error", "message": f"unknown action: {action!r}"},
            status_code=400,
        )

    return app


def start_edge_server(
    state: AppState,
    buf: FrameBuffer,
    *,
    host: str | None = None,
    port: int | None = None,
) -> FrameBuffer:
    """Start the edge HTTP server on a daemon thread. Returns the frame buffer."""
    if not EDGE_API_KEY:
        logger.warning("Edge server not started — EDGE_API_KEY not set.")
        return buf

    bind_host = host or WEB_HOST or "0.0.0.0"
    bind_port = port or WEB_PORT

    try:
        import uvicorn
    except ImportError:
        logger.warning("Edge server not started — uvicorn not installed.")
        return buf

    app = _build_app(state, buf)

    def _run() -> None:
        try:
            uvicorn.run(app, host=bind_host, port=bind_port, log_level="warning")
        except Exception as exc:
            logger.error("Edge server crashed: %s", exc)

    t = threading.Thread(target=_run, daemon=True, name="edge-server")
    t.start()
    logger.info("Edge server started on %s:%d", bind_host, bind_port)
    return buf
