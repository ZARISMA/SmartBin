"""
smartwaste/web.py — FastAPI web UI with live MJPEG stream.

Run with::

    python -m smartwaste.web

Replaces cv2.imshow when running inside Docker.
"""

import contextlib
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import StreamingResponse

from .config import (
    AUTO_INTERVAL,
    CROP_PERCENT,
    DISPLAY_SIZE,
    JPEG_QUALITY,
    MAX_DT,
    WEB_HOST,
    WEB_PORT,
)
from .database import get_entries, get_entry_count, get_label_counts
from .log_setup import get_logger
from .state import AppState
from .ui import draw_overlay
from .utils import encode_frame, launch_classify

logger = get_logger()

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(application: FastAPI):
    _start_camera_thread()
    yield


app = FastAPI(title="Smart Waste AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(_MODULE_DIR, "web_static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_MODULE_DIR, "web_templates"))

# ── Shared state ──────────────────────────────────────────────────────────────

_state = AppState()
_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()
_cameras_ok = False


def _set_frame(frame: np.ndarray) -> None:
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()


def _get_frame() -> np.ndarray | None:
    with _frame_lock:
        return _latest_frame.copy() if _latest_frame is not None else None


# ── Camera thread ─────────────────────────────────────────────────────────────


def _camera_loop() -> None:
    """Capture frames from dual OAK cameras in a background thread."""
    global _cameras_ok

    try:
        import depthai as dai  # noqa: I001

        from .cameraOak import crop_sides, make_pipeline
    except Exception as e:
        logger.error("Cannot import camera modules: %s", e)
        return

    with contextlib.ExitStack() as stack:
        infos = dai.Device.getAllAvailableDevices()
        if len(infos) < 2:
            logger.warning("Need 2 OAK devices, found %d. Camera stream unavailable.", len(infos))
            return

        devices = [stack.enter_context(dai.Device(info)) for info in infos[:2]]
        pipelines, queues = [], []
        for dev in devices:
            p, q = make_pipeline(dev)
            pipelines.append(p)
            queues.append(q)

        last_frames: list[np.ndarray | None] = [None, None]
        last_ts: list[float] = [0.0, 0.0]
        _cameras_ok = True
        logger.info("Camera thread started with %d devices", len(devices))

        try:
            while True:
                for i, q in enumerate(queues):
                    if q.has():
                        frame = q.get().getCvFrame()
                        last_ts[i] = time.time()
                        frame = crop_sides(frame, CROP_PERCENT)
                        frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
                        last_frames[i] = frame

                if last_frames[0] is not None and last_frames[1] is not None:
                    if abs(last_ts[0] - last_ts[1]) <= MAX_DT:
                        combined = cv2.hconcat([last_frames[0], last_frames[1]])
                        _set_frame(combined)

                        # Auto-classify
                        if _state.auto_classify:
                            now = time.time()
                            if now - _state.last_capture_time >= AUTO_INTERVAL:
                                if _state.start_classify():
                                    _state.last_capture_time = now
                                    img_bytes = encode_frame(combined)
                                    launch_classify(img_bytes, combined.copy(), _state)

                time.sleep(0.01)
        except Exception as e:
            logger.error("Camera loop error: %s", e)
        finally:
            for p in pipelines:
                with contextlib.suppress(Exception):
                    p.stop()


def _start_camera_thread() -> None:
    t = threading.Thread(target=_camera_loop, daemon=True, name="camera-thread")
    t.start()


# ── MJPEG generator ──────────────────────────────────────────────────────────


def _generate_frames():
    """Yield JPEG frames as multipart MJPEG stream."""
    while True:
        frame = _get_frame()
        if frame is not None:
            display = frame.copy()
            label, detail, auto_on = _state.get_display()
            draw_overlay(display, label, detail, auto_on, _state.get_history())
            ok, jpeg = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        else:
            # No camera — send placeholder
            placeholder = np.zeros((800, 1600, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Waiting for cameras...",
                (500, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (120, 120, 120),
                2,
                cv2.LINE_AA,
            )
            ok, jpeg = cv2.imencode(".jpg", placeholder)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.033)  # ~30 FPS cap


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/stream")
def video_feed():
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/classify")
def api_classify():
    frame = _get_frame()
    if frame is None:
        return JSONResponse({"error": "No camera frame available"}, status_code=503)
    if not _state.start_classify():
        return JSONResponse({"error": "Classification already in progress"}, status_code=409)
    img_bytes = encode_frame(frame)
    launch_classify(img_bytes, frame.copy(), _state)
    return {"status": "classifying"}


@app.post("/api/toggle-auto")
def api_toggle_auto():
    new_state = _state.toggle_auto()
    return {"auto_classify": new_state}


@app.get("/api/state")
def api_state():
    label, detail, auto_on = _state.get_display()
    return {
        "label": label,
        "detail": detail,
        "auto_on": auto_on,
        "is_classifying": _state.is_classifying,
        "history": [{"time": ts, "label": lbl} for ts, lbl in _state.get_history()],
    }


@app.get("/api/entries")
def api_entries(limit: int = 20, offset: int = 0):
    return get_entries(limit=limit, offset=offset)


@app.get("/api/stats")
def api_stats():
    return {
        "total": get_entry_count(),
        "by_category": get_label_counts(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("smartwaste.web:app", host=WEB_HOST, port=WEB_PORT)
