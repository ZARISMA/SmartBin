"""
smartwaste/web.py — FastAPI web UI with live MJPEG stream.

Run with::

    python -m smartwaste.web

Replaces cv2.imshow when running inside Docker.

Camera mode is selected via the SMARTWASTE_CAMERA_MODE env var:
  oak          — dual OAK-D USB3 cameras (default)
  raspberry    — dual Raspberry Pi cameras (picamera2)
  oak-native   — single OAK-D with depth/IMU/NN sensor fusion
"""

import base64
import contextlib
import csv
import io
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.request

# Make StaticFiles return the correct Content-Type for 3D model assets.
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse

from . import analytics
from .actuator import resolve_module
from .config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    AUTO_INTERVAL,
    BIN_ID,
    CAMERA_MODE,
    CROP_PERCENT,
    DATASET_DIR,
    DISPLAY_SIZE,
    EDGE_API_KEY,
    EDGE_MODE,
    JPEG_QUALITY,
    LLM_MAX_CONCURRENCY,
    LLM_QUEUE_TIMEOUT,
    MAX_DT,
    MAX_UPLOAD_BYTES,
    OAK_CHECK_INTERVAL,
    OAK_DETECT_CONFIRM_N,
    OAK_DISPLAY_H,
    OAK_DISPLAY_W,
    OAK_EMPTY_CONFIRM_N,
    OAK_VOTES_NEEDED,
    RECYCLABLE_CLASSES,
    SECRET_KEY,
    VALID_CLASSES,
    WEB_HOST,
    WEB_PORT,
)
from .database import (
    get_active_bins,
    get_entries,
    get_entry_by_id,
    get_entry_count,
    get_label_counts,
    insert_entry,
)
from .llm import CircuitOpenError, build_backend
from .log_setup import get_logger
from .schemas import (
    BinCommand,
    BinHeartbeat,
    EdgeClassifyRequest,
    EdgeClassifyResponse,
    EdgeReport,
)
from .state import AppState
from .ui import draw_nn_detections, draw_overlay
from .utils import encode_frame, launch_classify

logger = get_logger()

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(application: FastAPI):
    _start_camera_thread()
    if EDGE_MODE:
        from .edge_client import start_heartbeat_thread

        start_heartbeat_thread()
    yield


app = FastAPI(title="Smart Waste AI", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory=os.path.join(_MODULE_DIR, "web_static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_MODULE_DIR, "web_templates"))


# ── Auth helpers ─────────────────────────────────────────────────────────────


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""


def _is_admin(request: Request) -> bool:
    """Operator auth: session cookie or the admin password as a bearer token."""
    if request.session.get("user"):
        return True
    return _bearer_token(request) == ADMIN_PASSWORD


def _is_edge_client(request: Request) -> bool:
    """Edge-device auth — valid ONLY for the ingest endpoints
    (/api/report, /api/heartbeat, /api/edge/classify).

    The edge API key deliberately does not open admin routes; admins may
    still call the ingest endpoints."""
    token = _bearer_token(request)
    if EDGE_API_KEY and token == EDGE_API_KEY:
        return True
    return _is_admin(request)


_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(label: str, bin_id: str, ts: str) -> str:
    """Build a flat .jpg filename from client-supplied fields.

    Strips path separators and anything else that could escape DATASET_DIR."""
    parts = []
    for raw in (label, bin_id, ts):
        part = _SAFE_FILENAME_CHARS.sub("_", str(raw)).lstrip(".")[:40]
        parts.append(part or "x")
    return "_".join(parts) + ".jpg"


# ── Bin registry (in-memory, populated by edge heartbeats) ───────────────────


@dataclass
class BinInfo:
    bin_id: str
    status: str = "online"
    last_seen: datetime = field(default_factory=datetime.now)
    camera_mode: str = ""
    uptime_seconds: float = 0.0
    host: str = ""  # "ip:port" advertised by the edge for proxy routes
    strategy: str = ""
    pipeline: str = ""
    camera_count: int = 0
    running: bool = True
    auto_classify: bool = False
    warnings: list[dict] = field(default_factory=list)


_bin_registry: dict[str, BinInfo] = {}
_bin_lock = threading.Lock()

_BIN_ONLINE_TIMEOUT = 60  # seconds — bin considered offline after this


def _update_bin(hb: BinHeartbeat) -> None:
    with _bin_lock:
        _bin_registry[hb.bin_id] = BinInfo(
            bin_id=hb.bin_id,
            status=hb.status,
            last_seen=datetime.now(),
            camera_mode=hb.camera_mode,
            uptime_seconds=hb.uptime_seconds,
            host=hb.host,
            strategy=hb.strategy,
            pipeline=hb.pipeline or hb.camera_mode,
            camera_count=hb.camera_count,
            running=hb.running,
            auto_classify=hb.auto_classify,
            warnings=[w.model_dump() for w in hb.warnings],
        )


def _get_bin_info(bin_id: str) -> BinInfo | None:
    with _bin_lock:
        return _bin_registry.get(bin_id)


def _get_bin_status() -> list[dict]:
    now = datetime.now()
    with _bin_lock:
        out: list[dict] = []
        for info in _bin_registry.values():
            stale = (now - info.last_seen).total_seconds() >= _BIN_ONLINE_TIMEOUT
            if stale:
                live_status = "offline"
            elif not info.running:
                live_status = "stopped"
            else:
                live_status = info.status or "online"
            out.append(
                {
                    "bin_id": info.bin_id,
                    "status": live_status,
                    "last_seen": info.last_seen.isoformat(),
                    "camera_mode": info.camera_mode,
                    "uptime_seconds": info.uptime_seconds,
                    "host": info.host,
                    "strategy": info.strategy,
                    "pipeline": info.pipeline,
                    "camera_count": info.camera_count,
                    "running": info.running,
                    "auto_classify": info.auto_classify,
                    "warnings": list(info.warnings),
                }
            )
        return out


def _derive_alerts() -> dict:
    """Camera-availability alerts derived from the live bin registry.

    Live-only by design: a bin that never heartbeats (or went stale) is the
    Fleet page's offline problem, not a camera alert. An oak-native pipeline
    legitimately runs one camera but still gets the single-camera warning —
    the spec is "alert below 2 cameras" for every reporting bin.
    """
    alerts: list[dict] = []
    monitored = 0
    for bin_info in _get_bin_status():
        if bin_info["status"] == "offline":
            continue
        monitored += 1
        cameras = bin_info["camera_count"] or 0
        if cameras >= 2:
            continue
        severity, code, message = (
            ("error", "NO_CAMERA", "No camera detected")
            if cameras == 0
            else ("warning", "SINGLE_CAMERA", "Only one camera available")
        )
        alerts.append(
            {
                "bin_id": bin_info["bin_id"],
                "severity": severity,
                "code": code,
                "message": message,
                "camera_count": cameras,
                "pipeline": bin_info["pipeline"],
                "host": bin_info["host"],
                "last_seen": bin_info["last_seen"],
            }
        )
    alerts.sort(key=lambda a: (a["severity"] != "error", a["bin_id"]))
    errors = sum(1 for a in alerts if a["severity"] == "error")
    return {
        "alerts": alerts,
        "counts": {
            "error": errors,
            "warning": len(alerts) - errors,
            "total": len(alerts),
            "monitored": monitored,
        },
    }


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


# ── Camera loops ─────────────────────────────────────────────────────────────


def _camera_loop_oak() -> None:
    """Capture frames from dual OAK cameras in a background thread."""
    global _cameras_ok

    try:
        import depthai as dai  # noqa: I001

        from .cameraOak import crop_sides, make_pipeline
    except Exception as e:
        logger.error("Cannot import OAK camera modules: %s", e)
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
        logger.info("OAK camera thread started with %d devices", len(devices))

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
            logger.error("OAK camera loop error: %s", e)
        finally:
            for p in pipelines:
                with contextlib.suppress(Exception):
                    p.stop()


def _camera_loop_raspberry() -> None:
    """Capture frames from dual Raspberry Pi cameras in a background thread."""
    global _cameras_ok

    try:
        from .cameraraspberry import crop_sides, grab_frame, make_cameras, stop_cameras
    except Exception as e:
        logger.error("Cannot import Raspberry Pi camera modules: %s", e)
        return

    try:
        from picamera2 import Picamera2

        num_cameras = len(Picamera2.global_camera_info())
        if num_cameras < 2:
            logger.warning("Need 2 Pi cameras, found %d. Camera stream unavailable.", num_cameras)
            return
    except Exception as e:
        logger.error("picamera2 not available: %s", e)
        return

    cameras = make_cameras(2)
    _cameras_ok = True
    logger.info("Raspberry Pi camera thread started with %d cameras", len(cameras))

    try:
        while True:
            last_frames: list[np.ndarray | None] = [None, None]
            for i, cam in enumerate(cameras):
                frame = grab_frame(cam)
                frame = crop_sides(frame, CROP_PERCENT)
                frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
                last_frames[i] = frame

            if last_frames[0] is not None and last_frames[1] is not None:
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
        logger.error("Raspberry Pi camera loop error: %s", e)
    finally:
        stop_cameras(cameras)


def _camera_loop_oak_native() -> None:
    """Capture frames from a single OAK camera with sensor fusion in a background thread."""
    global _cameras_ok

    try:
        import depthai as dai  # noqa: I001

        from .oak_native import OAKOccupancyDetector
    except Exception as e:
        logger.error("Cannot import OAK native modules: %s", e)
        return

    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        logger.warning("No OAK device found. Camera stream unavailable.")
        return

    logger.info("Using OAK device: %s", infos[0].getDeviceId())

    with dai.Device(infos[0]) as device:
        detector = OAKOccupancyDetector(device)
        _cameras_ok = True
        _state.set_status("Calibrating", "Warming up sensors...")
        logger.info("OAK Native camera thread started")

        # State machine (mirrors mainoak.py)
        oak_state = "calibrating"
        detect_streak = 0
        empty_streak = 0
        last_check = 0.0

        try:
            while True:
                votes = detector.update()

                # Update frame for the web stream
                if votes.rgb_frame is not None:
                    disp = cv2.resize(
                        votes.rgb_frame,
                        (OAK_DISPLAY_W, OAK_DISPLAY_H),
                        interpolation=cv2.INTER_AREA,
                    )
                    draw_nn_detections(disp, votes.nn_detections)
                    _set_frame(disp)

                # State machine tick
                now = time.time()
                if now - last_check >= OAK_CHECK_INTERVAL:
                    last_check = now

                    if oak_state == "calibrating":
                        done = detector.calibrate()
                        pct = detector.calibration_progress()
                        _state.set_status("Calibrating", f"Warming up sensors... {pct}%")
                        if done:
                            oak_state = "ready"
                            _state.set_status("Ready", "Bin empty -- waiting for item.")
                            logger.info("Calibration complete.")

                    elif oak_state == "ready":
                        if votes.votes >= OAK_VOTES_NEEDED:
                            detect_streak += 1
                            _state.set_status(
                                "Detecting",
                                f"Object signals: {votes.votes} vote(s) ({detect_streak}/{OAK_DETECT_CONFIRM_N})",
                            )
                            if detect_streak >= OAK_DETECT_CONFIRM_N:
                                logger.info(
                                    "Occupancy confirmed (%d votes). Triggering classify.",
                                    votes.votes,
                                )
                                oak_state = "detected"
                                detect_streak = 0
                        else:
                            if detect_streak:
                                _state.set_status("Ready", "Bin empty -- waiting for item.")
                            detect_streak = 0

                    elif oak_state == "detected":
                        if votes.rgb_frame is not None and _state.start_classify():
                            img_bytes = encode_frame(votes.rgb_frame)
                            launch_classify(img_bytes, votes.rgb_frame.copy(), _state)
                            _state.set_status("Classifying...", "Sending to Gemini AI...")
                            oak_state = "classifying"

                    elif oak_state == "classifying":
                        if not _state.is_classifying:
                            logger.info("Classification complete.")
                            oak_state = "classified"

                    elif oak_state == "classified":
                        if votes.votes == 0:
                            empty_streak += 1
                            if empty_streak >= OAK_EMPTY_CONFIRM_N:
                                _state.set_status("Ready", "Bin cleared -- waiting for next item.")
                                logger.info("Bin empty confirmed -- returning to Ready.")
                                oak_state = "ready"
                                empty_streak = 0
                        else:
                            empty_streak = 0

                time.sleep(0.01)
        except Exception as e:
            logger.error("OAK-D Native camera loop error: %s", e)
        finally:
            detector.stop()


_CAMERA_LOOPS = {
    "oak": _camera_loop_oak,
    "raspberry": _camera_loop_raspberry,
    "oak-native": _camera_loop_oak_native,
}


def _start_camera_thread() -> None:
    if CAMERA_MODE == "none":
        logger.info("Camera mode is 'none' — running as server-only (no camera thread)")
        return
    loop_fn = _CAMERA_LOOPS.get(CAMERA_MODE)
    if loop_fn is None:
        logger.error(
            "Unknown SMARTWASTE_CAMERA_MODE=%r. Valid: %s, none",
            CAMERA_MODE,
            ", ".join(_CAMERA_LOOPS),
        )
        return
    logger.info("Starting camera thread in %r mode", CAMERA_MODE)
    t = threading.Thread(target=loop_fn, daemon=True, name="camera-thread")
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


# ── Routes: Auth ──────────────────────────────────────────────────────────────


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        request.session["user"] = username
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": "Invalid username or password"},
        status_code=401,
    )


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ── Routes: Public ────────────────────────────────────────────────────────────


@app.get("/site", response_class=HTMLResponse)
def site(request: Request):
    return templates.TemplateResponse(request=request, name="site.html")


# ── Routes: Protected dashboard ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"active": "devices", "user": request.session.get("user", "admin")},
    )


@app.get("/map", response_class=HTMLResponse)
def dashboard_map(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="dashboard_map.html",
        context={"active": "map", "user": request.session.get("user", "admin")},
    )


@app.get("/analytics", response_class=HTMLResponse)
def dashboard_analytics(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="dashboard_analytics.html",
        context={"active": "analytics", "user": request.session.get("user", "admin")},
    )


@app.get("/alerts", response_class=HTMLResponse)
def dashboard_alerts(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="dashboard_alerts.html",
        context={"active": "alerts", "user": request.session.get("user", "admin")},
    )


@app.get("/classifications", response_class=HTMLResponse)
def dashboard_classifications(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="dashboard_classifications.html",
        context={
            "active": "classifications",
            "user": request.session.get("user", "admin"),
            "categories": VALID_CLASSES,
        },
    )


@app.get("/bin/{bin_id}", response_class=HTMLResponse)
def bin_detail(request: Request, bin_id: str):
    if not _is_admin(request):
        return RedirectResponse("/login", status_code=302)

    is_local = bin_id == BIN_ID and CAMERA_MODE != "none" and _cameras_ok
    info = _get_bin_info(bin_id)
    remote_online = (
        info is not None
        and bool(info.host)
        and (datetime.now() - info.last_seen).total_seconds() < _BIN_ONLINE_TIMEOUT
    )

    if is_local:
        stream_url = "/stream"
        classify_url = "/api/classify"
        toggle_url = "/api/toggle-auto"
        state_url = "/api/state"
        has_camera = True
    elif remote_online:
        stream_url = f"/api/bin/{bin_id}/stream"
        classify_url = f"/api/bin/{bin_id}/classify"
        toggle_url = f"/api/bin/{bin_id}/toggle-auto"
        state_url = f"/api/bin/{bin_id}/state"
        has_camera = True
    else:
        stream_url = classify_url = toggle_url = state_url = ""
        has_camera = False

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "bin_id": bin_id,
            "has_local_camera": has_camera,
            "stream_url": stream_url,
            "classify_url": classify_url,
            "toggle_url": toggle_url,
            "state_url": state_url,
        },
    )


# ── Routes: Proxy to edge bins ────────────────────────────────────────────────


def _edge_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {EDGE_API_KEY}"} if EDGE_API_KEY else {}


def _proxy_stream(host: str):
    """Generator that yields chunks from the edge's MJPEG stream."""
    url = f"http://{host}/stream"
    req = urllib.request.Request(url, headers=_edge_headers())
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                yield chunk
    except Exception as exc:
        logger.warning("Proxy stream from %s failed: %s", host, exc)
        return


def _proxy_request(
    host: str, path: str, method: str = "GET", json_body: dict | None = None
) -> tuple[int, dict]:
    import json as _json

    url = f"http://{host}{path}"
    headers = _edge_headers()
    data: bytes | None = None
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        data = _json.dumps(json_body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                payload = _json.loads(body) if body else {}
            except Exception:
                payload = {"raw": body}
            return resp.status, payload
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            payload = _json.loads(err_body) if err_body else {}
        except Exception:
            payload = {"error": f"edge returned {e.code}"}
        return e.code, payload
    except Exception as e:
        return 502, {"error": f"edge unreachable: {e}"}


@app.get("/api/bin/{bin_id}/stream")
def proxy_bin_stream(request: Request, bin_id: str):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    info = _get_bin_info(bin_id)
    if info is None or not info.host:
        return JSONResponse({"error": "bin not registered"}, status_code=404)
    return StreamingResponse(
        _proxy_stream(info.host),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/bin/{bin_id}/classify")
def proxy_bin_classify(request: Request, bin_id: str):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    info = _get_bin_info(bin_id)
    if info is None or not info.host:
        return JSONResponse({"error": "bin not registered"}, status_code=404)
    status, data = _proxy_request(info.host, "/classify", method="POST")
    return JSONResponse(data, status_code=status)


@app.post("/api/bin/{bin_id}/toggle-auto")
def proxy_bin_toggle(request: Request, bin_id: str):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    info = _get_bin_info(bin_id)
    if info is None or not info.host:
        return JSONResponse({"error": "bin not registered"}, status_code=404)
    status, data = _proxy_request(info.host, "/toggle-auto", method="POST")
    return JSONResponse(data, status_code=status)


@app.get("/api/bin/{bin_id}/state")
def proxy_bin_state(request: Request, bin_id: str):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    info = _get_bin_info(bin_id)
    if info is None or not info.host:
        return JSONResponse({"error": "bin not registered"}, status_code=404)
    status, data = _proxy_request(info.host, "/state", method="GET")
    return JSONResponse(data, status_code=status)


@app.get("/api/bin/{bin_id}/diagnostics")
def proxy_bin_diagnostics(request: Request, bin_id: str):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    info = _get_bin_info(bin_id)
    if info is None or not info.host:
        return JSONResponse({"error": "bin not registered"}, status_code=404)
    status, data = _proxy_request(info.host, "/diagnostics", method="GET")
    return JSONResponse(data, status_code=status)


# ── Command proxy with simple per-bin rate limiting ──────────────────────────

_command_audit: list[dict] = []
_command_last_ts: dict[str, float] = {}
_COMMAND_MIN_INTERVAL = 1.0  # seconds between commands per bin
_command_lock = threading.Lock()


def _rate_limited(bin_id: str) -> bool:
    now = time.monotonic()
    with _command_lock:
        last = _command_last_ts.get(bin_id, 0.0)
        if now - last < _COMMAND_MIN_INTERVAL:
            return True
        _command_last_ts[bin_id] = now
        return False


@app.post("/api/bin/{bin_id}/command")
def proxy_bin_command(request: Request, bin_id: str, cmd: BinCommand):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    info = _get_bin_info(bin_id)
    if info is None or not info.host:
        return JSONResponse({"error": "bin not registered"}, status_code=404)
    if _rate_limited(bin_id):
        return JSONResponse(
            {"error": "rate limited — wait a moment between commands"}, status_code=429
        )
    status, data = _proxy_request(info.host, "/command", method="POST", json_body=cmd.model_dump())
    with _command_lock:
        _command_audit.append(
            {
                "bin_id": bin_id,
                "action": cmd.action,
                "value": cmd.value,
                "user": request.session.get("user", "api"),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": status,
            }
        )
        if len(_command_audit) > 200:
            del _command_audit[:-200]
    return JSONResponse(data, status_code=status)


@app.get("/api/audit")
def api_audit(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    with _command_lock:
        return list(reversed(_command_audit))


@app.get("/stream")
def video_feed(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Routes: Protected API ────────────────────────────────────────────────────


@app.post("/api/classify")
def api_classify(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    frame = _get_frame()
    if frame is None:
        return JSONResponse({"error": "No camera frame available"}, status_code=503)
    if not _state.start_classify():
        return JSONResponse({"error": "Classification already in progress"}, status_code=409)
    img_bytes = encode_frame(frame)
    launch_classify(img_bytes, frame.copy(), _state)
    return {"status": "classifying"}


@app.post("/api/toggle-auto")
def api_toggle_auto(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    new_state = _state.toggle_auto()
    return {"auto_classify": new_state}


@app.get("/api/state")
def api_state(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    label, detail, auto_on = _state.get_display()
    return {
        "label": label,
        "detail": detail,
        "auto_on": auto_on,
        "is_classifying": _state.is_classifying,
        "history": [{"time": ts, "label": lbl} for ts, lbl in _state.get_history()],
    }


def _entry_filters(
    bin_id: str | None,
    label: str | None,
    q: str | None,
    since: str | None,
    until: str | None,
) -> tuple[dict, JSONResponse | None]:
    """Validate/whitelist the shared /api/entries filter params."""
    if label and label not in VALID_CLASSES:
        return {}, JSONResponse({"error": "invalid label"}, status_code=400)
    for name, value in (("since", since), ("until", until)):
        if value:
            try:
                datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return {}, JSONResponse({"error": f"invalid {name}"}, status_code=400)
    q = (q or "").strip()[:80] or None
    return {"bin_id": bin_id, "label": label, "q": q, "since": since, "until": until}, None


@app.get("/api/entries")
def api_entries(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    bin_id: str | None = None,
    label: str | None = None,
    q: str | None = None,
    since: str | None = None,
    until: str | None = None,
):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    filters, err = _entry_filters(bin_id, label, q, since, until)
    if err:
        return err
    return get_entries(limit=max(1, min(limit, 200)), offset=max(0, offset), **filters)


@app.get("/api/entries/count")
def api_entries_count(
    request: Request,
    bin_id: str | None = None,
    label: str | None = None,
    q: str | None = None,
    since: str | None = None,
    until: str | None = None,
):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    filters, err = _entry_filters(bin_id, label, q, since, until)
    if err:
        return err
    return {"total": get_entry_count(**filters)}


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.get("/api/entries/{entry_id}/image")
def api_entry_image(request: Request, entry_id: int):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    entry = get_entry_by_id(entry_id)
    if not entry or not entry.get("filename"):
        return JSONResponse({"error": "not found"}, status_code=404)
    # The filename column stores absolute paths from whichever host ingested
    # the frame (Windows or Docker) — keep only the basename and re-anchor it
    # in this host's DATASET_DIR, then confine the resolved path to it.
    name = os.path.basename(str(entry["filename"]).replace("\\", "/"))
    if os.path.splitext(name)[1].lower() not in _IMAGE_EXTENSIONS:
        return JSONResponse({"error": "not found"}, status_code=404)
    root = os.path.realpath(DATASET_DIR)
    path = os.path.realpath(os.path.join(root, name))
    try:
        inside = os.path.commonpath([root, path]) == root
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/stats")
def api_stats(request: Request, bin_id: str | None = None):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {
        "total": get_entry_count(bin_id=bin_id),
        "by_category": get_label_counts(bin_id=bin_id),
    }


@app.get("/api/public/stats")
def api_public_stats():
    """Fleet-wide aggregate for the public presentation site — no per-bin data."""
    by_category = get_label_counts()
    sorted_total = sum(n for label, n in by_category.items() if label != "Empty")
    recyclable = sum(by_category.get(label, 0) for label in RECYCLABLE_CLASSES)
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    bins = _get_bin_status()

    latest = None
    rows = get_entries(limit=1)
    if rows:
        row = rows[0]
        item = (
            row.get("brand_product") or row.get("description") or row.get("label") or ""
        ).strip()
        if len(item) > 60:
            item = item[:59] + "…"
        # SQLite stores timestamps as TEXT; PostgreSQL returns aware datetimes.
        ago_seconds = None
        seen = row.get("timestamp")
        if isinstance(seen, str):
            try:
                seen = datetime.strptime(seen, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                seen = None
        if isinstance(seen, datetime):
            now = datetime.now(seen.tzinfo) if seen.tzinfo else datetime.now()
            ago_seconds = max(0, int((now - seen).total_seconds()))
        latest = {
            "category": row.get("label"),
            "item": item,
            "confidence": row.get("confidence"),
            "ago_seconds": ago_seconds,
        }

    return {
        "total": get_entry_count(),
        "today": get_entry_count(since=midnight.strftime("%Y-%m-%d %H:%M:%S")),
        "by_category": by_category,
        "recyclable_share": (recyclable / sorted_total) if sorted_total else None,
        "bins": {
            "online": sum(1 for b in bins if b["status"] != "offline"),
            "total": len(bins),
        },
        "latest": latest,
    }


@app.get("/api/bins")
def api_bins(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return get_active_bins()


@app.get("/api/alerts")
def api_alerts(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return _derive_alerts()


@app.get("/api/analytics")
def api_analytics(request: Request, period: str = "7d"):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if period not in analytics.PERIODS:
        return JSONResponse({"error": "invalid period"}, status_code=400)
    return analytics.build_payload(period)


@app.get("/api/analytics/export")
def api_analytics_export(request: Request, period: str = "7d"):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if period not in analytics.PERIODS:
        return JSONResponse({"error": "invalid period"}, status_code=400)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=analytics.EXPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(analytics.build_export_rows(period))
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="smartbin-classifications-{period}.csv"'
            )
        },
    )


# ── Routes: Edge reporting (used by edge devices) ────────────────────────────


@app.post("/api/report")
def api_report(request: Request, report: EdgeReport):
    if not _is_edge_client(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    entry = {
        "filename": "",
        "label": report.label,
        "description": report.description,
        "brand_product": report.brand_product,
        "location": report.location,
        "weight": report.weight,
        "timestamp": report.timestamp,
        "bin_id": report.bin_id,
        "confidence": report.confidence,
        "llm_backend": report.llm_backend,
    }
    env = {
        "simulated_temperature": report.simulated_temperature,
        "simulated_humidity": report.simulated_humidity,
        "simulated_vibration": report.simulated_vibration,
        "simulated_air_pollution": report.simulated_air_pollution,
        "simulated_smoke": report.simulated_smoke,
    }

    # Save image if provided
    if report.image_b64:
        try:
            img_data = base64.b64decode(report.image_b64)
            if len(img_data) > MAX_UPLOAD_BYTES:
                raise ValueError("image exceeds SMARTWASTE_MAX_UPLOAD_MB")
            os.makedirs(DATASET_DIR, exist_ok=True)
            filepath = os.path.join(
                DATASET_DIR, _safe_filename(report.label, report.bin_id, report.timestamp)
            )
            with open(filepath, "wb") as f:
                f.write(img_data)
            entry["filename"] = filepath
        except Exception as e:
            logger.warning("Failed to save edge image: %s", e)

    row_id = insert_entry(entry, env)
    if row_id is None:
        return JSONResponse(
            {"status": "error", "detail": "database insert failed"}, status_code=500
        )
    return {"status": "ok", "id": row_id}


# Bound concurrent LLM calls so slow local inference can't exhaust the
# request threadpool (sync routes each hold a worker thread while they run).
_llm_semaphore = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)


@app.post("/api/edge/classify", response_model=EdgeClassifyResponse)
def api_edge_classify(request: Request, req: EdgeClassifyRequest):
    """Classify a frame POSTed by an edge device.

    This is the End Device → Server → LLM → Server → End Device round-trip:
    the server runs the configured LLM backend on the image, persists the
    result (dashboard picks it up on its next poll), and returns the
    classification plus the actuation command in the same HTTP response.
    """
    if not _is_edge_client(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        img_data = base64.b64decode(req.image_b64, validate=True)
    except Exception:
        return JSONResponse({"error": "invalid base64 image"}, status_code=400)
    if len(img_data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "image too large"}, status_code=413)

    if not _llm_semaphore.acquire(timeout=LLM_QUEUE_TIMEOUT):
        return JSONResponse({"error": "classifier busy — try again later"}, status_code=503)
    try:
        result = build_backend().classify(img_data)
    except CircuitOpenError as e:
        return JSONResponse({"error": f"classifier paused: {e}"}, status_code=503)
    except Exception as e:
        logger.error("Edge classify failed: %s", e)
        return JSONResponse({"error": f"classification failed: {e}"}, status_code=502)
    finally:
        _llm_semaphore.release()

    module = resolve_module(result.category)
    command = {
        "action": "open_module" if module is not None else "none",
        "module": module,
        "category": result.category,
    }

    status = "ok"
    row_id = None
    if result.category != "Empty":
        timestamp = req.captured_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "filename": "",
            "label": result.category,
            "description": result.description,
            "brand_product": result.brand_product,
            "location": req.location,
            "weight": req.weight,
            "timestamp": timestamp,
            "bin_id": req.bin_id,
            "confidence": result.confidence,
            "llm_backend": result.backend,
        }
        env = {
            "simulated_temperature": req.simulated_temperature,
            "simulated_humidity": req.simulated_humidity,
            "simulated_vibration": req.simulated_vibration,
            "simulated_air_pollution": req.simulated_air_pollution,
            "simulated_smoke": req.simulated_smoke,
        }
        try:
            os.makedirs(DATASET_DIR, exist_ok=True)
            filepath = os.path.join(
                DATASET_DIR, _safe_filename(result.category, req.bin_id, timestamp)
            )
            with open(filepath, "wb") as f:
                f.write(img_data)
            entry["filename"] = filepath
        except Exception as e:
            logger.warning("Failed to save edge image: %s", e)

        row_id = insert_entry(entry, env)
        if row_id is None:
            # Classified fine but not persisted — still return the command so
            # the bin can open the right module (contrast with /api/report,
            # which is persistence-only and therefore 500s).
            status = "db_error"

    return {
        "status": status,
        "id": row_id,
        "result": {
            "category": result.category,
            "description": result.description,
            "brand_product": result.brand_product,
            "confidence": result.confidence,
            "backend": result.backend,
            "escalated": result.escalated,
        },
        "command": command,
    }


@app.post("/api/heartbeat")
def api_heartbeat(request: Request, hb: BinHeartbeat):
    if not _is_edge_client(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    _update_bin(hb)
    return {"status": "ok"}


@app.get("/api/dashboard")
def api_dashboard(request: Request):
    """Combined bin registry + database stats for the dashboard page."""
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    # Merge heartbeat registry with DB data
    db_bins = {b["bin_id"]: b for b in get_active_bins()}
    heartbeat_bins = {b["bin_id"]: b for b in _get_bin_status()}

    all_bin_ids = set(db_bins.keys()) | set(heartbeat_bins.keys())
    bins = []
    for bid in sorted(all_bin_ids):
        db = db_bins.get(bid, {})
        hb = heartbeat_bins.get(bid, {})
        bins.append(
            {
                "bin_id": bid,
                "location": db.get("location", ""),
                "status": hb.get("status", "offline"),
                "last_seen": hb.get("last_seen", ""),
                "last_timestamp": db.get("last_timestamp", ""),
                "total_entries": db.get("total", 0),
                "camera_mode": hb.get("camera_mode", ""),
                "strategy": hb.get("strategy", ""),
                "pipeline": hb.get("pipeline", hb.get("camera_mode", "")),
                "camera_count": hb.get("camera_count", 0),
                "running": hb.get("running", True),
                "auto_classify": hb.get("auto_classify", False),
                "warnings": hb.get("warnings", []),
                "has_host": bool(hb.get("host", "")),
            }
        )

    online = sum(1 for b in bins if b["status"] == "online")
    degraded = sum(1 for b in bins if b["status"] == "degraded")
    offline = sum(1 for b in bins if b["status"] == "offline")

    return {
        "bins": bins,
        "total_bins": len(bins),
        "total_entries": get_entry_count(),
        "online": online,
        "degraded": degraded,
        "offline": offline,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("smartwaste.web:app", host=WEB_HOST, port=WEB_PORT)
