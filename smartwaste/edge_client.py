"""
smartwaste/edge_client.py — HTTP client for edge-to-server communication.

When EDGE_MODE is enabled, the edge device POSTs classification results
and periodic heartbeats to the central server.  Uses urllib (stdlib) to
avoid adding requests/httpx as a dependency on the edge image.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

from .config import (
    BIN_ID,
    CAMERA_MODE,
    CLASSIFY_TIMEOUT,
    EDGE_API_KEY,
    HEARTBEAT_INTERVAL,
    LOCATION,
    SERVER_URL,
    WEB_PORT,
)
from .log_setup import get_logger
from .state import AppState

logger = get_logger()

_start_time = time.monotonic()
_cached_host: str | None = None
_state_ref: AppState | None = None


def _detect_local_host() -> str:
    """
    Return "lan-ip:port" this edge is reachable on from the central server.

    Opens a dummy UDP socket to the server to discover which local interface
    will carry the traffic — this is the IP the server sees. Cached after the
    first successful call.
    """
    global _cached_host
    if _cached_host is not None:
        return _cached_host

    # Explicit override — required when running inside Docker (container IP
    # is not reachable from the laptop; set this to the Pi host's LAN IP).
    override = os.environ.get("SMARTWASTE_EDGE_HOST", "").strip()
    if override:
        _cached_host = override if ":" in override else f"{override}:{WEB_PORT}"
        return _cached_host

    server_host = ""
    try:
        parsed = urllib.parse.urlparse(SERVER_URL)
        server_host = parsed.hostname or ""
    except Exception:
        pass

    ip = ""
    if server_host:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(1.0)
                s.connect((server_host, 80))
                ip = s.getsockname()[0]
        except Exception as exc:
            logger.debug("LAN IP detection via server failed: %s", exc)

    if not ip:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = ""

    if ip and not ip.startswith("127."):
        _cached_host = f"{ip}:{WEB_PORT}"
        return _cached_host
    return ""


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EDGE_API_KEY}",
    }


def _post(path: str, payload: dict) -> bool:
    """POST JSON to the server. Returns True on success."""
    url = f"{SERVER_URL.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status < 300:
                return True
            logger.warning("Edge POST %s returned %s", path, resp.status)
            return False
    except urllib.error.HTTPError as e:
        logger.warning("Edge POST %s HTTP error: %s", path, e.code)
        return False
    except Exception as e:
        logger.warning("Edge POST %s failed: %s", path, e)
        return False


def report_classification(entry: dict, env: dict, image_bytes: bytes | None = None) -> bool:
    """POST a classification result to the central server."""
    payload = {
        "bin_id": entry.get("bin_id", BIN_ID),
        "label": entry.get("label", ""),
        "description": entry.get("description", ""),
        "brand_product": entry.get("brand_product", ""),
        "location": entry.get("location", ""),
        "weight": entry.get("weight", ""),
        "timestamp": entry.get("timestamp", ""),
        "simulated_temperature": env.get("simulated_temperature", 0.0),
        "simulated_humidity": env.get("simulated_humidity", 0.0),
        "simulated_vibration": env.get("simulated_vibration", 0.0),
        "simulated_air_pollution": env.get("simulated_air_pollution", 0.0),
        "simulated_smoke": env.get("simulated_smoke", 0.0),
        "confidence": entry.get("confidence"),
        "llm_backend": entry.get("llm_backend", ""),
    }
    if image_bytes:
        payload["image_b64"] = base64.b64encode(image_bytes).decode("ascii")

    ok = _post("/api/report", payload)
    if ok:
        logger.info("Edge: reported classification '%s' to server", entry.get("label"))
    return ok


class EdgeServerError(Exception):
    """The central server could not classify a frame (network error, timeout,
    or non-2xx response)."""


def classify_remote(
    img_bytes: bytes,
    env: dict | None = None,
    timeout: float | None = None,
) -> dict:
    """POST a frame to the central server for classification.

    Returns the parsed EdgeClassifyResponse dict:
    ``{"status", "id", "result": {...}, "command": {"action", "module", ...}}``.

    Unlike ``_post`` (which swallows errors — fine for heartbeats), this raises
    EdgeServerError so the caller knows the frame was NOT classified.
    """
    if not SERVER_URL:
        raise EdgeServerError("SMARTWASTE_SERVER_URL is not configured")

    env = env or {}
    payload = {
        "bin_id": BIN_ID,
        "image_b64": base64.b64encode(img_bytes).decode("ascii"),
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": LOCATION,
        "weight": "",
        "simulated_temperature": env.get("simulated_temperature", 0.0),
        "simulated_humidity": env.get("simulated_humidity", 0.0),
        "simulated_vibration": env.get("simulated_vibration", 0.0),
        "simulated_air_pollution": env.get("simulated_air_pollution", 0.0),
        "simulated_smoke": env.get("simulated_smoke", 0.0),
    }
    url = f"{SERVER_URL.rstrip('/')}/api/edge/classify"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout or CLASSIFY_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise EdgeServerError(f"server returned HTTP {e.code}: {detail}") from e
    except Exception as e:
        raise EdgeServerError(str(e)) from e

    if not isinstance(body, dict):
        raise EdgeServerError("unexpected server response (not a JSON object)")
    return body


def _derive_status(state: AppState | None) -> str:
    if state is None:
        return "online"
    if not state.running:
        return "stopped"
    warnings = state.warnings.list()
    if any(w["severity"] == "error" for w in warnings):
        return "degraded"
    if any(w["severity"] == "warning" for w in warnings):
        return "degraded"
    return "online"


def send_heartbeat() -> bool:
    """Send a single heartbeat to the server."""
    state = _state_ref
    payload = {
        "bin_id": BIN_ID,
        "status": _derive_status(state),
        "camera_mode": CAMERA_MODE,
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "host": _detect_local_host(),
        "strategy": state.get_strategy() if state else "",
        "pipeline": state.get_pipeline() if state else CAMERA_MODE,
        "camera_count": state.get_camera_count() if state else 0,
        "running": bool(state.running) if state else True,
        "auto_classify": bool(state.auto_classify) if state else False,
        "warnings": state.warnings.list() if state else [],
    }
    return _post("/api/heartbeat", payload)


def _heartbeat_loop() -> None:
    """Background loop that sends heartbeats at HEARTBEAT_INTERVAL."""
    while True:
        try:
            send_heartbeat()
        except Exception as exc:
            logger.warning("Heartbeat send failed: %s", exc)
        time.sleep(HEARTBEAT_INTERVAL)


def start_heartbeat_thread(state: AppState | None = None) -> None:
    """Start the heartbeat daemon thread.

    Pass the shared AppState so heartbeats can include live strategy /
    warnings / camera_count for the admin dashboard.
    """
    global _state_ref
    if state is not None:
        _state_ref = state
    if not SERVER_URL:
        logger.warning("Edge: SERVER_URL not set, heartbeat disabled")
        return
    t = threading.Thread(target=_heartbeat_loop, daemon=True, name="edge-heartbeat")
    t.start()
    logger.info("Edge: heartbeat thread started (interval=%ds)", HEARTBEAT_INTERVAL)
