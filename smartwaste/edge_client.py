"""
smartwaste/edge_client.py — HTTP client for edge-to-server communication.

When EDGE_MODE is enabled, the edge device POSTs classification results
and periodic heartbeats to the central server.  Uses urllib (stdlib) to
avoid adding requests/httpx as a dependency on the edge image.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request

from .config import BIN_ID, CAMERA_MODE, EDGE_API_KEY, HEARTBEAT_INTERVAL, SERVER_URL
from .log_setup import get_logger

logger = get_logger()

_start_time = time.monotonic()


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


def report_classification(
    entry: dict, env: dict, image_bytes: bytes | None = None
) -> bool:
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
    }
    if image_bytes:
        payload["image_b64"] = base64.b64encode(image_bytes).decode("ascii")

    ok = _post("/api/report", payload)
    if ok:
        logger.info("Edge: reported classification '%s' to server", entry.get("label"))
    return ok


def send_heartbeat() -> bool:
    """Send a single heartbeat to the server."""
    payload = {
        "bin_id": BIN_ID,
        "status": "online",
        "camera_mode": CAMERA_MODE,
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
    }
    return _post("/api/heartbeat", payload)


def _heartbeat_loop() -> None:
    """Background loop that sends heartbeats at HEARTBEAT_INTERVAL."""
    while True:
        send_heartbeat()
        time.sleep(HEARTBEAT_INTERVAL)


def start_heartbeat_thread() -> None:
    """Start the heartbeat daemon thread."""
    if not SERVER_URL:
        logger.warning("Edge: SERVER_URL not set, heartbeat disabled")
        return
    t = threading.Thread(target=_heartbeat_loop, daemon=True, name="edge-heartbeat")
    t.start()
    logger.info("Edge: heartbeat thread started (interval=%ds)", HEARTBEAT_INTERVAL)
