"""
smartwaste/schemas.py — Pydantic models for edge-to-server communication.
"""

from __future__ import annotations

from pydantic import BaseModel


class EdgeReport(BaseModel):
    """Classification result sent from an edge device to the central server."""

    bin_id: str
    label: str
    description: str = "N/A"
    brand_product: str = "Unknown"
    location: str = ""
    weight: str = ""
    timestamp: str  # ISO format  e.g. "2026-04-14 10:30:00"
    simulated_temperature: float = 0.0
    simulated_humidity: float = 0.0
    simulated_vibration: float = 0.0
    simulated_air_pollution: float = 0.0
    simulated_smoke: float = 0.0
    image_b64: str | None = None  # optional base64-encoded JPEG


class BinHeartbeat(BaseModel):
    """Periodic heartbeat from an edge device to signal it is alive."""

    bin_id: str
    status: str = "online"  # "online" or "idle"
    camera_mode: str = ""
    uptime_seconds: float = 0.0
