"""
smartwaste/schemas.py — Pydantic models for edge-to-server communication.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class WarningInfo(BaseModel):
    """One active warning on the edge device."""

    code: str
    severity: str = "warning"
    message: str = ""
    first_seen: str = ""
    last_seen: str = ""


class BinHeartbeat(BaseModel):
    """Periodic heartbeat from an edge device to signal it is alive."""

    bin_id: str
    status: str = "online"  # "online" | "idle" | "degraded" | "stopped"
    camera_mode: str = ""
    uptime_seconds: float = 0.0
    host: str = ""  # "lan-ip:port" where the edge HTTP server is reachable

    # New in v2 — surfaced on the admin dashboard
    strategy: str = ""          # "manual" | "auto" | "oak-native"
    pipeline: str = ""          # "oak" | "oak-native"
    camera_count: int = 0
    running: bool = True
    auto_classify: bool = False
    warnings: list[WarningInfo] = Field(default_factory=list)


BinCommandAction = Literal[
    "stop",
    "start",
    "restart",
    "set_strategy",
    "set_pipeline",
    "classify",
    "toggle_auto",
    "clear_warnings",
]


class BinCommand(BaseModel):
    """A control command sent from the admin dashboard to an edge device."""

    action: BinCommandAction
    value: str | None = None  # strategy/pipeline name for set_* actions


class CommandResult(BaseModel):
    """Edge response to a BinCommand."""

    status: str = "ok"
    message: str = ""
