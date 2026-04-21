"""
smartwaste/warnings.py — structured runtime warnings for the admin UI.

Warnings are surfaced on the central dashboard (via heartbeats) so the admin
can see bin health at a glance without tailing logs.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"

_VALID_SEVERITIES = {SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR}


@dataclass
class Warning:
    """A structured warning.  Deduped in the registry by ``code``."""

    code: str
    severity: str
    message: str
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def as_dict(self) -> dict:
        return asdict(self)


class WarningRegistry:
    """Thread-safe dedup'd registry of active warnings keyed by code."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Warning] = {}

    def add(self, code: str, message: str, severity: str = SEVERITY_WARNING) -> None:
        if severity not in _VALID_SEVERITIES:
            severity = SEVERITY_WARNING
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            existing = self._items.get(code)
            if existing is not None:
                existing.message = message
                existing.severity = severity
                existing.last_seen = now
            else:
                self._items[code] = Warning(code=code, severity=severity, message=message)

    def clear(self, code: str) -> None:
        with self._lock:
            self._items.pop(code, None)

    def clear_all(self) -> None:
        with self._lock:
            self._items.clear()

    def list(self) -> list[dict]:
        with self._lock:
            return [w.as_dict() for w in self._items.values()]
