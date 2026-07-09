"""
smartwaste/actuator.py — pluggable bin-module actuation.

When a classification decides which physical module should open ("open module
2"), the command lands here. The default implementation only logs — a real
hardware driver (GPIO servo/stepper, serial link to a motor controller, ...)
drops in later:

  1. Implement the ``Actuator`` protocol (an ``open_module`` method).
  2. Register the class in ``_ACTUATORS`` under a new name (e.g. ``"gpio"``).
  3. Set ``SMARTWASTE_ACTUATOR=gpio`` on the edge device.

The category → module mapping comes from ``SMARTWASTE_MODULE_MAP`` (JSON) and
defaults to ``config.DEFAULT_MODULE_MAP``. "Empty" never opens a module.
"""

from __future__ import annotations

import threading
from typing import Protocol

from .config import ACTUATOR, MODULE_MAP
from .log_setup import get_logger

logger = get_logger()


class Actuator(Protocol):
    def open_module(self, module: int, category: str) -> None: ...


class LogActuator:
    """Default actuator: log the command (no hardware attached yet)."""

    def open_module(self, module: int, category: str) -> None:
        logger.info("ACTUATE: open module %d (%s)", module, category)


class NullActuator:
    """Discard actuation commands entirely (SMARTWASTE_ACTUATOR=none)."""

    def open_module(self, module: int, category: str) -> None:
        pass


_ACTUATORS: dict[str, type] = {
    "log": LogActuator,
    "none": NullActuator,
    # "gpio": GpioActuator,  ← future hardware driver registers here
}

_instance: Actuator | None = None
_instance_lock = threading.Lock()


def get_actuator() -> Actuator:
    """Return the configured actuator (singleton)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                cls = _ACTUATORS.get(ACTUATOR)
                if cls is None:
                    logger.warning("Unknown SMARTWASTE_ACTUATOR %r — using 'log'.", ACTUATOR)
                    cls = LogActuator
                _instance = cls()
    return _instance


def resolve_module(category: str) -> int | None:
    """Map a waste category to its physical module number (None = keep closed)."""
    return MODULE_MAP.get(category)


def dispatch(category: str, module: int | None = None) -> None:
    """Open the module for *category* (resolving it when *module* is None).

    Never raises — a broken actuator must not kill the classification worker.
    """
    try:
        if module is None:
            module = resolve_module(category)
        if module is None:
            return
        get_actuator().open_module(int(module), category)
    except Exception as exc:
        logger.error("Actuator dispatch failed: %s", exc)
