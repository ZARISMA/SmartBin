"""
hexabin/camera_config.py — per-camera geometry (rotate / flip / crop).

Replaces the single global symmetric ``CROP_PERCENT`` side-crop with a
per-camera transform an operator can edit from the dashboard and save.

Two pieces:

* ``CameraConfig`` — a pure, validated description of one camera's transform
  (rotation in 90° steps, optional H/V flip, a normalized crop rectangle) plus
  ``apply_transform`` which applies it to a BGR frame. Order is **rotate → flip
  → crop**; the crop is stored in ``[0, 1]`` coordinates of the *rotated+flipped*
  frame so it is resolution-independent and round-trips through the editor.
* ``CameraConfigStore`` — a thread-safe holder mapping camera index → config,
  the latest *raw* (untransformed) frame per camera for the editor preview, and
  JSON load/save so a saved edit survives a process restart.

The default config (``default_config``) is derived from ``CROP_PERCENT`` so an
unedited camera behaves exactly as it did before this module existed.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CROP_PERCENT
from .log_setup import get_logger

logger = get_logger()

VALID_ROTATIONS = (0, 90, 180, 270)

#: A crop side must keep at least this fraction of the frame — guards against a
#: degenerate zero-area rectangle that would yield an empty image.
MIN_CROP_SIZE = 0.05

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass(frozen=True)
class CameraConfig:
    """One camera's geometry transform.

    ``crop`` is ``(x0, y0, x1, y1)`` in ``[0, 1]`` of the rotated+flipped frame,
    half-open like a pixel slice: ``x0 < x1`` and ``y0 < y1``.
    """

    rotation: int = 0
    flip_h: bool = False
    flip_v: bool = False
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def validate(self) -> "CameraConfig":
        """Return self if valid, else raise ValueError with a clear message."""
        if self.rotation not in VALID_ROTATIONS:
            raise ValueError(f"rotation must be one of {VALID_ROTATIONS}, got {self.rotation!r}")
        if len(self.crop) != 4:
            raise ValueError("crop must be 4 numbers (x0, y0, x1, y1)")
        x0, y0, x1, y1 = self.crop
        for name, v in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1)):
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"crop {name} must be a number in [0, 1], got {v!r}")
        if x1 - x0 < MIN_CROP_SIZE or y1 - y0 < MIN_CROP_SIZE:
            raise ValueError(f"crop is too small — keep at least {MIN_CROP_SIZE:.0%} on each axis")
        return self

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
            "crop": list(self.crop),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        """Build (and validate) a config from a plain dict (API/JSON input)."""
        if not isinstance(data, dict):
            raise ValueError("camera config must be an object")
        crop = data.get("crop", (0.0, 0.0, 1.0, 1.0))
        try:
            crop_t = tuple(float(v) for v in crop)
        except (TypeError, ValueError):
            raise ValueError("crop must be a list of 4 numbers")
        cfg = cls(
            rotation=int(data.get("rotation", 0)),
            flip_h=bool(data.get("flip_h", False)),
            flip_v=bool(data.get("flip_v", False)),
            crop=crop_t,  # type: ignore[arg-type]
        )
        return cfg.validate()


def default_config() -> CameraConfig:
    """The transform that reproduces the legacy behaviour: no rotation, no flip,
    a symmetric ``CROP_PERCENT`` side-crop (top/bottom untouched)."""
    return CameraConfig(crop=(CROP_PERCENT, 0.0, 1.0 - CROP_PERCENT, 1.0))


def apply_transform(frame: np.ndarray, cfg: CameraConfig | None) -> np.ndarray:
    """Apply *cfg* to a BGR *frame*: rotate → flip → crop.

    A ``None`` config means "leave the frame untouched" (used when a camera has
    no meaningful raw frame yet). Invalid configs fall back to no-op rather than
    raising, so a bad on-disk value can never crash the capture loop.
    """
    if frame is None or cfg is None:
        return frame

    out = frame
    code = _ROTATE_CODES.get(cfg.rotation)
    if code is not None:
        out = cv2.rotate(out, code)

    if cfg.flip_h and cfg.flip_v:
        out = cv2.flip(out, -1)
    elif cfg.flip_h:
        out = cv2.flip(out, 1)
    elif cfg.flip_v:
        out = cv2.flip(out, 0)

    h, w = out.shape[:2]
    x0f, y0f, x1f, y1f = cfg.crop
    x0 = max(0, min(int(round(x0f * w)), w - 1))
    x1 = max(x0 + 1, min(int(round(x1f * w)), w))
    y0 = max(0, min(int(round(y0f * h)), h - 1))
    y1 = max(y0 + 1, min(int(round(y1f * h)), h))
    return out[y0:y1, x0:x1]


class CameraConfigStore:
    """Thread-safe map of camera index → CameraConfig, with raw-frame stash.

    The capture loop calls ``set_raw(i, frame)`` with the *untransformed* frame
    each iteration (feeds the editor preview) and ``get(i)`` to fetch the config
    to apply. The dashboard mutates configs via ``set`` and persists with
    ``save_json``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._configs: dict[int, CameraConfig] = {}
        self._raw: dict[int, np.ndarray] = {}

    # ── config ────────────────────────────────────────────────────────────────

    def get(self, index: int) -> CameraConfig:
        with self._lock:
            return self._configs.get(index) or default_config()

    def set(self, index: int, cfg: CameraConfig) -> None:
        with self._lock:
            self._configs[index] = cfg.validate()

    def all(self) -> dict[int, CameraConfig]:
        with self._lock:
            return dict(self._configs)

    def reset(self, index: int) -> None:
        """Drop any saved config for *index* — reverts it to the default."""
        with self._lock:
            self._configs.pop(index, None)

    # ── raw frames (editor preview) ────────────────────────────────────────────

    def set_raw(self, index: int, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        with self._lock:
            self._raw[index] = frame.copy()

    def get_raw(self, index: int) -> np.ndarray | None:
        with self._lock:
            f = self._raw.get(index)
            return None if f is None else f.copy()

    # ── persistence ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {"cameras": {str(i): c.to_dict() for i, c in self._configs.items()}}

    def load_dict(self, data: dict) -> None:
        """Replace configs from a plain dict; bad per-camera entries are skipped."""
        cameras = (data or {}).get("cameras", {})
        parsed: dict[int, CameraConfig] = {}
        for key, raw in cameras.items():
            try:
                parsed[int(key)] = CameraConfig.from_dict(raw)
            except (ValueError, TypeError) as exc:
                logger.warning("Ignoring invalid camera config for %r: %s", key, exc)
        with self._lock:
            self._configs = parsed

    def load_json(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                self.load_dict(json.load(f))
            logger.info("Loaded camera config from %s", path)
        except FileNotFoundError:
            logger.info("No camera config at %s — using defaults.", path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read camera config %s: %s", path, exc)

    def save_json(self, path: str) -> None:
        import os

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("Saved camera config to %s", path)
        except OSError as exc:
            logger.error("Failed to write camera config %s: %s", path, exc)
