from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
_AA = cv2.LINE_AA

# ── Color palette (BGR) ───────────────────────────────────────────────────────
_C_WHITE = (255, 255, 255)
_C_GRAY = (140, 140, 140)       # Stone Gray #8C8C8C
_C_DARK_PANEL = (20, 26, 11)    # Dark green-tinted panel
_C_ACCENT = (66, 90, 45)        # Forest Green #2D5A42 (BGR)
_C_MODE_AUTO = (80, 175, 76)    # Success #4CAF50 (BGR)
_C_MODE_MANUAL = (107, 77, 26)  # Deep Smart Blue #1A4D6B (BGR)

# Per-category label colors (BGR) — brand modular system colors
_CAT_COLOR: dict[str, tuple[int, int, int]] = {
    "Plastic": (235, 206, 135),   # #87CEEB
    "Glass": (208, 224, 64),      # #40E0D0
    "Paper": (140, 180, 210),     # #D2B48C
    "Organic": (43, 77, 30),      # #1E4D2B
    "Aluminum": (169, 169, 169),  # #A9A9A9
    "Other": (219, 112, 147),     # #9370DB
    "Empty": (140, 140, 140),     # #8C8C8C
}

# ── Layout proportions (relative to frame dimensions) ─────────────────────────
_PANEL_ALPHA = 0.72  # panel opacity
_BAR_H_RATIO = 0.19  # top panel height / frame height
_MARGIN_RATIO = 0.016  # side / top margin / frame width
_HIST_X_RATIO = 0.76  # history column left edge / frame width


def _cat_color(label: str) -> tuple[int, int, int]:
    return _CAT_COLOR.get(label, _C_WHITE)


def draw_overlay(
    img: np.ndarray,
    label: str,
    detail: str,
    auto_on: bool,
    history: list[tuple[str, str]] | None = None,
) -> None:
    """Draw a responsive, semi-transparent HUD overlay onto *img* in-place.

    Parameters
    ----------
    img:     frame to annotate (modified in-place)
    label:   current classification label (e.g. "Plastic") or status string
    detail:  short descriptor / brand string
    auto_on: True when auto-classify mode is active
    history: optional list of (time_str, label) pairs, newest-first
    """
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return

    m = max(int(w * _MARGIN_RATIO), 4)
    bar_h = max(int(h * _BAR_H_RATIO), 40)

    # Font scales — proportional to width, with a floor
    base = max(w / 1600.0, 0.35)
    fs_title = base * 0.82
    fs_label = base * 0.76
    fs_small = base * 0.55
    fs_hist = base * 0.50

    # Line heights derived from actual font metrics
    (_, lh_t), _ = cv2.getTextSize("Ay", _FONT_BOLD, fs_title, 2)
    (_, lh_l), _ = cv2.getTextSize("Ay", _FONT_BOLD, fs_label, 2)
    (_, lh_s), _ = cv2.getTextSize("Ay", _FONT, fs_small, 1)
    (_, lh_h), _ = cv2.getTextSize("Ay", _FONT, fs_hist, 1)

    # ── 1. Semi-transparent top panel ─────────────────────────────────────────
    panel = img[:bar_h].copy()
    cv2.rectangle(panel, (0, 0), (w, bar_h), _C_DARK_PANEL, -1)
    img[:bar_h] = cv2.addWeighted(panel, _PANEL_ALPHA, img[:bar_h], 1.0 - _PANEL_ALPHA, 0)

    # ── 2. Accent divider line at panel bottom ─────────────────────────────────
    cv2.line(img, (0, bar_h - 1), (w, bar_h - 1), _C_ACCENT, 1, _AA)

    # ── 3. Title ───────────────────────────────────────────────────────────────
    y = m + lh_t
    cv2.putText(img, "Smart Waste AI", (m, y), _FONT_BOLD, fs_title, _C_ACCENT, 2, _AA)

    # ── 4. Category / status label with color dot ──────────────────────────────
    y += lh_t // 2 + lh_l + 2
    cat_col = _cat_color(label)
    dot_r = max(lh_l // 2 - 1, 3)
    dot_cx = m + dot_r + 2
    dot_cy = y - dot_r
    cv2.circle(img, (dot_cx, dot_cy), dot_r, cat_col, -1, _AA)
    cv2.putText(
        img, label or "Ready", (dot_cx + dot_r + 5, y), _FONT_BOLD, fs_label, cat_col, 2, _AA
    )

    # ── 5. Detail line ─────────────────────────────────────────────────────────
    y += lh_s + lh_s // 2
    if y < bar_h - 2:
        cv2.putText(img, detail, (dot_cx + dot_r + 5, y), _FONT, fs_small, _C_GRAY, 1, _AA)

    # ── 6. Mode badge (top-right corner) ──────────────────────────────────────
    mode_text = "\u25cf AUTO" if auto_on else "\u25cb MANUAL"
    mode_col = _C_MODE_AUTO if auto_on else _C_MODE_MANUAL
    (tw, _), _ = cv2.getTextSize(mode_text, _FONT_BOLD, fs_small, 2)
    cv2.putText(img, mode_text, (w - tw - m, m + lh_s), _FONT_BOLD, fs_small, mode_col, 2, _AA)

    # ── 7. Classification history (right column) ───────────────────────────────
    if history:
        x_h = max(int(w * _HIST_X_RATIO), 80)
        y_h = m + lh_t + lh_h + 2
        cv2.putText(img, "Recent:", (x_h, y_h), _FONT, fs_hist * 0.85, _C_GRAY, 1, _AA)
        for ts, lbl in history[:5]:
            y_h += lh_h + 4
            if y_h > bar_h - 2:
                break
            cv2.putText(img, f"{ts}  {lbl}", (x_h, y_h), _FONT, fs_hist, _cat_color(lbl), 1, _AA)


# ── NN bounding-box drawing ──────────────────────────────────────────────────

# MobileNetSSD was trained on Pascal VOC (21 classes including background)
_VOC_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]

_C_GREEN_BOX = (0, 255, 0)  # BGR
_C_BLACK = (0, 0, 0)


def draw_nn_detections(
    img: np.ndarray,
    detections: list,
) -> None:
    """Draw green bounding boxes with labels for each NN detection onto *img* in-place."""
    if not detections:
        return
    h, w = img.shape[:2]
    base = max(w / 1600.0, 0.35)
    fs = base * 0.50
    thickness = max(int(base * 2.0), 1)

    for det in detections:
        x1 = int(det.xmin * w)
        y1 = int(det.ymin * h)
        x2 = int(det.xmax * w)
        y2 = int(det.ymax * h)

        label = _VOC_LABELS[det.label_idx] if det.label_idx < len(_VOC_LABELS) else f"id:{det.label_idx}"
        text = f"{label} {det.confidence:.0%}"

        # Green bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), _C_GREEN_BOX, thickness, _AA)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(text, _FONT, fs, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), _C_GREEN_BOX, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4), _FONT, fs, _C_BLACK, 1, _AA)
