import cv2
import numpy as np

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_AA    = cv2.LINE_AA
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)


def draw_overlay(img: np.ndarray, label: str, detail: str, auto_on: bool) -> None:
    """Draw semi-transparent status bar onto img in-place."""
    overlay = img.copy()
    cv2.rectangle(overlay, (15, 15), (1575, 135), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    cv2.putText(img, "Smart Waste AI (2x OAK -> 1 image)", (30, 45),  _FONT, 0.9,  _WHITE, 2, _AA)
    cv2.putText(img, f"Status: {label}",                   (30, 85),  _FONT, 0.85, _GREEN, 2, _AA)
    mode = "AUTO" if auto_on else "MANUAL"
    cv2.putText(img, f"Mode: {mode} | {detail}",           (30, 120), _FONT, 0.65, _WHITE, 2, _AA)
