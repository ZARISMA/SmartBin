import numpy as np
from picamera2 import Picamera2


def make_cameras(num_cameras: int = 2) -> list:
    """Create and start Picamera2 instances for all connected Pi cameras."""
    cameras = []
    for i in range(num_cameras):
        cam = Picamera2(i)
        config = cam.create_preview_configuration(main={"format": "BGR888", "size": (1280, 720)})
        cam.configure(config)
        cam.start()
        cameras.append(cam)
    return cameras


def grab_frame(cam) -> np.ndarray:
    """Capture a single BGR frame from a Picamera2 instance."""
    return np.asarray(cam.capture_array("main"))


def stop_cameras(cameras: list) -> None:
    """Stop all Picamera2 instances cleanly."""
    for cam in cameras:
        try:
            cam.stop()
        except Exception:
            pass
