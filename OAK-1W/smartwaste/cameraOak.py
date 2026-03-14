import depthai as dai
import numpy as np


def make_pipeline(device: dai.Device) -> tuple:
    """Create and start a camera pipeline for one OAK device."""
    pipeline = dai.Pipeline(device)
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    queue = (
        cam.requestFullResolutionOutput(type=dai.ImgFrame.Type.BGR888p)
           .createOutputQueue(maxSize=4, blocking=False)
    )
    pipeline.start()
    return pipeline, queue


def crop_sides(frame: np.ndarray, crop_percent: float) -> np.ndarray:
    """Remove crop_percent fraction from both left and right sides."""
    if crop_percent <= 0:
        return frame
    h, w = frame.shape[:2]
    left  = int(w * crop_percent)
    right = int(w * (1 - crop_percent))
    return frame[:, left:right]
