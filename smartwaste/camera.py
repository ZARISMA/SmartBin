import depthai as dai


def make_pipeline(device: dai.Device) -> tuple:
    """Create and start a camera pipeline for one OAK device."""
    pipeline = dai.Pipeline(device)
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    queue = cam.requestFullResolutionOutput(type=dai.ImgFrame.Type.BGR888p).createOutputQueue(
        maxSize=4, blocking=False
    )
    pipeline.start()
    return pipeline, queue
