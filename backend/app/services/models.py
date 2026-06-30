from threading import Lock

import numpy as np
from ultralytics import YOLO

from backend.app.core.config import MODEL_PATH
from backend.app.core.device import get_yolo_device

_detector: YOLO | None = None
_detector_lock = Lock()


def get_detector() -> YOLO:
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                if not MODEL_PATH.exists():
                    raise RuntimeError(f"Model not found: {MODEL_PATH}")
                _detector = YOLO(str(MODEL_PATH))
    return _detector


def warm_detector(image_size: int = 320, max_det: int = 10000) -> None:
    yolo_device = get_yolo_device()
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    get_detector().predict(
        image,
        imgsz=image_size,
        device=yolo_device,
        half=yolo_device != "cpu",
        max_det=max_det,
        verbose=False,
    )
