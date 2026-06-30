import base64
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException
from ultralytics.utils.plotting import Annotator, colors

from backend.app.core.device import get_yolo_device
from backend.app.services.models import get_detector


def read_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Khong doc duoc file anh.")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 88) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Khong ma hoa duoc anh ket qua.")
    return base64.b64encode(buffer).decode("utf-8")


def clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    if not np.isfinite(box[:4]).all():
        return 0, 0, 0, 0
    x1 = max(0, min(width - 1, int(round(float(box[0])))))
    y1 = max(0, min(height - 1, int(round(float(box[1])))))
    x2 = max(0, min(width, int(round(float(box[2])))))
    y2 = max(0, min(height, int(round(float(box[3])))))
    return x1, y1, x2, y2


def draw_detection(image: np.ndarray, box: np.ndarray, label: str, confidence: float) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    text = f"{label} {confidence:.2f}"
    class_key = abs(hash(label)) % 1000
    annotator = Annotator(image, line_width=None, pil=False)
    annotator.box_label((x1, y1, x2, y2), text, color=colors(class_key, True))


def detection_payload(
    class_id: int,
    label: str,
    confidence: float,
    box: tuple[int, int, int, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    payload: dict[str, Any] = {
        "class_id": int(class_id),
        "label": label,
        "confidence": round(confidence, 4),
        "box": {
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
        },
    }
    if extra:
        payload.update(extra)
    return payload


def draw_detections(image: np.ndarray, detections: list[dict[str, Any]]) -> None:
    for detection in detections:
        box = detection["box"]
        draw_detection(
            image,
            np.array([box["x1"], box["y1"], box["x2"], box["y2"]]),
            str(detection["label"]),
            float(detection["confidence"]),
        )


def detect_image(
    image: np.ndarray,
    confidence: float,
    image_size: int,
    max_det: int,
    iou: float = 0.45,
    draw: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model = get_detector()
    yolo_device = get_yolo_device()
    results = model.predict(
        image,
        conf=confidence,
        imgsz=image_size,
        device=yolo_device,
        half=yolo_device != "cpu",
        max_det=max_det,
        iou=iou,
        verbose=False,
    )
    result = results[0]
    annotated = image.copy()
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return annotated, detections

    height, width = image.shape[:2]
    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for box, confidence_score, class_id in zip(xyxy, confidences, classes):
        x1, y1, x2, y2 = clip_box(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(names.get(int(class_id), class_id))
        conf_val = float(confidence_score)
        if draw:
            draw_detection(annotated, box, label, conf_val)
        detections.append(detection_payload(int(class_id), label, conf_val, (x1, y1, x2, y2)))
    
    return annotated, detections
