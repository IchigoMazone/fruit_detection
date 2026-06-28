import base64
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image
import torch
from ultralytics.utils.plotting import Annotator, colors

from backend.app.core.device import get_torch_device, get_yolo_device
from backend.app.services.models import get_classifier, get_detector, normalize_model_option


FAMILY_ALIASES = {
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "tomato": "tomato",
}


def fruit_family(label: str) -> str:
    return FAMILY_ALIASES.get(label.split("_", 1)[0].strip().lower(), label.split("_", 1)[0].strip().lower())


def ripeness_label(label: str) -> str:
    parts = label.strip().lower().split("_", 1)
    return parts[1] if len(parts) == 2 else label


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


def classify_crops(
    crops: list[np.ndarray],
    model_option: str,
    allowed_families: list[str] | None = None,
) -> list[tuple[int, str, float]]:
    classifier_bundle = get_classifier(model_option)
    if classifier_bundle is None or not crops:
        return []

    classifier, classes, transform = classifier_bundle
    tensors = []
    for crop in crops:
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(transform(Image.fromarray(rgb_crop)))

    device = get_torch_device()
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(classifier(batch), dim=1)

    predictions: list[tuple[int, str, float]] = []
    for crop_index, row in enumerate(probabilities):
        allowed_indexes: list[int] = []
        if allowed_families is not None:
            family = fruit_family(allowed_families[crop_index])
            allowed_indexes = [index for index, label in enumerate(classes) if fruit_family(label) == family]

        if allowed_indexes:
            family_scores = row[allowed_indexes]
            score, family_index = torch.max(family_scores, dim=0)
            index = torch.tensor(allowed_indexes[int(family_index.item())], device=row.device)
        elif allowed_families is not None:
            family = fruit_family(allowed_families[crop_index])
            predictions.append((-1, family, 1.0))
            continue
        else:
            score, index = torch.max(row, dim=0)
        class_id = int(index.item())
        label = classes[class_id] if class_id < len(classes) else str(class_id)
        predictions.append((class_id, label, float(score.item())))
    return predictions


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
    model_option: str | None = None,
    classifier_confidence: float = 0.25,
    iou: float = 0.45,
    draw: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    option = normalize_model_option(model_option)
    model = get_detector(option)
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

    if get_classifier(option) is not None:
        boxes: list[tuple[int, int, int, int]] = []
        crops: list[np.ndarray] = []
        detector_meta: list[tuple[str, float]] = []
        detector_families: list[str] = []
        for box, detector_confidence, class_id in zip(xyxy, confidences, classes):
            clipped_box = clip_box(box, width, height)
            x1, y1, x2, y2 = clipped_box
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(clipped_box)
            crops.append(image[y1:y2, x1:x2])
            detector_label = str(names.get(int(class_id), class_id))
            detector_meta.append((detector_label, float(detector_confidence)))
            detector_families.append(fruit_family(detector_label))

        for box, prediction, meta in zip(boxes, classify_crops(crops, option, detector_families), detector_meta):
            classifier_class_id, label, classifier_score = prediction
            detector_label, detector_score = meta
            detector_family = fruit_family(detector_label)
            classifier_family = fruit_family(label)
            if classifier_class_id >= 0 and classifier_family != detector_family:
                full_classifier_label = detector_family
                classifier_score = detector_score
            else:
                full_classifier_label = label
            ripeness = ripeness_label(full_classifier_label)
            label = f"{detector_family} {ripeness}" if ripeness != detector_family else detector_family
            if draw:
                draw_detection(annotated, box, label, classifier_score)
            detections.append(
                detection_payload(
                    classifier_class_id,
                    label,
                    classifier_score,
                    box,
                    {
                        "classifier_label": full_classifier_label,
                        "detector_label": detector_label,
                        "detector_confidence": round(detector_score, 4),
                    },
                )
            )
        return annotated, detections

    for box, confidence_score, class_id in zip(xyxy, confidences, classes):
        x1, y1, x2, y2 = clip_box(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(names.get(int(class_id), class_id))
        confidence = float(confidence_score)
        if draw:
            draw_detection(annotated, box, label, confidence)
        detections.append(detection_payload(int(class_id), label, confidence, (x1, y1, x2, y2)))
    return annotated, detections
