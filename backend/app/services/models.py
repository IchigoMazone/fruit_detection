from threading import Lock

import numpy as np
import torch
from torchvision import models, transforms
from ultralytics import YOLO

from backend.app.core.config import (
    DEFAULT_MODEL_OPTION,
    MODEL_OPTION_ALIASES,
    MODEL_OPTIONS,
)
from backend.app.core.device import get_torch_device, get_yolo_device


CLASSIFIER_MEAN = (0.485, 0.456, 0.406)
CLASSIFIER_STD = (0.229, 0.224, 0.225)

_detectors: dict[str, YOLO] = {}
_classifiers: dict[str, tuple[torch.nn.Module, list[str], transforms.Compose]] = {}
_detector_lock = Lock()
_classifier_lock = Lock()


def normalize_model_option(model_option: str | None) -> str:
    raw_option = (model_option or DEFAULT_MODEL_OPTION).strip().lower()
    option = MODEL_OPTION_ALIASES.get(raw_option, raw_option)
    if option not in MODEL_OPTIONS:
        valid = ", ".join(MODEL_OPTIONS)
        raise ValueError(f"Model option khong hop le: {model_option}. Hay dung: {valid}.")
    return option


def get_model_option_info(model_option: str | None) -> dict:
    option = normalize_model_option(model_option)
    return MODEL_OPTIONS[option]


def get_detector(model_option: str | None = None) -> YOLO:
    option = normalize_model_option(model_option)
    if option not in _detectors:
        with _detector_lock:
            if option not in _detectors:
                model_path = MODEL_OPTIONS[option]["yolo_path"]
                if not model_path.exists():
                    raise RuntimeError(f"Model not found: {model_path}")
                _detectors[option] = YOLO(str(model_path))
    return _detectors[option]


def get_classifier(model_option: str | None = None) -> tuple[torch.nn.Module, list[str], transforms.Compose] | None:
    option = normalize_model_option(model_option)
    classifier_path = MODEL_OPTIONS[option]["classifier_path"]
    if classifier_path is None:
        return None

    if option not in _classifiers:
        with _classifier_lock:
            if option not in _classifiers:
                if not classifier_path.exists():
                    raise RuntimeError(f"Classifier not found: {classifier_path}")
                device = get_torch_device()
                try:
                    checkpoint = torch.load(classifier_path, map_location=device, weights_only=True)
                except TypeError:
                    checkpoint = torch.load(classifier_path, map_location=device)

                classes = list(checkpoint["classes"])
                image_size = int(checkpoint.get("img_size", 224))
                classifier = models.resnet18(weights=None)
                classifier.fc = torch.nn.Linear(classifier.fc.in_features, int(checkpoint["num_classes"]))
                classifier.load_state_dict(checkpoint["model_state_dict"])
                classifier.to(device)
                classifier.eval()
                transform = transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=CLASSIFIER_MEAN, std=CLASSIFIER_STD),
                    ]
                )
                _classifiers[option] = (classifier, classes, transform)
    return _classifiers[option]


def warm_detector(image_size: int = 320, model_option: str | None = None, max_det: int = 10000) -> None:
    yolo_device = get_yolo_device()
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    get_detector(model_option).predict(
        image,
        imgsz=image_size,
        device=yolo_device,
        half=yolo_device != "cpu",
        max_det=max_det,
        verbose=False,
    )
    get_classifier(model_option)
