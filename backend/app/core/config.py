import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "best (2).pt")).resolve()
BEST2_MODEL_PATH = Path(os.getenv("BEST2_MODEL_PATH", os.getenv("MODEL_PATH", ROOT_DIR / "best (2).pt"))).resolve()
RESNET_YOLO_MODEL_PATH = Path(os.getenv("RESNET_YOLO_MODEL_PATH", ROOT_DIR / "best.pt")).resolve()
RESNET_CLASSIFIER_PATH = Path(os.getenv("RESNET_CLASSIFIER_PATH", ROOT_DIR / "fruit_classifier.pth")).resolve()
DEFAULT_MODEL_OPTION = os.getenv("MODEL_OPTION", "best2").strip().lower()
MODEL_OPTIONS = {
    "best2": {
        "label": "YOLO",
        "yolo_path": BEST2_MODEL_PATH,
        "classifier_path": None,
    },
    "resnet": {
        "label": "YOLO + ResNet",
        "yolo_path": RESNET_YOLO_MODEL_PATH,
        "classifier_path": RESNET_CLASSIFIER_PATH,
    },
}
MODEL_OPTION_ALIASES = {
    "best": "best2",
    "best2": "best2",
    "best_2": "best2",
    "best-2": "best2",
    "best (2).pt": "best2",
    "yolo": "best2",
    "best.pt": "resnet",
    "resnet": "resnet",
    "yolo+resnet": "resnet",
    "yolo + resnet": "resnet",
    "best_resnet": "resnet",
    "best-resnet": "resnet",
    "best.pt+resnet": "resnet",
    "best.pt + resnet": "resnet",
}
DEVICE = os.getenv("DEVICE", "auto").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/fruit_yolo11_outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
