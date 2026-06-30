import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "best (2).pt")).resolve()
DEVICE = os.getenv("DEVICE", "auto").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/fruit_yolo11_outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
