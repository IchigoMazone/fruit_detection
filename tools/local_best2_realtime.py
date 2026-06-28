import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


MODEL_PATH = Path("best (2).pt")
CAMERA_INDEX = 0
CONF = 0.25
IMGSZ = 320
MAX_DET = 10000
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
MIRROR = False


def resolve_device() -> str | int:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return 0
    return "cpu"


def draw_status(frame, fps: float, infer_ms: float, device: str | int, detections: int) -> None:
    lines = [
        f"best (2).pt | FPS {fps:5.1f} | infer {infer_ms:5.1f} ms | device {device}",
        f"detections {detections} | q/esc quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def run_realtime() -> None:
    model = YOLO(str(MODEL_PATH))
    device = resolve_device()
    capture = cv2.VideoCapture(CAMERA_INDEX)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera: {CAMERA_INDEX}")

    print(f"device={device}")
    print(f"model={MODEL_PATH}")
    print(f"source=camera:{CAMERA_INDEX}")
    print("press q or esc to quit")

    frame_count = 0
    total_infer = 0.0
    fps_ema = 0.0
    last_frame_time = time.perf_counter()

    try:
        ok, warm_frame = capture.read()
        if ok:
            model.predict(
                warm_frame,
                conf=CONF,
                imgsz=IMGSZ,
                max_det=MAX_DET,
                device=device,
                half=torch.cuda.is_available(),
                verbose=False,
            )

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if MIRROR:
                frame = cv2.flip(frame, 1)

            start = time.perf_counter()
            result = model.predict(
                frame,
                conf=CONF,
                imgsz=IMGSZ,
                max_det=MAX_DET,
                device=device,
                half=torch.cuda.is_available(),
                verbose=False,
            )[0]
            infer_elapsed = time.perf_counter() - start
            total_infer += infer_elapsed
            frame_count += 1

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
            fps_ema = instant_fps if fps_ema == 0 else (fps_ema * 0.85 + instant_fps * 0.15)
            last_frame_time = now

            annotated = result.plot()
            detections = len(result.boxes) if result.boxes is not None else 0
            draw_status(annotated, fps_ema, infer_elapsed * 1000, device, detections)

            cv2.imshow("YOLO realtime - best (2).pt", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    avg_ms = (total_infer / frame_count * 1000) if frame_count else 0
    eff_fps = (frame_count / total_infer) if total_infer > 0 else 0
    print(f"frames={frame_count}")
    print(f"avg_infer={avg_ms:.2f} ms/frame")
    print(f"effective_fps={eff_fps:.2f}")


def main() -> None:
    run_realtime()


if __name__ == "__main__":
    main()
