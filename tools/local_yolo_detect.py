import argparse
import time
from pathlib import Path
from typing import Union

import cv2
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
VideoSource = Union[int, Path]
CLASSIFIER_MEAN = (0.485, 0.456, 0.406)
CLASSIFIER_STD = (0.229, 0.224, 0.225)
BOX_COLOR = (34, 197, 94)
TEXT_BG_COLOR = (20, 83, 45)
TEXT_COLOR = (255, 255, 255)


def resolve_device() -> str | int:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return 0
    return "cpu"


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda:0")
    return torch.device("cpu")


def parse_video_source(source: str) -> VideoSource:
    if source.isdigit():
        return int(source)
    return Path(source)


def source_label(source: VideoSource) -> str:
    return f"camera:{source}" if isinstance(source, int) else str(source)


def draw_status(frame, fps: float, infer_ms: float, device: str | int, show_window: bool) -> None:
    lines = [
        f"FPS {fps:5.1f} | infer {infer_ms:5.1f} ms | device {device}",
        "q/esc quit" if show_window else "headless realtime",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def load_classifier(checkpoint_path: Path, device: torch.device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
    return classifier, classes, transform


def classify_crops(classifier, classes: list[str], transform, crops: list, device: torch.device) -> list[tuple[str, float]]:
    if not crops:
        return []

    tensors = []
    for crop in crops:
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_crop)
        tensors.append(transform(image))

    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(classifier(batch), dim=1)

    predictions = []
    for row in probabilities:
        score, index = torch.max(row, dim=0)
        predictions.append((classes[int(index.item())], float(score.item())))
    return predictions


def draw_detection(frame, box: tuple[int, int, int, int], label: str, score: float) -> None:
    x1, y1, x2, y2 = box
    text = f"{label} {score:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    label_y1 = max(0, y1 - text_height - baseline - 8)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_width + 10, y1), TEXT_BG_COLOR, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - baseline - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT_COLOR, 2, cv2.LINE_AA)


def annotate_with_classifier(
    frame,
    result,
    classifier,
    classes: list[str],
    transform,
    classifier_device: torch.device,
    classifier_conf: float,
) -> tuple[int, int]:
    if result.boxes is None or len(result.boxes) == 0:
        return 0, 0

    height, width = frame.shape[:2]
    boxes = []
    crops = []
    for raw_box in result.boxes:
        x1, y1, x2, y2 = raw_box.xyxy[0].detach().cpu().tolist()
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width, int(round(x2))))
        y2 = max(0, min(height, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        crops.append(frame[y1:y2, x1:x2])

    predictions = classify_crops(classifier, classes, transform, crops, classifier_device)
    drawn = 0
    for box, (label, score) in zip(boxes, predictions):
        if score < classifier_conf:
            continue
        draw_detection(frame, box, label, score)
        drawn += 1
    return len(boxes), drawn


def run_image(model: YOLO, source: Path, output: Path, conf: float, imgsz: int, max_det: int) -> None:
    device = resolve_device()
    model.predict(
        str(source),
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        device=device,
        half=torch.cuda.is_available(),
        verbose=False,
    )

    start = time.perf_counter()
    result = model.predict(
        str(source),
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        device=device,
        half=torch.cuda.is_available(),
        verbose=False,
    )[0]
    elapsed = time.perf_counter() - start

    annotated = result.plot()
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)

    print(f"device={device}")
    print(f"source={source}")
    print(f"output={output}")
    print(f"detections={len(result.boxes) if result.boxes is not None else 0}")
    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = names.get(class_id, str(class_id))
            score = float(box.conf.item())
            print(f"- {label}: {score:.4f}")
    print(f"time={elapsed * 1000:.2f} ms")


def run_video(model: YOLO, source: Path, output: Path, conf: float, imgsz: int, max_det: int) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output}")

    frame_count = 0
    total_infer = 0.0
    device = resolve_device()
    warm_frame = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if warm_frame is None:
                warm_frame = frame
                model.predict(
                    warm_frame,
                    conf=conf,
                    imgsz=imgsz,
                    max_det=max_det,
                    device=device,
                    half=torch.cuda.is_available(),
                    verbose=False,
                )

            start = time.perf_counter()
            result = model.predict(
                frame,
                conf=conf,
                imgsz=imgsz,
                max_det=max_det,
                device=device,
                half=torch.cuda.is_available(),
                verbose=False,
            )[0]
            total_infer += time.perf_counter() - start
            writer.write(result.plot())
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    avg_ms = (total_infer / frame_count * 1000) if frame_count else 0
    fps_eff = (frame_count / total_infer) if total_infer > 0 else 0
    print(f"device={device}")
    print(f"source={source}")
    print(f"output={output}")
    print(f"frames={frame_count}")
    print(f"avg_infer={avg_ms:.2f} ms/frame")
    print(f"effective_fps={fps_eff:.2f}")


def run_realtime(
    model: YOLO,
    source: VideoSource,
    conf: float,
    imgsz: int,
    max_det: int,
    classifier,
    classifier_classes: list[str],
    classifier_transform,
    classifier_device: torch.device,
    classifier_conf: float,
    yolo_only: bool,
    camera_width: int,
    camera_height: int,
    mirror: bool,
    no_window: bool,
    max_frames: int,
) -> None:
    capture_arg = source if isinstance(source, int) else str(source)
    capture = cv2.VideoCapture(capture_arg)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open realtime source: {source_label(source)}")

    device = resolve_device()
    frame_count = 0
    total_infer = 0.0
    fps_ema = 0.0
    last_frame_time = time.perf_counter()
    show_window = not no_window

    print(f"device={device}")
    print(f"source={source_label(source)}")
    print("press q or esc to quit")

    try:
        ok, warm_frame = capture.read()
        if ok:
            model.predict(
                warm_frame,
                conf=conf,
                imgsz=imgsz,
                max_det=max_det,
                device=device,
                half=torch.cuda.is_available(),
                verbose=False,
            )
            if not yolo_only:
                warm_result = model.predict(
                    warm_frame,
                    conf=conf,
                    imgsz=imgsz,
                    max_det=max_det,
                    device=device,
                    half=torch.cuda.is_available(),
                    verbose=False,
                )[0]
                annotate_with_classifier(
                    warm_frame,
                    warm_result,
                    classifier,
                    classifier_classes,
                    classifier_transform,
                    classifier_device,
                    classifier_conf,
                )
            if not isinstance(source, int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if mirror and isinstance(source, int):
                frame = cv2.flip(frame, 1)

            start = time.perf_counter()
            result = model.predict(
                frame,
                conf=conf,
                imgsz=imgsz,
                max_det=max_det,
                device=device,
                half=torch.cuda.is_available(),
                verbose=False,
            )[0]
            if yolo_only:
                annotated = result.plot()
                accepted = len(result.boxes) if result.boxes is not None else 0
            else:
                annotated = frame.copy()
                _, accepted = annotate_with_classifier(
                    annotated,
                    result,
                    classifier,
                    classifier_classes,
                    classifier_transform,
                    classifier_device,
                    classifier_conf,
                )
            infer_elapsed = time.perf_counter() - start
            total_infer += infer_elapsed
            frame_count += 1

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
            fps_ema = instant_fps if fps_ema == 0 else (fps_ema * 0.85 + instant_fps * 0.15)
            last_frame_time = now

            draw_status(annotated, fps_ema, infer_elapsed * 1000, device, show_window)
            cv2.putText(
                annotated,
                f"shown {accepted}",
                (12, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"shown {accepted}",
                (12, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if show_window:
                title = "YOLO best.pt + ResNet realtime" if not yolo_only else "YOLO realtime"
                cv2.imshow(title, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if max_frames > 0 and frame_count >= max_frames:
                break
    finally:
        capture.release()
        if show_window:
            cv2.destroyAllWindows()

    avg_ms = (total_infer / frame_count * 1000) if frame_count else 0
    eff_fps = (frame_count / total_infer) if total_infer > 0 else 0
    print(f"frames={frame_count}")
    print(f"avg_infer={avg_ms:.2f} ms/frame")
    print(f"effective_fps={eff_fps:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO locally without FastAPI.")
    parser.add_argument("source", nargs="?", default="0", help="Image/video path, or camera index like 0.")
    parser.add_argument("--model", type=Path, default=Path("best.pt"))
    parser.add_argument("--classifier", type=Path, default=Path("fruit_classifier.pth"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/local_yolo_result.jpg"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--max-det", type=int, default=10000)
    parser.add_argument("--classifier-conf", type=float, default=0.70)
    parser.add_argument("--yolo-only", action="store_true", help="Skip ResNet classification and draw YOLO boxes directly.")
    parser.add_argument("--realtime", action="store_true", help="Show live detections from webcam or video.")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=360)
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam preview.")
    parser.add_argument("--no-window", action="store_true", help="Process realtime frames without opening a GUI window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop realtime mode after N frames. 0 means unlimited.")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    classifier = None
    classifier_classes: list[str] = []
    classifier_transform = None
    classifier_device = resolve_torch_device()
    if not args.yolo_only:
        classifier, classifier_classes, classifier_transform = load_classifier(args.classifier, classifier_device)

    realtime_source = parse_video_source(args.source)
    if args.realtime or isinstance(realtime_source, int):
        run_realtime(
            model,
            realtime_source,
            args.conf,
            args.imgsz,
            args.max_det,
            classifier,
            classifier_classes,
            classifier_transform,
            classifier_device,
            args.classifier_conf,
            args.yolo_only,
            args.camera_width,
            args.camera_height,
            args.mirror,
            args.no_window,
            args.max_frames,
        )
        return

    source_path = Path(args.source)
    suffix = source_path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        run_video(model, source_path, args.output, args.conf, args.imgsz, args.max_det)
    else:
        run_image(model, source_path, args.output, args.conf, args.imgsz, args.max_det)


if __name__ == "__main__":
    main()
