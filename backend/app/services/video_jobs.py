import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import cv2
from fastapi import HTTPException

from backend.app.core.config import OUTPUT_DIR
from backend.app.services.image_detection import detect_image, draw_detections


_jobs_lock = Lock()
_video_jobs: dict[str, dict[str, Any]] = {}
MIN_OUTPUT_FPS = 5.0
MAX_OUTPUT_FPS = 60.0
DEFAULT_OUTPUT_FPS = 25.0
BOX_SMOOTHING_ALPHA = 0.45
LABEL_SWITCH_MARGIN = 0.12
TRACK_IOU_THRESHOLD = 0.25


def remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def update_video_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        if job_id in _video_jobs:
            _video_jobs[job_id].update(updates)


def public_video_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _video_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Khong tim thay video job.")
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "processed_frames": job["processed_frames"],
            "total_frames": job["total_frames"],
            "error": job.get("error"),
            "result_url": f"/detect/video/jobs/{job_id}/result" if job["status"] == "done" else None,
        }


def get_video_output_path(job_id: str) -> Path:
    with _jobs_lock:
        job = _video_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Khong tim thay video job.")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail="Video chua detect xong.")
        return Path(job["output_path"])


def open_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV khong mo duoc WebM VP8 writer.")
    return writer


def sanitize_fps(raw_fps: float) -> float:
    if raw_fps < MIN_OUTPUT_FPS or raw_fps > MAX_OUTPUT_FPS:
        return DEFAULT_OUTPUT_FPS
    return raw_fps


def box_iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    x1 = max(float(box_a["x1"]), float(box_b["x1"]))
    y1 = max(float(box_a["y1"]), float(box_b["y1"]))
    x2 = min(float(box_a["x2"]), float(box_b["x2"]))
    y2 = min(float(box_a["y2"]), float(box_b["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, float(box_a["x2"]) - float(box_a["x1"])) * max(0.0, float(box_a["y2"]) - float(box_a["y1"]))
    area_b = max(0.0, float(box_b["x2"]) - float(box_b["x1"])) * max(0.0, float(box_b["y2"]) - float(box_b["y1"]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def smooth_box(previous: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    return {
        key: round((float(previous[key]) * (1 - BOX_SMOOTHING_ALPHA)) + (float(current[key]) * BOX_SMOOTHING_ALPHA), 2)
        for key in ("x1", "y1", "x2", "y2")
    }


def stabilize_detections(
    detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stabilized: list[dict[str, Any]] = []
    next_tracks: list[dict[str, Any]] = []
    used_track_indexes: set[int] = set()

    for detection in detections:
        best_index = None
        best_iou = 0.0
        for index, track in enumerate(tracks):
            if index in used_track_indexes:
                continue
            iou = box_iou(detection["box"], track["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = index

        stable_detection = dict(detection)
        stable_detection["box"] = dict(detection["box"])

        if best_index is not None and best_iou >= TRACK_IOU_THRESHOLD:
            track = tracks[best_index]
            used_track_indexes.add(best_index)
            stable_detection["box"] = smooth_box(track["box"], detection["box"])

            new_confidence = float(detection["confidence"])
            old_confidence = float(track["confidence"])
            if detection["label"] != track["label"] and new_confidence < old_confidence + LABEL_SWITCH_MARGIN:
                stable_detection["label"] = track["label"]
                stable_detection["confidence"] = old_confidence

        next_tracks.append(
            {
                "box": stable_detection["box"],
                "label": stable_detection["label"],
                "confidence": stable_detection["confidence"],
            }
        )
        stabilized.append(stable_detection)

    return stabilized, next_tracks


def process_video_job(
    job_id: str,
    input_path: Path,
    output_path: Path,
    confidence: float,
    image_size: int,
    max_det: int,
    iou: float,
) -> None:
    capture = cv2.VideoCapture(str(input_path))
    writer: cv2.VideoWriter | None = None

    try:
        if not capture.isOpened():
            raise RuntimeError("Khong doc duoc file video.")

        fps = sanitize_fps(float(capture.get(cv2.CAP_PROP_FPS) or DEFAULT_OUTPUT_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("Video khong co kich thuoc hop le.")

        update_video_job(job_id, status="processing", total_frames=total_frames, progress=0)
        writer = open_video_writer(output_path, fps, width, height)
        processed_frames = 0
        tracks: list[dict[str, Any]] = []

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, detections = detect_image(
                frame,
                confidence,
                image_size,
                max_det,
                iou,
                draw=False,
            )
            detections, tracks = stabilize_detections(detections, tracks)
            draw_detections(annotated, detections)
            writer.write(annotated)
            processed_frames += 1

            progress = min(99, round((processed_frames / total_frames) * 100)) if total_frames > 0 else 0
            update_video_job(job_id, processed_frames=processed_frames, progress=progress)

        if writer is not None:
            writer.release()
            writer = None

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Khong tao duoc video ket qua.")

        update_video_job(job_id, status="done", progress=100, processed_frames=processed_frames)
    except Exception as exc:
        remove_file(output_path)
        update_video_job(job_id, status="error", error=str(exc), progress=0)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        remove_file(input_path)


def create_video_job(
    file_name: str | None,
    file_bytes: bytes,
    confidence: float,
    image_size: int,
    max_det: int,
    iou: float,
) -> dict[str, Any]:
    suffix = Path(file_name or "video.mp4").suffix or ".mp4"
    job_id = uuid.uuid4().hex
    input_path = OUTPUT_DIR / f"{job_id}{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_detected.webm"
    input_path.write_bytes(file_bytes)

    with _jobs_lock:
        _video_jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "processed_frames": 0,
            "total_frames": 0,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "error": None,
        }

    Thread(
        target=process_video_job,
        args=(job_id, input_path, output_path, confidence, image_size, max_det, iou),
        daemon=True,
    ).start()
    return public_video_job(job_id)
