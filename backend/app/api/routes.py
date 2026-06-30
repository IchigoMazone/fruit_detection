from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from backend.app.core.config import (
    IMAGE_TYPES,
    MODEL_PATH,
    VIDEO_TYPES,
)
from backend.app.core.device import get_runtime_info
from backend.app.services.image_detection import detect_image, encode_jpeg, read_image
from backend.app.services.video_jobs import create_video_job, get_video_output_path, public_video_job


router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        **get_runtime_info(),
    }


@router.post("/detect/image")
async def detect_uploaded_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=0.99),
    image_size: int = Query(1280, alias="imgsz", ge=256, le=2048),
    max_det: int = Query(10000, ge=1, le=100000),
    iou: float = Query(0.45, ge=0.1, le=0.95),
    jpeg_quality: int = Query(88, alias="quality", ge=40, le=95),
) -> dict[str, Any]:
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Hay upload anh JPG, PNG hoac WEBP.")

    image = read_image(await file.read())
    annotated, detections = detect_image(
        image,
        confidence,
        image_size,
        max_det,
        iou,
    )

    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image": f"data:image/jpeg;base64,{encode_jpeg(annotated, jpeg_quality)}",
    }


@router.post("/detect/video")
async def detect_uploaded_video(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=0.99),
    image_size: int = Query(1280, alias="imgsz", ge=256, le=2048),
    max_det: int = Query(10000, ge=1, le=100000),
    iou: float = Query(0.45, ge=0.1, le=0.95),
) -> dict[str, Any]:
    if file.content_type not in VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Hay upload video MP4, MOV, AVI hoac WEBM.")

    return create_video_job(
        file.filename,
        await file.read(),
        confidence,
        image_size,
        max_det,
        iou,
    )


@router.get("/detect/video/jobs/{job_id}")
def get_video_job(job_id: str) -> dict[str, Any]:
    return public_video_job(job_id)


@router.get("/detect/video/jobs/{job_id}/result")
def get_video_result(job_id: str) -> FileResponse:
    return FileResponse(
        get_video_output_path(job_id),
        media_type="video/webm",
        filename="detected.webm",
    )
