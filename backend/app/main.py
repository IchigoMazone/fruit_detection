import os
import traceback
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import router
from backend.app.core.config import DEFAULT_CORS_ORIGINS
from backend.app.services.models import warm_detector


app = FastAPI(title="Fruit YOLO11 Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or exc.__class__.__name__},
    )


@app.on_event("startup")
def warmup_models() -> None:
    warm_detector()
