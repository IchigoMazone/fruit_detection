from typing import Any

import torch

from backend.app.core.config import DEVICE


def resolve_torch_device() -> torch.device:
    if DEVICE in {"auto", "cuda", "gpu"}:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
        return torch.device("cpu")

    if DEVICE.isdigit():
        index = int(DEVICE)
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return torch.device(f"cuda:{index}")
        return torch.device("cpu")

    if DEVICE.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(DEVICE)
        return torch.device("cpu")

    return torch.device("cpu")


def get_torch_device() -> torch.device:
    return resolve_torch_device()


def get_yolo_device() -> str | int:
    device = get_torch_device()
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return "cpu"


def get_runtime_info() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    torch_device = get_torch_device()
    return {
        "device_setting": DEVICE,
        "torch_device": str(torch_device),
        "yolo_device": get_yolo_device(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(torch_device.index or 0)
        if torch_device.type == "cuda"
        else None,
        "torch_cuda": torch.version.cuda,
    }
