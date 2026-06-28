from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SchemaProgress(BaseModel):
    model_config = ConfigDict(extra="allow")

    total: int | None = Field(default=None, ge=0)
    desc: str | None = Field(default="Processing", max_length=30)
    unit: str = "it"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    status: str = "idle"
    metadata: dict[str, Any] = Field(default_factory=dict)
