from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SchemaDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    train: float = Field(default=0.7, ge=0.0, le=1.0)
    test: float = Field(default=0.2, ge=0.0, le=1.0)
    val: float = Field(default=0.1, ge=0.0, le=1.0)
