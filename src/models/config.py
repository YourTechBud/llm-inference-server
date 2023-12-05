from typing import Optional
from pydantic import BaseModel

from inference.types import ModelLoadingOptions, InferenceOptions


class LoadModelRequest(BaseModel):
    path: str
    options: ModelLoadingOptions = ModelLoadingOptions()


