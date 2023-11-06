from typing import Optional
from pydantic import BaseModel

from inference.types import ModelLoadingOptions, InferenceOptions


class LoadModelRequest(BaseModel):
    path: str
    options: ModelLoadingOptions = ModelLoadingOptions()


class StandardResponse(BaseModel):
    message: str
    error: Optional[str] = None

    def to_json(self):
        return self.__dict__
