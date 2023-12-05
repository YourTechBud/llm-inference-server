from typing import List, Optional

from pydantic import BaseModel

from inference.prompts import PromptMessage, PromptTemplateName


class ModelLoadingOptions(BaseModel):
    n_ctx: int = 2048
    n_batch: int = 1024
    n_gpu_layers: int = -1
    n_threads: Optional[int] = None
    prompt_tmpl: str = "default"

class InferenceOptions(BaseModel):
    messages: List[PromptMessage]
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 25
