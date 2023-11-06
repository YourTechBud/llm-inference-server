from typing import Iterator
from llama_cpp import Completion

from fastapi import APIRouter, HTTPException
from inference.llama_cpp import LlamaCppModel

from .models_config import *
from .models_openai import *


class Server:
    # TODO: Make this a dictionary of models
    model: Optional[LlamaCppModel]

    def __init__(self) -> None:
        self.model = None
        self.router = APIRouter()

    def initialise_routes(self) -> APIRouter:
        self.router.add_api_route(
            "/config/v1/load-model", self.load_model, methods=["POST"], operation_id="load-model"
        )
        self.router.add_api_route(
            "/config/v1/unload-model", self.unload_model, methods=["POST"], operation_id="unload-model"
        )
        self.router.add_api_route(
            "/api/v1/chat/completions", self.chat_completion, methods=["POST"], operation_id="chat-completion"
        )

        return self.router

    def hello(self) -> str:
        return "hello world"

    def load_model(self, req: LoadModelRequest) -> StandardResponse:
        self.model = LlamaCppModel(req.path, req.options)

        return StandardResponse(message="Model loaded successfuilly")

    def unload_model(self) -> StandardResponse:
        del self.model
        return StandardResponse(message="Model unloaded successfuilly")
    
    def chat_completion(self, req: CreateChatCompletionRequest) -> CreateChatCompletionResponse:
        # Check if the model is loaded
        if self.model is None:
            raise HTTPException(status_code=400, detail=StandardResponse(message="Load model before attempting inference").to_json())
        
        # Extract the system and user messages
        system_content = "".join([message.content or "" for message in req.messages if message.role == "system"])
        user_content = "".join([message.content or "" for message in req.messages if message.role == "user"])

        # Populate the options
        opts = InferenceOptions(system_message=system_content, user_prompt=user_content)
        if req.max_tokens is not None:
            opts.max_tokens = req.max_tokens
        if req.temperature is not None:
            opts.temperature = req.temperature
        if req.top_p is not None:
            opts.top_p = req.top_p

        # Run inference
        output = self.model.infer(opts)
        if isinstance(output, Iterator):
            raise HTTPException(status_code=500, detail=StandardResponse(message="Can't work with streams yet.").to_json())
        
        print("==================")
        print(f"System: {system_content}")
        print(f"User: {user_content}")
        print(f"Options: {opts}")
        print(f"Output: {output}")
        print("==================")
        usage: CompletionUsage = CompletionUsage(
            completion_tokens=output["usage"]["completion_tokens"],
            prompt_tokens=output["usage"]["prompt_tokens"],
            total_tokens=output["usage"]["total_tokens"],
        )
        
        convert_choice = lambda choice: CreateChatCompletionChoice(
            index=choice["index"],
            finish_reason=choice["finish_reason"],
            message=ChatCompletionResponseMessage(
                content=choice["text"],
                function_call=None,
                role="system"
            )
        )
        res = CreateChatCompletionResponse(
            id = output["id"],
            created=output["created"],
            usage=usage,
            model=req.model,
            object="chat.completion",
            choices=list(map(convert_choice, output["choices"]))
        )

        return res 

        
    

