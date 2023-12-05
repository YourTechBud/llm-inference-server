import json
import yaml

from typing import Callable, Iterator
from llama_cpp import Completion, CompletionChoice

from fastapi import APIRouter, HTTPException
from inference.llama_cpp import LlamaCppModel
from inference.prompts import PromptMessage

from models import *


class Server:
    # TODO: Make this a dictionary of models
    model: Optional[LlamaCppModel]

    def __init__(self) -> None:
        self.model = None
        self.router = APIRouter()

    def initialise_routes(self) -> APIRouter:
        self.router.add_api_route(
            "/config/v1/load-model",
            self.load_model,
            methods=["POST"],
            operation_id="load-model",
        )
        self.router.add_api_route(
            "/config/v1/unload-model",
            self.unload_model,
            methods=["POST"],
            operation_id="unload-model",
        )
        self.router.add_api_route(
            "/api/v1/chat/completions",
            self.chat_completion,
            methods=["POST"],
            operation_id="chat-completion",
            response_model_exclude_none=True,
        )

        return self.router

    def load_model(self, req: LoadModelRequest) -> StandardResponse:
        self.model = LlamaCppModel(req.path, req.options)

        return StandardResponse(message="Model loaded successfuilly")

    def unload_model(self) -> StandardResponse:
        del self.model
        return StandardResponse(message="Model unloaded successfuilly")

    def chat_completion(
        self, req: CreateChatCompletionRequest
    ) -> CreateChatCompletionResponse:
        # Check if the model is loaded
        if self.model is None:
            raise HTTPException(
                status_code=400,
                detail=StandardResponse(
                    message="Load model before attempting inference"
                ).to_json(),
            )

        # print("================")
        # print("Request:")
        # print(yaml.dump(req))
        # print("================")

        # Add functions as a system prompt
        if req.functions is not None:
            system = """You may use the following FUNCTIONS in the response. Only use one function at a time. Give output in following OUTPUT_FORMAT in strict JSON if you want to call a function.
FUNCTIONS:"""
            for f in req.functions:
                system += f"1. Name: {f.name}\n"
                system += f"{f.description}\n"
                system += f"Parameters:\n"
                system += json.dumps(f.parameters) + "\n"
                system += "\n\n"

            system += """OUTPUT_FORMAT:
{
    "type": "FUNC_CALL",
    "name": "<name of function>",
    "parameters": "<parameters to pass to function>"
}
"""
            req.messages.append(
                ChatCompletionRequestMessage(content=system, role="system")
            )

        # Extract the system and user messages
        convert_message: Callable[
            [ChatCompletionRequestMessage], PromptMessage
        ] = lambda msg: PromptMessage(
            role=msg.role,
            content=msg.content,
        )

        # Populate the options
        opts = InferenceOptions(messages=list(map(convert_message, req.messages)))
        if req.max_tokens is not None:
            opts.max_tokens = req.max_tokens
        if req.temperature is not None:
            opts.temperature = req.temperature
        if req.top_p is not None:
            opts.top_p = req.top_p

        # Run inference
        while True:
            output = self.model.infer(opts)
            if isinstance(output, Iterator):
                raise HTTPException(
                    status_code=500,
                    detail=StandardResponse(
                        message="Can't work with streams yet."
                    ).to_json(),
                )

            # Rerun inference if output was smaller than expected
            if len(output["choices"][0]["text"].strip()) <= 1:
                continue

            # Perform some checks on output
            caught_exception = False
            for choice in output["choices"]:
                # First clean the output
                choice["text"] = choice["text"].lstrip()

                # We will need to clean the output a bit if it was a function call request
                try:
                    if "FUNC_CALL" in choice["text"]:
                        # First try to sanitize the text
                        choice["text"] = sanitize_json_text(choice["text"])

                        # Attempt to serialize it
                        f = dict(json.loads(choice["text"]))
                except json.JSONDecodeError:
                    # We want to retry inference if the response was not JSON serializable
                    caught_exception = True

            # Retry inference if we caught an exception
            if caught_exception:
                continue

            # Print for debugging purposes
            print("================")
            print("Output:")
            for choice in output["choices"]:
                choice["text"] = choice["text"].lstrip()
                print("---------")
                print(f'{choice["index"]}: {choice["text"]}')
            print("================")
            break

        usage: CompletionUsage = CompletionUsage(
            completion_tokens=output["usage"]["completion_tokens"],
            prompt_tokens=output["usage"]["prompt_tokens"],
            total_tokens=output["usage"]["total_tokens"],
        )

        res = CreateChatCompletionResponse(
            id=output["id"],
            created=output["created"],
            usage=usage,
            model=req.model,
            object="chat.completion",
            choices=list(map(prepare_chat_completion_message, output["choices"])),
        )

        return res

def sanitize_json_text(text):
  # find the index of the first "{"
  start = text.find("{")
  # find the index of the last "}"
  end = text.rfind("}")
  # return the substring between the start and end indices
  return text[start:end+1]

def prepare_chat_completion_message(
    choice: CompletionChoice,
) -> CreateChatCompletionChoice:
    fn_call: Optional[ChatCompletionMessageFunctionCall] = None
    # Check if output suggests a function call
    if "FUNC_CALL" in choice["text"]:
        f = dict(json.loads(choice["text"]))
        fn_call = ChatCompletionMessageFunctionCall(
            name=f.get("name", ""), arguments=json.dumps(f.get("parameters"))
        )
    return CreateChatCompletionChoice(
        index=choice["index"],
        finish_reason=choice["finish_reason"]
        if choice["finish_reason"] is not None
        else "stop",
        message=ChatCompletionResponseMessage(
            content=choice["text"], role="system", function_call=fn_call
        ),
    )
