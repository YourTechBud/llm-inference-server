import uvicorn
import argparse

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

from server.server import Server

app = FastAPI(title="LLM Inference Server")


class Greeting(BaseModel):
    greeting: str

    def __init__(self, greeting: str) -> None:
        self.greeting = greeting


@app.get("/")
def read_root() -> Greeting:
    return Greeting("Hey buddy")


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


# @app.post("/load")
# def loadModel(req: LoadModelRequest) -> dict:
#     ser
#     return {}


def start():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Program to host LLMs.")

    # Add all server arguments
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="The port to start the server on."
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Set number of worker processes to run.",
    )
    parser.add_argument(
        "--reload", type=bool, default=False, help="Enable auto reload."
    )

    # Parse the arguments
    args = parser.parse_args()

    server = Server()

    router = server.initialise_routes()
    app.include_router(router)

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )