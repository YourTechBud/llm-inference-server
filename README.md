# LLM Inference Server

Simply python server to host Large Language Models on a restful API.

## Features:
- Configure a prompt template
- OpenAI compatible rest api
- Config APIs provided to load and unload models

## Setup

```bash
make setup
conda activate llm-inference-server
```

## Start server

```bash
make start
```

## OpenAPI spec

- Open [http://localhost:8000/docs](http://localhost:8000/docs) to open swagger ui
- Open [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) to access the raw openapi spec