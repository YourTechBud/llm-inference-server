# LLM Inference Server

Simply python server to host Large Language Models on a restful API.

## Features:
- Configure a prompt template
- OpenAI compatible rest api
- Config APIs provided to load and unload models

## Setup

```bash
make setup
```

## Start server

```bash
make start
```

## Config APIs

### 1. Load model

```bash
curl --request POST \
  --url http://localhost:8000/config/v1/load-model \
  --header 'content-type: application/json' \
  --data '{
  "path": "./mistral-7b-openorca.Q5_K_M.gguf",
  "options": {
    "prompt_tmpl": "chatml"
  }
}'

> Make sure you have downloaded a quantized ggufv2 model. Example: https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF

```

### 2. Unload model

```bash
curl --request POST \
  --url http://localhost:8000/config/v1/unload-model
```



## OpenAPI spec

- Open [http://localhost:8000/docs](http://localhost:8000/docs) to open swagger ui
- Open [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) to access the raw openapi spec