SHELL := /bin/bash
FLAGS = --port 8000

setup:
	conda env create -f environment.yaml

start:
	source ~/miniconda3/bin/activate llm-inference-server; python src/main.py $(FLAGS)

.PHONY: setup start
