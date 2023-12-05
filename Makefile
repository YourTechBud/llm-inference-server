FLAGS = --port 8000

setup:
	conda env create -f environment.yaml

start:
	python src/main.py $(FLAGS)

.PHONY: setup start
