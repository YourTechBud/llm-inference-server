setup:
	conda env create -f environment.yaml

start:
	python src/main.py

.PHONY: setup start