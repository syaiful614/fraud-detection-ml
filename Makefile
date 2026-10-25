.PHONY: install data train train-fast test lint format serve docker-build docker-run clean

PYTHON ?= python
CONFIG ?= configs/default.yaml

install:            ## install package + dev tools in editable mode
	$(PYTHON) -m pip install -e ".[dev]"

data:               ## download the dataset (no Kaggle account needed)
	$(PYTHON) -m fraudlab download

train:              ## full experiment: compare, tune, evaluate, explain, save model + reports
	$(PYTHON) -m fraudlab train --config $(CONFIG)

train-fast:         ## quick run (few trials) for a sanity check
	$(PYTHON) -m fraudlab train --config configs/fast.yaml

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check . && ruff format --check .

format:
	ruff check . --fix && ruff format .

serve:              ## start the API on http://127.0.0.1:8000/docs
	$(PYTHON) -m fraudlab serve

docker-build:
	docker build -t fraud-detection-api .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models fraud-detection-api

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
