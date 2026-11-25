.PHONY: install test lint format demo data train compare decay serve docker clean

PY ?= python3
DATA ?= data/hc3_all.jsonl
MAX_SAMPLES ?= 60000

install:            ## install the package with dev tools
	$(PY) -m pip install -e ".[dev]"

test:               ## run the test suite with coverage
	$(PY) -m pytest --cov=detectability --cov-report=term-missing

lint:               ## static checks
	ruff check .
	ruff format --check .

format:
	ruff format .

demo:               ## zero-download demo model (synthetic data)
	detectability demo

data:               ## download HC3 (~74 MB)
	$(PY) scripts/download_hc3.py --subset all

train:              ## train the random forest on HC3
	detectability train --data $(DATA) --max-samples $(MAX_SAMPLES)

compare:            ## train and compare all four detectors
	detectability train --data $(DATA) --max-samples $(MAX_SAMPLES) --compare

decay:              ## run the decay experiment across all detectors
	detectability decay --data $(DATA) --max-samples $(MAX_SAMPLES) --generations 8

serve:              ## start the web app on http://127.0.0.1:8000
	detectability serve

docker:
	docker build -t ai-detectability .
	docker run --rm -p 8000:8000 ai-detectability

clean:
	rm -rf models results .pytest_cache .ruff_cache .coverage htmlcov
