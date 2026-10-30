PY ?= python3.11
VENV = .venv
ifeq ($(OS),Windows_NT)
BIN = $(VENV)/Scripts
else
BIN = $(VENV)/bin
endif

.PHONY: setup data train evaluate serve dashboard test lint docker

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt
	$(BIN)/python -m pip install -e .

data:
	$(BIN)/python scripts/download_data.py

train:
	$(BIN)/python -m fraud.train

evaluate:
	$(BIN)/python -m fraud.evaluate

serve:
	$(BIN)/uvicorn fraud.api.main:app --reload --port 8000

dashboard:
	$(BIN)/streamlit run app/dashboard.py

test:
	$(BIN)/pytest --cov=src/fraud --cov-report=term-missing

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

docker:
	docker compose up --build
