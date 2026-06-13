# WC26 Match Predictor - canonical commands.
# Every target wraps `uv` or docker compose so the same entrypoints run locally,
# in CI, and (for the pipeline) in GitHub Actions. see the README.

.PHONY: help setup lint format test up down pipeline train monitor snapshot app api dashboard clean

help:  ## Show available targets
	@echo "Targets:"
	@echo "  setup     Install deps (uv sync) + pre-commit hooks"
	@echo "  lint      ruff check + ruff format --check"
	@echo "  format    ruff format + ruff check --fix"
	@echo "  test      pytest with coverage"
	@echo "  up        docker compose up (airflow, postgres, mlflow, app)"
	@echo "  down      docker compose down"
	@echo "  pipeline  run ETL+train+simulate once, locally (no Airflow)"
	@echo "  train     evaluate+train models and run the tournament simulation"
	@echo "  snapshot  export the portable SQLite/parquet/model snapshot"
	@echo "  app       run FastAPI + Streamlit together locally"
	@echo "  api       run just the FastAPI API"
	@echo "  dashboard run just the Streamlit dashboard"

setup:  ## uv sync + pre-commit install
	uv sync
	uv run pre-commit install

lint:  ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

test:  ## pytest with coverage
	uv run pytest

up:  ## full local stack
	docker compose up -d

down:  ## stop local stack
	docker compose down

pipeline:  ## run the full pipeline once, locally (ETL + train + simulate; no Airflow)
	uv run python -m wc26.pipeline all

train:  ## evaluate + train models and run the tournament simulation (needs features)
	uv run python -m wc26.pipeline train
	uv run python -m wc26.pipeline simulate

monitor:  ## generate the drift report + refresh the live prediction-vs-actual loop
	uv run python -m wc26.pipeline monitor

snapshot:  ## export the portable SQLite/parquet/model snapshot for the app
	uv run python -m wc26.snapshot

app:  ## run FastAPI + Streamlit together locally
	uv run python -m wc26.run_app

api:  ## run just the FastAPI prediction API
	uv run uvicorn wc26.api.app:app --host 0.0.0.0 --port 8000

dashboard:  ## run just the Streamlit dashboard
	uv run streamlit run src/wc26/dashboard/app.py

clean:  ## remove caches and build artifacts
	uv run python -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+['.pytest_cache','.ruff_cache','htmlcov','.coverage']]"
