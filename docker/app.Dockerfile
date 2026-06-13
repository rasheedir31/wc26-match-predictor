# Deployable app image: FastAPI + Streamlit, serving the bundled snapshot.
# This is the ONLY image that deploys (to Render). It is self-contained - it reads
# data/snapshot/ at runtime and needs no database, warehouse, Airflow, or MLflow.
#
# Build:  docker build -f docker/app.Dockerfile -t wc26-app .
# Run:    docker run -p 8501:8501 -p 8000:8000 wc26-app

FROM python:3.12-slim AS base

# uv for fast, reproducible installs from the committed lockfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime deps first (no dev group) for layer caching. README.md is included
# because the package metadata (pyproject `readme = `) references it at build time.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen --no-install-project

# App code + the bundled snapshot (the runtime data the app serves).
COPY src/ ./src/
COPY data/snapshot/ ./data/snapshot/
RUN uv sync --no-dev --frozen

# Render provides $PORT; the launcher binds Streamlit to it (API on 8000 internally).
EXPOSE 8501 8000
ENV PORT=8501

CMD ["python", "-m", "wc26.run_app"]
