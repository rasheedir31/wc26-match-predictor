# Local-only Airflow image (never deployed - see the README scope guardrails).
# Extends the official Airflow image and installs the wc26 package + its runtime
# deps so the DAG (dags/wc26_pipeline.py) can import the same task functions that
# `make pipeline` and GitHub Actions call. Airflow itself supplies the scheduler/UI;
# this is the orchestration *showcase*, run via docker-compose, not in production.

FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Install the project's runtime dependencies (constrained to Airflow's environment)
# and the package itself in editable mode (mounted at /opt/airflow/wc26 by compose).
COPY --chown=airflow:airflow pyproject.toml /opt/airflow/wc26/pyproject.toml
RUN pip install --no-cache-dir \
    "pandas>=2.2" "numpy>=1.26" "scipy>=1.13" "scikit-learn>=1.5" "xgboost>=2.1" \
    "pyarrow>=17.0" "sqlalchemy>=1.4,<2.0" "psycopg2-binary>=2.9" "pydantic-settings>=2.5" \
    "requests>=2.32"

ENV PYTHONPATH=/opt/airflow/wc26/src
