# Airflow DAG for the WC26 ETL + feature pipeline.
#
# Thin wrapper only: every task delegates to a plain function in ``wc26.pipeline``.
# No pipeline logic lives here - this exists so the same code that runs under
# ``make pipeline`` (and the GitHub Actions schedule) can be demonstrated as an
# Airflow DAG locally. Airflow runs inside its own container (``docker/airflow.Dockerfile``,
# ) where both ``apache-airflow`` and the ``wc26`` package are installed; it is
# intentionally not a dependency of the app environment.

from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

from wc26.pipeline import run_features, run_ingest, run_load, run_validate

with DAG(
    dag_id="wc26_pipeline",
    description="Ingest -> validate -> feature-engineer -> load (WC26 match data)",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["wc26", "etl"],
) as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=run_ingest)
    validate = PythonOperator(task_id="validate", python_callable=run_validate)
    features = PythonOperator(task_id="features", python_callable=run_features)
    load = PythonOperator(task_id="load", python_callable=run_load)

    # Linear dependency: each stage reads the previous stage's artifacts from disk.
    ingest >> validate >> features >> load
