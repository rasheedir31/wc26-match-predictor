# MLflow experiment tracking helpers.
#
# Every model's CV evaluation is logged as an MLflow run (params, CV-mean and pooled
# out-of-fold metrics, calibration curve, and the fitted-model artifact). The best
# model is marked as champion.
#
# Note on the registry: the MLflow *Model Registry* requires a database-backed store.
# This project deliberately uses the zero-infra local **file** store, so "registering" the champion means tagging its run ``champion=true``
# and persisting the fitted model into the portable snapshot. All logging is
# best-effort: a tracking hiccup logs a warning and never breaks training.

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import joblib
import mlflow

from wc26.config import settings
from wc26.evaluate.runner import EvalResult
from wc26.models.base import MatchPredictor

logger = logging.getLogger(__name__)


def init_mlflow() -> bool:
    # Point MLflow at the configured tracking store and experiment.
    #
    # We use the zero-infrastructure local **file** store (the README: no DB). MLflow
    # 3.x gates the file store behind an explicit opt-in, which we set here. Returns
    # True on success; a failure is logged and returns False (training continues).
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        return True
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        logger.warning("MLflow init failed (%s); continuing without tracking", exc)
        return False


def log_evaluation(
    result: EvalResult,
    params: dict[str, object] | None = None,
    model: MatchPredictor | None = None,
) -> str | None:
    # Log one model's evaluation as an MLflow run; return the run id (or None).
    try:
        with mlflow.start_run(run_name=result.name) as run:
            mlflow.log_params(
                {"model": result.name, "n_folds": result.n_folds, "n_eval": result.n_eval}
                | dict(params or {})
            )
            mlflow.log_metrics({f"cv_{k}": v for k, v in result.mean_metrics.items()})
            mlflow.log_metrics({f"oof_{k}": v for k, v in result.pooled_metrics.items()})
            _log_json(result.calibration, "calibration.json")
            if model is not None:
                _log_model_artifact(model)
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        logger.warning("MLflow logging failed for %s: %s", result.name, exc)
        return None


def mark_champion(run_id: str | None) -> None:
    # Tag a run as the champion model (our file-store stand-in for registration).
    if run_id is None:
        return
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("champion", "true")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not tag champion run %s: %s", run_id, exc)


def select_best(results: Iterable[EvalResult]) -> EvalResult:
    # Pick the best model by pooled out-of-fold log loss (lower is better).
    return min(results, key=lambda r: r.pooled_metrics["log_loss"])


def _log_json(obj: object, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(json.dumps(obj, indent=2))
        mlflow.log_artifact(str(path))


def _log_model_artifact(model: MatchPredictor) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{model.name}.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))
