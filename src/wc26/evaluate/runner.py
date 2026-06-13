# Cross-validation runner: score a model under time-based CV.
#
# For each time fold we fit a *fresh* model on the past block and predict the future
# block. We report per-fold metrics, their mean, and metrics on the pooled
# out-of-fold predictions, plus a reliability curve for the home-win class.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wc26 import schema
from wc26.config import settings
from wc26.evaluate.cv import time_series_folds
from wc26.evaluate.metrics import all_metrics, calibration_curve
from wc26.models.base import MatchPredictor, encode_targets

_METRIC_KEYS = ("log_loss", "brier", "rps", "accuracy")


@dataclass
class EvalResult:
    # Aggregated evaluation of one model under time-based CV.

    name: str
    n_folds: int
    n_eval: int
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    pooled_metrics: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, list[float]] = field(default_factory=dict)


def cross_validate(
    name: str,
    factory: Callable[[], MatchPredictor],
    df: pd.DataFrame,
    n_splits: int | None = None,
) -> EvalResult:
    # Evaluate ``factory()`` under expanding-window time CV on feature frame ``df``.
    oof_proba: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []

    for train, test in time_series_folds(df, n_splits):
        model = factory().fit(train)
        proba = model.predict_proba(test)
        y = encode_targets(test[schema.COL_TARGET])
        fold_metrics.append(all_metrics(proba, y))
        oof_proba.append(proba)
        oof_y.append(y)

    pooled_proba = np.vstack(oof_proba)
    pooled_y = np.concatenate(oof_y)
    mean_metrics = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in _METRIC_KEYS}
    pooled_metrics = all_metrics(pooled_proba, pooled_y)
    calib = calibration_curve(
        pooled_proba, pooled_y, class_idx=0, n_bins=settings.eval.calibration_bins
    )

    return EvalResult(
        name=name,
        n_folds=len(fold_metrics),
        n_eval=len(pooled_y),
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        pooled_metrics=pooled_metrics,
        calibration=calib,
    )
