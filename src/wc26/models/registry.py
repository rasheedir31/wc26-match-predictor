# Model registry - the canonical set of models to train and compare.

from __future__ import annotations

from collections.abc import Callable

from wc26.models.base import MatchPredictor
from wc26.models.elo import EloModel
from wc26.models.gbm import XGBoostModel
from wc26.models.logistic import LogisticModel
from wc26.models.poisson import DixonColesModel

# Factories (not instances) so each evaluation fold gets a fresh, unfitted model.
MODEL_FACTORIES: dict[str, Callable[[], MatchPredictor]] = {
    "elo": EloModel,
    "dixon_coles": DixonColesModel,
    "logistic": LogisticModel,
    "xgboost": XGBoostModel,
}


def default_models() -> dict[str, Callable[[], MatchPredictor]]:
    # Return the canonical model factories keyed by name.
    return dict(MODEL_FACTORIES)
