# Multinomial logistic regression on the engineered features (1X2).
#
# A standard, well-calibrated linear baseline: standardise features, then a softmax
# regression over the three outcomes. Reads ``schema.FEATURE_COLUMNS``; class order is
# forced to (home, draw, away) by encoding targets as 0/1/2 so ``predict_proba``
# columns already line up with the contract.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc26 import schema
from wc26.config import settings
from wc26.models.base import MatchPredictor, encode_targets, normalize_proba


class LogisticModel(MatchPredictor):
    # Standardised multinomial logistic regression.

    name = "logistic"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.pipeline: Pipeline | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[list(schema.FEATURE_COLUMNS)].to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame) -> LogisticModel:
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        max_iter=2000,
                        random_state=settings.model.random_seed,
                    ),
                ),
            ]
        )
        self.pipeline.fit(self._matrix(train), encode_targets(train[schema.COL_TARGET]))
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None, "model not fitted"
        # classes_ are [0, 1, 2] == [home, draw, away] thanks to integer encoding.
        return normalize_proba(self.pipeline.predict_proba(self._matrix(fixtures)))
