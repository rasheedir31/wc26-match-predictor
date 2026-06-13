# XGBoost gradient-boosted trees on the engineered features (1X2).
#
# The workhorse model - expected to win on aggregate metrics. Multi-class softprob
# over (home, draw, away); reads ``schema.FEATURE_COLUMNS``. Targets are encoded
# 0/1/2 so the predicted-probability columns match the contract order.

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from wc26 import schema
from wc26.config import settings
from wc26.models.base import MatchPredictor, encode_targets, normalize_proba


class XGBoostModel(MatchPredictor):
    # Gradient-boosted decision trees (multi:softprob).

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        self.model: XGBClassifier | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[list(schema.FEATURE_COLUMNS)].to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame) -> XGBoostModel:
        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=settings.model.random_seed,
            n_jobs=-1,
            **self.params,
        )
        self.model.fit(self._matrix(train), encode_targets(train[schema.COL_TARGET]))
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "model not fitted"
        return normalize_proba(self.model.predict_proba(self._matrix(fixtures)))
