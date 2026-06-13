# Common model contract.
#
# Every model - Elo, Dixon-Coles Poisson, logistic, XGBoost - implements the same
# interface:
#
#     model.fit(train_df).predict_proba(fixtures_df) -> ndarray, shape (n, 3)
#
# The three columns are **always** ordered (home win, draw, away win) and each row
# sums to 1. Inputs are feature-matrix frames (see ``wc26.features.build``): models
# read whichever columns they need (team names + Elo for Elo, raw scores for Poisson,
# engineered features for logistic/XGBoost), so they are interchangeable in
# evaluation and serving.

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from wc26 import schema

# Canonical class order for the 1X2 target. Ordered H < D < A so the ordinal RPS
# metric and any ordered model see outcomes in natural order.
LABEL_ORDER: tuple[str, str, str] = (
    schema.Outcome.HOME.value,
    schema.Outcome.DRAW.value,
    schema.Outcome.AWAY.value,
)
LABEL_TO_IDX: dict[str, int] = {label: i for i, label in enumerate(LABEL_ORDER)}

_EPS = 1e-15


def encode_targets(targets: pd.Series) -> np.ndarray:
    # Map 'H'/'D'/'A' labels to class indices 0/1/2 (home/draw/away).
    return targets.map(LABEL_TO_IDX).to_numpy()


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    # Clip away exact zeros and renormalise rows to sum to 1 (keeps log loss finite).
    proba = np.clip(np.asarray(proba, dtype=float), _EPS, None)
    return proba / proba.sum(axis=1, keepdims=True)


def proba_to_frame(proba: np.ndarray) -> pd.DataFrame:
    # Wrap a (n, 3) probability array in a frame with the canonical column names.
    return pd.DataFrame(proba, columns=list(schema.PROBA_COLUMNS))


class MatchPredictor(ABC):
    # Abstract base for all 1X2 match-outcome models.

    #: Short identifier used in MLflow runs, the registry, and the dashboard.
    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> MatchPredictor:
        # Fit on a training frame (chronologically earlier matches). Returns self.
        ...

    @abstractmethod
    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        # Predict P(home win, draw, away win) for each row; shape (n, 3), rows sum to 1.
        ...

    def predict_frame(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        # Convenience: probabilities as a named DataFrame (p_home, p_draw, p_away).
        return proba_to_frame(self.predict_proba(fixtures))
