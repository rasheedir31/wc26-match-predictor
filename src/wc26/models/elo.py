# Elo 1X2 model (baseline).
#
# Elo natively yields an *expected score* ``E`` for the home team (its win
# probability plus half its draw probability), not a three-way split. We turn ``E``
# into P(home/draw/away) with a one-parameter draw model and fit that parameter by
# maximum likelihood (minimising log loss) on the training data.
#
# Draw model. With draw probability ``d`` and expected score ``E``::
#
#     E = P(home) * 1 + P(draw) * 0.5 + P(away) * 0   =>   P(home) = E - d/2
#     P(home) + P(draw) + P(away) = 1                 =>   P(away) = 1 - E - d/2
#
# Both stay non-negative iff ``d <= 2 * min(E, 1 - E) = 1 - |2E - 1|``. So we set::
#
#     d = theta * (1 - |2E - 1|),   theta in [0, 1]
#
# which is automatically feasible and intuitive: evenly matched games (E ~ 0.5) draw
# most, lopsided games draw least. ``theta`` is the only fitted parameter.
#
# The Elo *ratings* themselves are the point-in-time ``elo_home`` / ``elo_away``
# feature columns (computed leak-free upstream), so this model needs no rating pass.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from wc26 import schema
from wc26.config import settings
from wc26.features.elo import ELO_SCALE
from wc26.models.base import MatchPredictor, encode_targets, normalize_proba

_EPS = 1e-15


def _expected_home_score(fixtures: pd.DataFrame, home_advantage: float) -> np.ndarray:
    # Elo expected home score from the rating columns (home advantage off at
    # neutral venues).
    elo_h = fixtures[schema.COL_ELO_HOME].to_numpy(dtype=float)
    elo_a = fixtures[schema.COL_ELO_AWAY].to_numpy(dtype=float)
    neutral = fixtures[schema.COL_NEUTRAL].to_numpy(dtype=float)
    diff = (elo_h + home_advantage * (1.0 - neutral)) - elo_a
    return 1.0 / (1.0 + 10.0 ** (-diff / ELO_SCALE))


def _proba_from_expected(expected: np.ndarray, theta: float) -> np.ndarray:
    # Split expected score into [P(home), P(draw), P(away)] via the draw model.
    draw = theta * (1.0 - np.abs(2.0 * expected - 1.0))
    p_home = expected - draw / 2.0
    p_away = 1.0 - expected - draw / 2.0
    return np.column_stack([p_home, draw, p_away])


class EloModel(MatchPredictor):
    # Elo expected score + a fitted one-parameter draw model.

    name = "elo"

    def __init__(self, home_advantage: float | None = None) -> None:
        self.home_advantage = (
            settings.model.elo_home_advantage if home_advantage is None else home_advantage
        )
        self.theta_: float = 0.0

    def fit(self, train: pd.DataFrame) -> EloModel:
        expected = _expected_home_score(train, self.home_advantage)
        y = encode_targets(train[schema.COL_TARGET])

        def neg_log_likelihood(theta: float) -> float:
            proba = normalize_proba(_proba_from_expected(expected, theta))
            return float(-np.log(proba[np.arange(len(y)), y] + _EPS).mean())

        # theta in (0, 1); bounded scalar minimisation of log loss.
        res = minimize_scalar(neg_log_likelihood, bounds=(0.0, 0.999), method="bounded")
        self.theta_ = float(res.x)
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        expected = _expected_home_score(fixtures, self.home_advantage)
        return normalize_proba(_proba_from_expected(expected, self.theta_))
