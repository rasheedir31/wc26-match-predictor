# Dixon-Coles behavioural + correctness checks.

from __future__ import annotations

import numpy as np
import pandas as pd

from wc26 import schema
from wc26.etl.validate import validate_results
from wc26.models.poisson import DixonColesModel


def _fixture(home: str, away: str, neutral: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                schema.COL_HOME_TEAM: home,
                schema.COL_AWAY_TEAM: away,
                schema.COL_NEUTRAL: neutral,
            }
        ]
    )


def _strong_weak_results() -> pd.DataFrame:
    # Strong beats Weak repeatedly; Even1 vs Even2 trade draws/splits.
    rows = []
    date = pd.Timestamp("2018-01-01")
    for _ in range(30):
        date += pd.Timedelta(days=5)
        rows.append(
            {
                schema.COL_DATE: date,
                schema.COL_HOME_TEAM: "Strong",
                schema.COL_AWAY_TEAM: "Weak",
                schema.COL_HOME_SCORE: 3,
                schema.COL_AWAY_SCORE: 0,
                schema.COL_TOURNAMENT: "Friendly",
                schema.COL_CITY: "X",
                schema.COL_COUNTRY: "X",
                schema.COL_NEUTRAL: "TRUE",
            }
        )
    return validate_results(pd.DataFrame(rows))


def test_dixon_coles_favours_stronger_team() -> None:
    model = DixonColesModel().fit(_strong_weak_results())
    p = model.predict_proba(_fixture("Strong", "Weak"))[0]
    assert p.sum() == np.float64(1.0) or np.isclose(p.sum(), 1.0)
    # Strong should be a clear favourite over Weak.
    assert p[0] > p[2]
    assert p[0] > 0.5


def test_dixon_coles_unknown_teams_are_balanced() -> None:
    model = DixonColesModel().fit(_strong_weak_results())
    # Two unseen teams default to average strength -> symmetric at a neutral venue.
    p = model.predict_proba(_fixture("Ghost1", "Ghost2"))[0]
    assert np.isclose(p[0], p[2], atol=1e-6)


def test_dixon_coles_rho_and_params_are_finite() -> None:
    model = DixonColesModel().fit(_strong_weak_results())
    assert np.isfinite(model.rho_)
    assert np.isfinite(model.gamma_)
    assert np.all(np.isfinite(model.attack_))
    assert np.all(np.isfinite(model.defense_))
    # Attack/defence are mean-centred for identifiability.
    assert np.isclose(model.attack_.mean(), 0.0, atol=1e-8)
