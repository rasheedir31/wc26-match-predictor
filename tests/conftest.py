# Shared pytest fixtures: tiny, hand-checkable match frames.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc26 import schema
from wc26.etl.validate import validate_results
from wc26.features.build import build_features


def _match(date, home, away, hs, as_, tournament="Friendly", neutral="FALSE"):
    return {
        schema.COL_DATE: date,
        schema.COL_HOME_TEAM: home,
        schema.COL_AWAY_TEAM: away,
        schema.COL_HOME_SCORE: hs,
        schema.COL_AWAY_SCORE: as_,
        schema.COL_TOURNAMENT: tournament,
        schema.COL_CITY: "Nowhere",
        schema.COL_COUNTRY: home,
        schema.COL_NEUTRAL: neutral,
    }


@pytest.fixture
def tiny_results() -> pd.DataFrame:
    # A small, deterministic set of matches among three teams.
    #
    # Chronology (A, B, C):
    #     2020-01-01  A 2-0 B   (home win, competitive)
    #     2020-01-10  B 1-1 C   (draw)
    #     2020-01-20  A 0-1 C   (away win)
    #     2020-02-01  A 3-1 B   (A vs B rematch; gives A-B head-to-head history)
    rows = [
        _match("2020-01-01", "A", "B", 2, 0, tournament="FIFA World Cup qualification"),
        _match("2020-01-10", "B", "C", 1, 1),
        _match("2020-01-20", "A", "C", 0, 1),
        _match("2020-02-01", "A", "B", 3, 1, tournament="FIFA World Cup qualification"),
    ]
    return pd.DataFrame(rows)


def make_synthetic_results(n_teams: int = 8, n_rounds: int = 10, seed: int = 0) -> pd.DataFrame:
    # Deterministic synthetic results with latent team strengths + home advantage.
    #
    # Goals are Poisson around strength-driven rates, so stronger teams genuinely win
    # more - enough signal for every model to fit and be sanity-checked.
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = {t: float(rng.normal(0.0, 0.5)) for t in teams}
    date = pd.Timestamp("2015-01-01")
    rows = []
    for r in range(n_rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i >= j:
                    continue
                date += pd.Timedelta(days=2)
                lam_h = np.exp(0.3 + strength[teams[i]] - strength[teams[j]])
                lam_a = np.exp(strength[teams[j]] - strength[teams[i]])
                hs = int(rng.poisson(lam_h))
                as_ = int(rng.poisson(lam_a))
                rows.append(
                    _match(
                        date.strftime("%Y-%m-%d"),
                        teams[i],
                        teams[j],
                        hs,
                        as_,
                        tournament="Friendly" if r % 3 == 0 else "FIFA World Cup qualification",
                    )
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_clean() -> pd.DataFrame:
    # Validated synthetic results (raw match rows) for featurizer/serving tests.
    return validate_results(make_synthetic_results())


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    # A realistic engineered feature matrix (~280 matches, 8 teams) for model tests.
    return build_features(validate_results(make_synthetic_results()))
