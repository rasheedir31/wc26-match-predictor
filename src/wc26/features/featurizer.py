# Reusable matchup featurizer.
#
# Bundles the four stateful accumulators (Elo, form, rest, head-to-head) behind one
# object so the *same* point-in-time logic serves two callers:
#
# - ``wc26.features.build`` - one chronological pass over history to build the
#   training matrix (read features, then update state with the result).
# - prediction (``wc26.simulate``, the API) - after fitting on all history, the
#   featurizer holds each team's current state and can emit a feature row for any
#   hypothetical fixture, with no duplicated feature code.

from __future__ import annotations

import pandas as pd

from wc26 import schema
from wc26.config import settings
from wc26.features.elo import EloRatings
from wc26.features.form import RecentForm
from wc26.features.h2h import HeadToHead
from wc26.features.rest import RestTracker


def is_competitive(tournament: str) -> bool:
    # A match is competitive unless it is an international friendly.
    return str(tournament).strip().lower() != "friendly"


def fill_defaults(feat: pd.DataFrame) -> pd.DataFrame:
    # Fill cold-start NaNs (teams/pairs with no prior history) with neutral values.
    fp = settings.features
    fills = {
        schema.COL_FORM_HOME: fp.points_draw,
        schema.COL_FORM_AWAY: fp.points_draw,
        schema.COL_GD_HOME: 0.0,
        schema.COL_GD_AWAY: 0.0,
        schema.COL_H2H_HOME_RATE: 0.5,  # no prior meetings -> even
    }
    return feat.fillna(fills)


def add_diffs(feat: pd.DataFrame) -> pd.DataFrame:
    # Home-minus-away difference columns (computed after defaults are filled).
    feat[schema.COL_ELO_DIFF] = feat[schema.COL_ELO_HOME] - feat[schema.COL_ELO_AWAY]
    feat[schema.COL_FORM_DIFF] = feat[schema.COL_FORM_HOME] - feat[schema.COL_FORM_AWAY]
    feat[schema.COL_GD_DIFF] = feat[schema.COL_GD_HOME] - feat[schema.COL_GD_AWAY]
    feat[schema.COL_REST_DIFF] = feat[schema.COL_REST_HOME] - feat[schema.COL_REST_AWAY]
    return feat


class MatchupFeaturizer:
    # Holds the per-team feature state and emits pre-match feature rows.

    def __init__(self) -> None:
        self.elo = EloRatings()
        self.form = RecentForm()
        self.rest = RestTracker()
        self.h2h = HeadToHead()

    def raw_features(
        self, home: str, away: str, *, neutral: bool, date: pd.Timestamp, tournament: str
    ) -> dict[str, object]:
        # Pre-match feature dict (pre-fill, pre-diff) from current state only.
        row: dict[str, object] = {
            schema.COL_ELO_HOME: self.elo.get(home),
            schema.COL_ELO_AWAY: self.elo.get(away),
            schema.COL_NEUTRAL: int(neutral),
            schema.COL_IS_COMPETITIVE: int(is_competitive(tournament)),
        }
        row.update(self.form.pre_match(home, away))
        row.update(self.rest.pre_match(home, away, date))
        row.update(self.h2h.pre_match(home, away))
        return row

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        *,
        neutral: bool,
        date: pd.Timestamp,
    ) -> None:
        # Fold one observed result into all accumulators.
        self.elo.update(home, away, home_score, away_score, neutral)
        self.form.update(home, away, home_score, away_score)
        self.rest.update(home, away, date)
        self.h2h.update(home, away, home_score, away_score)

    @classmethod
    def fitted_on(cls, results: pd.DataFrame) -> MatchupFeaturizer:
        # Walk all (validated, chronologically sorted) results and return the
        # featurizer holding final per-team state - ready to feature future fixtures.
        f = cls()
        ordered = results.sort_values([schema.COL_DATE, schema.COL_MATCH_ID])
        for r in ordered.itertuples(index=False):
            f.update(
                getattr(r, schema.COL_HOME_TEAM),
                getattr(r, schema.COL_AWAY_TEAM),
                int(getattr(r, schema.COL_HOME_SCORE)),
                int(getattr(r, schema.COL_AWAY_SCORE)),
                neutral=bool(getattr(r, schema.COL_NEUTRAL)),
                date=getattr(r, schema.COL_DATE),
            )
        return f

    def fixture_features(
        self,
        home: str,
        away: str,
        *,
        neutral: bool = True,
        date: pd.Timestamp | None = None,
        tournament: str = "FIFA World Cup",
    ) -> pd.DataFrame:
        # A single fully-finalised feature row (filled + diffed) for a hypothetical
        # fixture, including team/meta columns so every model can consume it.
        if date is None:
            date = pd.Timestamp.utcnow().tz_localize(None)
        raw = self.raw_features(home, away, neutral=neutral, date=date, tournament=tournament)
        raw |= {
            schema.COL_HOME_TEAM: home,
            schema.COL_AWAY_TEAM: away,
            schema.COL_DATE: date,
            schema.COL_TOURNAMENT: tournament,
        }
        df = add_diffs(fill_defaults(pd.DataFrame([raw])))
        return df
