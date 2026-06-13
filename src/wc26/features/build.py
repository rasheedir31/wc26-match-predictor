# Assemble the engineered feature matrix from validated match results.
#
# Correctness contract: every feature for a match is computed from **only** matches
# that occurred before it. This is enforced structurally - a single chronological
# pass where, for each match, we *read* the featurizer's current state to emit
# features, then *update* it with that match's result. The match never contributes
# to its own features, so there is no look-ahead leakage. This is what makes the
# time-based CV valid.

from __future__ import annotations

import logging

import pandas as pd

from wc26 import schema
from wc26.features.featurizer import MatchupFeaturizer, add_diffs, fill_defaults

logger = logging.getLogger(__name__)


def build_features(results: pd.DataFrame) -> pd.DataFrame:
    # Build the per-match feature matrix (one row per historical match).
    #
    # Expects a validated results frame (see ``wc26.etl.validate``). Returns metadata
    # columns + the 1X2 ``target`` + every column in ``schema.FEATURE_COLUMNS``.
    df = results.sort_values([schema.COL_DATE, schema.COL_MATCH_ID]).reset_index(drop=True)
    featurizer = MatchupFeaturizer()

    rows: list[dict[str, object]] = []
    for r in df.itertuples(index=False):
        home = getattr(r, schema.COL_HOME_TEAM)
        away = getattr(r, schema.COL_AWAY_TEAM)
        date = getattr(r, schema.COL_DATE)
        hs = int(getattr(r, schema.COL_HOME_SCORE))
        as_ = int(getattr(r, schema.COL_AWAY_SCORE))
        neutral = bool(getattr(r, schema.COL_NEUTRAL))
        tournament = getattr(r, schema.COL_TOURNAMENT)

        # --- READ pre-match state (features) ---
        meta: dict[str, object] = {
            schema.COL_MATCH_ID: getattr(r, schema.COL_MATCH_ID),
            schema.COL_DATE: date,
            schema.COL_HOME_TEAM: home,
            schema.COL_AWAY_TEAM: away,
            schema.COL_TOURNAMENT: tournament,
            schema.COL_HOME_SCORE: hs,
            schema.COL_AWAY_SCORE: as_,
            schema.COL_TARGET: schema.result_to_outcome(hs, as_).value,
        }
        feats = featurizer.raw_features(
            home, away, neutral=neutral, date=date, tournament=tournament
        )
        rows.append(meta | feats)

        # --- UPDATE state with this match (after features are recorded) ---
        featurizer.update(home, away, hs, as_, neutral=neutral, date=date)

    feat = add_diffs(fill_defaults(pd.DataFrame(rows)))

    ordered = [
        schema.COL_MATCH_ID,
        schema.COL_DATE,
        schema.COL_HOME_TEAM,
        schema.COL_AWAY_TEAM,
        schema.COL_TOURNAMENT,
        schema.COL_HOME_SCORE,
        schema.COL_AWAY_SCORE,
        schema.COL_TARGET,
        *schema.FEATURE_COLUMNS,
    ]
    feat = feat[ordered]
    logger.info("Built feature matrix: %d rows x %d columns", feat.shape[0], feat.shape[1])
    return feat
