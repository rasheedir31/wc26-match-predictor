# Data validation: turn raw source frames into clean, typed, sorted match data.
#
# Validation is a hard gate - bad rows raise rather than silently propagating into
# features and models. Returns a canonical frame sorted by date with a stable
# ``match_id``, ready for feature engineering.

from __future__ import annotations

import logging

import pandas as pd

from wc26 import schema

logger = logging.getLogger(__name__)

_TRUTHY = {"true", "1", "yes", "t"}
_FALSEY = {"false", "0", "no", "f", ""}


def _coerce_bool(series: pd.Series) -> pd.Series:
    # Coerce martj42's TRUE/FALSE (and common variants) to a real bool dtype.
    if series.dtype == bool:
        return series

    def to_bool(v: object) -> bool:
        s = str(v).strip().lower()
        if s in _TRUTHY:
            return True
        if s in _FALSEY:
            return False
        raise ValueError(f"Unparseable boolean value: {v!r}")

    return series.map(to_bool)


def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    # Validate and canonicalise the raw results frame.
    #
    # Two tiers of strictness:
    #
    # - **Structural problems raise** - missing required columns, or no usable rows
    #   left after filtering. These mean the input is the wrong shape, not merely dirty.
    # - **Row-level problems are dropped** (with a logged count) - unplayed/scheduled
    #   fixtures with blank scores, unparseable dates, empty team names, self-matches,
    #   negative or non-integer scores. Real-world feeds (e.g. martj42) legitimately
    #   carry such rows; dropping them is correct, crashing on them is not.
    #
    # Returns a frame sorted by date with non-negative integer scores, a real bool
    # ``neutral``, exact duplicates removed, and a stable integer ``match_id``.
    missing = set(schema.RESULTS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")

    out = df.loc[:, list(schema.RESULTS_COLUMNS)].copy()

    # Coerce candidate types (invalids become NaN/NaT, filtered by the mask below).
    out[schema.COL_DATE] = pd.to_datetime(out[schema.COL_DATE], format="mixed", errors="coerce")
    home_score = pd.to_numeric(out[schema.COL_HOME_SCORE], errors="coerce")
    away_score = pd.to_numeric(out[schema.COL_AWAY_SCORE], errors="coerce")
    for col in (schema.COL_HOME_TEAM, schema.COL_AWAY_TEAM):
        out[col] = out[col].astype("string").str.strip()

    # Row-level validity mask: a row must be fully usable to survive.
    valid = (
        out[schema.COL_DATE].notna()
        & home_score.notna()
        & away_score.notna()
        & (home_score >= 0)
        & (away_score >= 0)
        & (home_score == home_score.round())
        & (away_score == away_score.round())
        & out[schema.COL_HOME_TEAM].notna()
        & out[schema.COL_AWAY_TEAM].notna()
        & (out[schema.COL_HOME_TEAM] != "")
        & (out[schema.COL_AWAY_TEAM] != "")
        & (out[schema.COL_HOME_TEAM] != out[schema.COL_AWAY_TEAM])
    )
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.info("Dropped %d unusable rows (blank scores / bad dates / self-matches)", n_dropped)

    out = out.loc[valid].copy()
    if out.empty:
        raise ValueError("no valid rows remain after validation")

    out[schema.COL_HOME_SCORE] = home_score.loc[valid].astype(int)
    out[schema.COL_AWAY_SCORE] = away_score.loc[valid].astype(int)
    out[schema.COL_NEUTRAL] = _coerce_bool(out[schema.COL_NEUTRAL])
    out[schema.COL_TOURNAMENT] = out[schema.COL_TOURNAMENT].astype("string").str.strip()

    # Drop exact duplicate fixtures, then sort deterministically.
    before = len(out)
    out = out.drop_duplicates(
        subset=[
            schema.COL_DATE,
            schema.COL_HOME_TEAM,
            schema.COL_AWAY_TEAM,
            schema.COL_HOME_SCORE,
            schema.COL_AWAY_SCORE,
        ]
    )
    dropped = before - len(out)
    if dropped:
        logger.info("Dropped %d duplicate result rows", dropped)

    out = out.sort_values(
        [schema.COL_DATE, schema.COL_HOME_TEAM, schema.COL_AWAY_TEAM]
    ).reset_index(drop=True)
    out[schema.COL_MATCH_ID] = out.index.astype(int)

    logger.info(
        "Validated results: %d matches, %s to %s",
        len(out),
        out[schema.COL_DATE].min().date(),
        out[schema.COL_DATE].max().date(),
    )
    return out
