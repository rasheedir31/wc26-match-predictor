# Canonical column names and the 1X2 outcome labels.
#
# Centralised so ingest, ETL, features, models, and the simulation all agree on
# identifiers - no stringly-typed column names scattered across modules.

from __future__ import annotations

from enum import StrEnum


class Outcome(StrEnum):
    # The 1X2 match outcome from the home team's perspective.

    HOME = "H"
    DRAW = "D"
    AWAY = "A"


# Probability columns in the canonical predict() output order. Every model returns
# probabilities in exactly this order (see wc26.models.base).
PROBA_COLUMNS: tuple[str, str, str] = ("p_home", "p_draw", "p_away")


# --- Raw match schema (martj42 results.csv) -------------------------------------
COL_DATE = "date"
COL_HOME_TEAM = "home_team"
COL_AWAY_TEAM = "away_team"
COL_HOME_SCORE = "home_score"
COL_AWAY_SCORE = "away_score"
COL_TOURNAMENT = "tournament"
COL_CITY = "city"
COL_COUNTRY = "country"
COL_NEUTRAL = "neutral"

RESULTS_COLUMNS: tuple[str, ...] = (
    COL_DATE,
    COL_HOME_TEAM,
    COL_AWAY_TEAM,
    COL_HOME_SCORE,
    COL_AWAY_SCORE,
    COL_TOURNAMENT,
    COL_CITY,
    COL_COUNTRY,
    COL_NEUTRAL,
)

# --- Derived / engineered columns ----------------------------------------------
COL_MATCH_ID = "match_id"
COL_TARGET = "target"  # Outcome value: "H" / "D" / "A"
COL_IS_COMPETITIVE = "is_competitive"  # tournament is not a friendly

# Feature columns (all strictly pre-match / point-in-time).
COL_ELO_HOME = "elo_home"
COL_ELO_AWAY = "elo_away"
COL_ELO_DIFF = "elo_diff"
COL_FORM_HOME = "form_home"
COL_FORM_AWAY = "form_away"
COL_FORM_DIFF = "form_diff"
COL_GD_HOME = "gd_home"
COL_GD_AWAY = "gd_away"
COL_GD_DIFF = "gd_diff"
COL_REST_HOME = "rest_home"
COL_REST_AWAY = "rest_away"
COL_REST_DIFF = "rest_diff"
COL_H2H_HOME_RATE = "h2h_home_rate"  # home team's win rate in prior meetings
COL_H2H_MATCHES = "h2h_matches"  # number of prior meetings counted

FEATURE_COLUMNS: tuple[str, ...] = (
    COL_ELO_HOME,
    COL_ELO_AWAY,
    COL_ELO_DIFF,
    COL_FORM_HOME,
    COL_FORM_AWAY,
    COL_FORM_DIFF,
    COL_GD_HOME,
    COL_GD_AWAY,
    COL_GD_DIFF,
    COL_REST_HOME,
    COL_REST_AWAY,
    COL_REST_DIFF,
    COL_H2H_HOME_RATE,
    COL_H2H_MATCHES,
    COL_NEUTRAL,
    COL_IS_COMPETITIVE,
)


def result_to_outcome(home_score: int, away_score: int) -> Outcome:
    # Map a final score to its 1X2 outcome (home team's perspective).
    if home_score > away_score:
        return Outcome.HOME
    if home_score < away_score:
        return Outcome.AWAY
    return Outcome.DRAW
