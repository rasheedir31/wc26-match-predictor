# Unit tests for the validation gate.

from __future__ import annotations

import pandas as pd
import pytest

from wc26 import schema
from wc26.etl.validate import validate_results


def test_validate_canonicalises_types_and_sorts(tiny_results) -> None:
    # Feed rows out of order to confirm sorting.
    shuffled = tiny_results.iloc[::-1].reset_index(drop=True)
    out = validate_results(shuffled)

    assert out[schema.COL_DATE].is_monotonic_increasing
    assert out[schema.COL_NEUTRAL].dtype == bool
    assert out[schema.COL_HOME_SCORE].dtype.kind == "i"
    # Stable integer match_id, contiguous from 0.
    assert list(out[schema.COL_MATCH_ID]) == list(range(len(out)))


def test_validate_coerces_neutral_strings() -> None:
    # Two distinct matches (different dates) so dedupe keeps both.
    rows = pd.DataFrame(
        [
            {
                schema.COL_DATE: date,
                schema.COL_HOME_TEAM: "A",
                schema.COL_AWAY_TEAM: "B",
                schema.COL_HOME_SCORE: 1,
                schema.COL_AWAY_SCORE: 0,
                schema.COL_TOURNAMENT: "Friendly",
                schema.COL_CITY: "X",
                schema.COL_COUNTRY: "A",
                schema.COL_NEUTRAL: val,
            }
            for date, val in (("2020-01-01", "TRUE"), ("2020-01-02", "FALSE"))
        ]
    )
    out = validate_results(rows)
    # Rows are sorted by date, so order is preserved here.
    assert list(out[schema.COL_NEUTRAL]) == [True, False]


def test_validate_drops_exact_duplicates(tiny_results) -> None:
    dup = pd.concat([tiny_results, tiny_results.iloc[[0]]], ignore_index=True)
    out = validate_results(dup)
    assert len(out) == len(tiny_results)


# --- Structural problems (wrong shape) -> raise --------------------------------


def test_validate_requires_columns(tiny_results) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_results(tiny_results.drop(columns=[schema.COL_HOME_SCORE]))


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda df: df.assign(home_score=[None] * 4), id="all-blank-scores"),
        pytest.param(lambda df: df.assign(date=["nope", "x", "y", "z"]), id="all-bad-dates"),
        pytest.param(lambda df: df.assign(away_team=df["home_team"]), id="all-self-matches"),
    ],
)
def test_validate_raises_when_nothing_valid(tiny_results, mutate) -> None:
    with pytest.raises(ValueError, match="no valid rows"):
        validate_results(mutate(tiny_results))


# --- Row-level problems (dirty data) -> drop the bad rows, keep the good -------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda df: df.assign(home_score=[None, 0, 1, 3]), id="one-blank-score"),
        pytest.param(lambda df: df.assign(home_score=[-1, 0, 1, 3]), id="one-negative-score"),
        pytest.param(lambda df: df.assign(home_score=[1.5, 0, 1, 3]), id="one-non-integer-score"),
        pytest.param(
            lambda df: df.assign(date=["nope", "2020-01-10", "2020-01-20", "2020-02-01"]),
            id="one-bad-date",
        ),
    ],
)
def test_validate_drops_one_bad_row_keeps_rest(tiny_results, mutate) -> None:
    out = validate_results(mutate(tiny_results))
    assert len(out) == len(tiny_results) - 1  # exactly the one bad row is gone
