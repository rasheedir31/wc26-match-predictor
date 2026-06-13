# Unit tests for feature calculations - exact, hand-checkable values.

from __future__ import annotations

import math

import pytest

from wc26 import schema
from wc26.etl.validate import validate_results
from wc26.features.build import build_features
from wc26.features.elo import EloRatings
from wc26.features.form import RecentForm
from wc26.features.h2h import HeadToHead
from wc26.features.rest import RestTracker

# --- Elo -----------------------------------------------------------------------


def test_elo_initial_and_neutral_expected() -> None:
    elo = EloRatings(k=32, home_advantage=65, initial=1500)
    assert elo.get("X") == 1500
    # Equal ratings at a neutral venue -> even expected score.
    assert elo.expected_home("X", "Y", neutral=True) == pytest.approx(0.5)


def test_elo_home_advantage_expected() -> None:
    elo = EloRatings(k=32, home_advantage=65, initial=1500)
    exp = elo.expected_home("X", "Y", neutral=False)
    assert exp == pytest.approx(1.0 / (1.0 + 10.0 ** (-65 / 400)))
    assert exp > 0.5  # home advantage tilts the expectation


def test_elo_update_is_zero_sum_and_correct() -> None:
    elo = EloRatings(k=32, home_advantage=65, initial=1500)
    # Neutral so expected = 0.5; home wins -> delta = 32 * (1 - 0.5) = 16.
    elo.update("X", "Y", 1, 0, neutral=True)
    assert elo.get("X") == pytest.approx(1516)
    assert elo.get("Y") == pytest.approx(1484)
    # Zero-sum: the pool mean is conserved.
    assert elo.get("X") + elo.get("Y") == pytest.approx(3000)


# --- Recent form ---------------------------------------------------------------


def test_form_no_history_is_none() -> None:
    form = RecentForm(window=5)
    out = form.pre_match("A", "B")
    assert out["form_home"] is None and out["gd_home"] is None


def test_form_tracks_points_and_goal_diff() -> None:
    form = RecentForm(window=5)
    form.update("A", "B", 2, 0)  # A: +3 pts, gd +2 ; B: 0 pts, gd -2
    form.update("A", "C", 1, 1)  # A: +1 pt, gd 0
    out = form.pre_match("A", "C")
    assert out["form_home"] == pytest.approx((3 + 1) / 2)
    assert out["gd_home"] == pytest.approx((2 + 0) / 2)


def test_form_window_evicts_old_matches() -> None:
    form = RecentForm(window=2)
    form.update("A", "B", 5, 0)  # gd +5 (should be evicted)
    form.update("A", "C", 1, 0)  # gd +1
    form.update("A", "D", 2, 0)  # gd +2
    out = form.pre_match("A", "Z")
    assert out["gd_home"] == pytest.approx((1 + 2) / 2)  # only the last two count


# --- Rest days -----------------------------------------------------------------


def test_rest_first_match_is_capped() -> None:
    import pandas as pd

    rest = RestTracker(cap=30)
    out = rest.pre_match("A", "B", pd.Timestamp("2020-01-01"))
    assert out["rest_home"] == 30 and out["rest_away"] == 30


def test_rest_counts_days_since_last_match() -> None:
    import pandas as pd

    rest = RestTracker(cap=30)
    rest.update("A", "B", pd.Timestamp("2020-01-01"))
    out = rest.pre_match("A", "C", pd.Timestamp("2020-01-10"))
    assert out["rest_home"] == 9  # A played 9 days ago
    assert out["rest_away"] == 30  # C is new -> capped


# --- Head to head --------------------------------------------------------------


def test_h2h_perspective_is_home_relative() -> None:
    h2h = HeadToHead(max_matches=10)
    h2h.update("A", "B", 2, 0)  # A beat B
    a_home = h2h.pre_match("A", "B")
    b_home = h2h.pre_match("B", "A")
    assert a_home["h2h_home_rate"] == pytest.approx(1.0)
    assert b_home["h2h_home_rate"] == pytest.approx(0.0)
    assert a_home["h2h_matches"] == 1 and b_home["h2h_matches"] == 1


def test_h2h_draw_counts_as_half() -> None:
    h2h = HeadToHead(max_matches=10)
    h2h.update("A", "B", 1, 1)  # draw
    out = h2h.pre_match("A", "B")
    assert out["h2h_home_rate"] == pytest.approx(0.5)


# --- build_features: point-in-time correctness ---------------------------------


def test_build_first_match_uses_only_priors(tiny_results) -> None:
    feat = build_features(validate_results(tiny_results))
    first = feat.iloc[0]
    # Earliest match: both teams unseen -> initial Elo, no edge, neutral h2h.
    assert first[schema.COL_ELO_HOME] == pytest.approx(1500)
    assert first[schema.COL_ELO_AWAY] == pytest.approx(1500)
    assert first[schema.COL_ELO_DIFF] == pytest.approx(0.0)
    assert first[schema.COL_H2H_MATCHES] == 0
    assert first[schema.COL_H2H_HOME_RATE] == pytest.approx(0.5)


def test_build_rematch_sees_exactly_prior_meeting(tiny_results) -> None:
    feat = build_features(validate_results(tiny_results))
    # The A-B rematch is the last row chronologically.
    rematch = feat[
        (feat[schema.COL_HOME_TEAM] == "A")
        & (feat[schema.COL_AWAY_TEAM] == "B")
        & (feat[schema.COL_DATE] == feat[schema.COL_DATE].max())
    ].iloc[0]
    # Exactly one prior A-B meeting (the opener), which A won 2-0.
    assert rematch[schema.COL_H2H_MATCHES] == 1
    assert rematch[schema.COL_H2H_HOME_RATE] == pytest.approx(1.0)
    # Rest days come from each team's previous fixture, not this one.
    assert rematch[schema.COL_REST_HOME] == 12  # A last played 2020-01-20
    assert rematch[schema.COL_REST_AWAY] == 22  # B last played 2020-01-10
    # A has out-rated B after winning their head-to-head -> positive Elo edge.
    assert rematch[schema.COL_ELO_DIFF] > 0


def test_build_outputs_expected_schema_without_nans(tiny_results) -> None:
    feat = build_features(validate_results(tiny_results))
    for col in schema.FEATURE_COLUMNS:
        assert col in feat.columns
        assert not feat[col].isna().any(), f"{col} has NaNs"
    # Targets are valid 1X2 labels.
    assert set(feat[schema.COL_TARGET]) <= {"H", "D", "A"}


def test_build_is_deterministic(tiny_results) -> None:
    a = build_features(validate_results(tiny_results))
    b = build_features(validate_results(tiny_results))
    assert a.equals(b)


def test_no_nan_or_inf_in_elo_diff(tiny_results) -> None:
    feat = build_features(validate_results(tiny_results))
    assert all(math.isfinite(v) for v in feat[schema.COL_ELO_DIFF])
