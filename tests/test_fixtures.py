# Tests for WC26 group ingest and schedule generation.

from __future__ import annotations

from itertools import combinations

from wc26.ingest.fixtures import build_group_schedule, load_wc26_groups


def test_groups_are_twelve_of_four_distinct() -> None:
    groups = load_wc26_groups()
    assert len(groups) == 48
    sizes = groups.groupby("group")["team"].nunique()
    assert len(sizes) == 12
    assert (sizes == 4).all()
    assert groups["team"].nunique() == 48  # no team in two groups


def test_schedule_is_single_round_robin_per_group() -> None:
    groups = load_wc26_groups()
    schedule = build_group_schedule(groups)
    # 12 groups * C(4,2)=6 pairings = 72 fixtures.
    assert len(schedule) == 72
    assert (schedule.groupby("group").size() == 6).all()
    # No team faces itself; every within-group pair appears exactly once.
    assert (schedule["home_team"] != schedule["away_team"]).all()
    for group, gdf in schedule.groupby("group"):
        pairs = {frozenset((r.home_team, r.away_team)) for r in gdf.itertuples()}
        teams = list(groups[groups["group"] == group]["team"])
        assert pairs == {frozenset(p) for p in combinations(teams, 2)}


def test_schedule_is_deterministic() -> None:
    assert build_group_schedule().equals(build_group_schedule())
