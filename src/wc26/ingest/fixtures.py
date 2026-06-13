# WC26 tournament fixtures.
#
# The group draw is committed reference data (``data/seed/wc26_groups.csv``); the
# group-stage match list is *generated* from it as a single round-robin per group
# (each of the 4 teams plays the other 3 -> 6 matches/group, 72 group matches).
# The knockout bracket structure is built (``wc26.simulate``).

from __future__ import annotations

import itertools
import logging

import pandas as pd

from wc26.config import settings

logger = logging.getLogger(__name__)

GROUP_COLUMNS = ("group", "team", "pot")


def load_wc26_groups() -> pd.DataFrame:
    # Load the 48-team / 12-group WC26 group stage from the committed seed.
    #
    # Validates the canonical shape: 12 groups of exactly 4 distinct teams.
    path = settings.paths.seed_dir / "wc26_groups.csv"
    df = pd.read_csv(path)

    missing = set(GROUP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"wc26_groups.csv missing columns: {sorted(missing)}")

    sizes = df.groupby("group")["team"].nunique()
    bad = sizes[sizes != 4]
    if len(sizes) != 12 or not bad.empty:
        raise ValueError(
            f"Expected 12 groups of 4 distinct teams; got {len(sizes)} groups, "
            f"irregular: {bad.to_dict()}"
        )
    if df["team"].duplicated().any():
        dupes = df.loc[df["team"].duplicated(), "team"].tolist()
        raise ValueError(f"Teams appear in more than one group: {dupes}")

    logger.info("Loaded WC26 groups: %d teams across %d groups", len(df), len(sizes))
    return df


def build_group_schedule(groups: pd.DataFrame | None = None) -> pd.DataFrame:
    # Generate the group-stage fixture list (round-robin within each group).
    #
    # Deterministic: matches are ordered by (group, pairing order). Returns columns
    # ``group, matchday, home_team, away_team``. Home/away here is nominal - WC26
    # group games are at neutral or host venues; the simulation treats them per the
    # model's neutral-venue handling.
    if groups is None:
        groups = load_wc26_groups()

    rows: list[dict[str, object]] = []
    for group, gdf in groups.groupby("group"):
        teams = list(gdf.sort_values("pot")["team"])
        # itertools.combinations gives each unordered pair exactly once.
        for matchday, (home, away) in enumerate(itertools.combinations(teams, 2), start=1):
            rows.append(
                {"group": group, "matchday": matchday, "home_team": home, "away_team": away}
            )

    schedule = pd.DataFrame(rows, columns=["group", "matchday", "home_team", "away_team"])
    logger.info("Generated %d group-stage fixtures", len(schedule))
    return schedule
