# WC26 tournament structure: group standings and the knockout bracket tree.
#
# 48 teams in 12 groups of 4. Top two of each group (24) plus the 8 best third-placed
# teams advance to a 32-team single-elimination knockout: Round of 32 -> Round of 16
# -> Quarter-finals -> Semi-finals -> Final.
#
# Simplifications (documented; see the README "representative seed"): group ties are
# broken at random (the 1X2 sampling yields no goals, so goal-difference tiebreaks are
# approximated), and the 32 qualifiers are seeded into a *standard* single-elimination
# bracket by tier (group winners, then runners-up, then thirds) rather than FIFA's
# official third-place placement table.

from __future__ import annotations

import numpy as np

# Knockout stages, in order, that a team can *reach*.
STAGE_R32 = "reach_r32"
STAGE_R16 = "reach_r16"
STAGE_QF = "reach_qf"
STAGE_SF = "reach_sf"
STAGE_FINAL = "reach_final"
STAGE_CHAMPION = "champion"
KNOCKOUT_STAGES = (STAGE_R32, STAGE_R16, STAGE_QF, STAGE_SF, STAGE_FINAL, STAGE_CHAMPION)

_POINTS_WIN = 3
_POINTS_DRAW = 1


def seeding_order(n: int) -> list[int]:
    # Standard single-elimination seeding order for ``n`` (a power of two) seeds.
    #
    # Returns 1-based seed numbers arranged so that, paired consecutively, the top
    # seeds are spread across the bracket (1 meets the weakest, and 1 & 2 can only
    # meet in the final). E.g. n=4 -> [1, 4, 2, 3]; n=8 -> [1, 8, 4, 5, 2, 7, 3, 6].
    order = [1]
    while len(order) < n:
        m = len(order) * 2
        order = [s if i % 2 == 0 else m + 1 - s for s in order for i in (0, 1)]
    return order


def rank_group(team_points: dict[int, int], rng: np.random.Generator) -> list[int]:
    # Rank a group's teams best-to-worst by points, ties broken at random.
    teams = list(team_points)
    jitter = {t: rng.random() for t in teams}
    return sorted(teams, key=lambda t: (team_points[t], jitter[t]), reverse=True)


def points_for(outcome_idx: int) -> tuple[int, int]:
    # (home_points, away_points) for a sampled 1X2 outcome index (0=H,1=D,2=A).
    if outcome_idx == 0:
        return _POINTS_WIN, 0
    if outcome_idx == 2:
        return 0, _POINTS_WIN
    return _POINTS_DRAW, _POINTS_DRAW
