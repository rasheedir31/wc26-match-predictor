# Monte Carlo tournament simulation.
#
# Markov framing. Each team's tournament life is a walk through an absorbing Markov
# chain whose transient states are the rounds it is still alive in (R32, R16, QF, SF,
# Final) and whose absorbing states are "eliminated in round r" and "champion". Each
# knockout tie is one stochastic transition; losing is absorption. We don't form the
# transition matrix in closed form because the per-match win probabilities depend on
# the (random) identities of who advanced from the groups - instead we sample the
# chain by Monte Carlo: play the whole tournament ``n_runs`` times and average. The
# fraction of runs in which a team reaches each state estimates its absorption/visit
# probabilities (per-stage and championship odds).
#
# Per-match outcomes come from the chosen 1X2 model. Knockouts admit no draw, so a
# drawn regulation result is resolved (extra-time/penalties) by splitting the draw
# probability toward the stronger side - ``knockout_strength_bias`` controls how much.

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from wc26 import schema
from wc26.config import settings
from wc26.features.featurizer import MatchupFeaturizer, add_diffs, fill_defaults
from wc26.models.base import MatchPredictor
from wc26.simulate import bracket

logger = logging.getLogger(__name__)
_EPS = 1e-12


def build_probability_table(
    model: MatchPredictor, featurizer: MatchupFeaturizer, teams: list[str]
) -> np.ndarray:
    # Precompute P(home, draw, away) for every ordered team pair (neutral venue).
    #
    # Returns an array ``P`` of shape (T, T, 3); ``P[i, j]`` is team i (home) vs team j
    # (away). Computed once so each of the many simulated matches is a cheap lookup.
    rows: list[dict[str, object]] = []
    pairs: list[tuple[int, int]] = []
    now = pd.Timestamp.utcnow().tz_localize(None)
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            raw = featurizer.raw_features(
                home, away, neutral=True, date=now, tournament="FIFA World Cup"
            )
            raw |= {
                schema.COL_HOME_TEAM: home,
                schema.COL_AWAY_TEAM: away,
                schema.COL_DATE: now,
                schema.COL_TOURNAMENT: "FIFA World Cup",
            }
            rows.append(raw)
            pairs.append((i, j))

    proba = model.predict_proba(add_diffs(fill_defaults(pd.DataFrame(rows))))
    table = np.zeros((len(teams), len(teams), 3))
    for (i, j), p in zip(pairs, proba, strict=True):
        table[i, j] = p
    return table


class TournamentSimulator:
    # Simulates the WC26 bracket many times to estimate per-team stage odds.

    def __init__(
        self,
        teams: list[str],
        team_group: list[str],
        groups: dict[str, list[int]],
        prob: np.ndarray,
        knockout_bias: float | None = None,
    ) -> None:
        self.teams = teams
        self.team_group = team_group
        self.groups = groups
        self.prob = prob
        self.knockout_bias = (
            settings.model.knockout_strength_bias if knockout_bias is None else knockout_bias
        )

    @classmethod
    def from_model(
        cls,
        model: MatchPredictor,
        featurizer: MatchupFeaturizer,
        groups_df: pd.DataFrame,
        knockout_bias: float | None = None,
    ) -> TournamentSimulator:
        # Build a simulator from a fitted model + featurizer and the group draw.
        teams = list(groups_df["team"])
        idx = {t: i for i, t in enumerate(teams)}
        groups: dict[str, list[int]] = {}
        team_group = [""] * len(teams)
        for g, gdf in groups_df.groupby("group"):
            members = [idx[t] for t in gdf["team"]]
            groups[str(g)] = members
            for m in members:
                team_group[m] = str(g)
        prob = build_probability_table(model, featurizer, teams)
        return cls(teams, team_group, groups, prob, knockout_bias)

    # --- single tournament -----------------------------------------------------

    def _resolve_knockout(self, a: int, b: int, rng: np.random.Generator) -> int:
        # Sample the winner of a knockout tie (no draws); draw mass tilts to the
        # stronger side per ``knockout_bias``.
        p_a, p_d, p_b = self.prob[a, b]
        edge = (p_a - p_b) / (p_a + p_b + _EPS)
        share_a = float(np.clip(0.5 + 0.5 * self.knockout_bias * edge, 0.0, 1.0))
        p_a_adv = p_a + p_d * share_a
        return a if rng.random() < p_a_adv else b

    def _simulate_group_stage(
        self, rng: np.random.Generator
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        # Return (group winners, runners-up, third-place (team, points)) lists.
        winners: list[int] = []
        runners: list[int] = []
        thirds: list[tuple[int, int]] = []
        for g in sorted(self.groups):
            members = self.groups[g]
            points = dict.fromkeys(members, 0)
            for a, b in combinations(members, 2):
                outcome = rng.choice(3, p=self.prob[a, b])
                hp, ap = bracket.points_for(int(outcome))
                points[a] += hp
                points[b] += ap
            ranked = bracket.rank_group(points, rng)
            winners.append(ranked[0])
            runners.append(ranked[1])
            thirds.append((ranked[2], points[ranked[2]]))
        return winners, runners, thirds

    def _qualifiers(self, rng: np.random.Generator) -> list[int]:
        # Seeded list of the 32 knockout qualifiers (winners, runners, best 8 thirds).
        winners, runners, thirds = self._simulate_group_stage(rng)
        # Best 8 third-placed teams by points, ties broken at random.
        thirds_sorted = sorted(thirds, key=lambda tp: (tp[1], rng.random()), reverse=True)
        best_thirds = [t for t, _ in thirds_sorted[:8]]
        return winners + runners + best_thirds  # 12 + 12 + 8 = 32, tier-ordered = seeds

    def simulate_once(self, rng: np.random.Generator, counts: np.ndarray) -> None:
        # Play one tournament and add each team's reached stages into ``counts``.
        seeds = self._qualifiers(rng)  # index = seed-1
        order = bracket.seeding_order(32)
        stage_cols = {s: i for i, s in enumerate(bracket.KNOCKOUT_STAGES)}

        # Round of 32 line-up via standard seeding; everyone here "reaches R32".
        alive = [seeds[s - 1] for s in order]
        for t in alive:
            counts[t, stage_cols[bracket.STAGE_R32]] += 1

        # Reduce the bracket one round at a time; winners reach the next stage.
        next_stages = [
            bracket.STAGE_R16,
            bracket.STAGE_QF,
            bracket.STAGE_SF,
            bracket.STAGE_FINAL,
            bracket.STAGE_CHAMPION,
        ]
        for stage in next_stages:
            winners = [
                self._resolve_knockout(alive[k], alive[k + 1], rng) for k in range(0, len(alive), 2)
            ]
            for t in winners:
                counts[t, stage_cols[stage]] += 1
            alive = winners

    # --- many tournaments ------------------------------------------------------

    def run(self, n_runs: int | None = None, seed: int | None = None) -> pd.DataFrame:
        # Run ``n_runs`` simulations; return per-team stage and championship odds.
        n_runs = settings.model.simulation_runs if n_runs is None else n_runs
        seed = settings.model.random_seed if seed is None else seed
        rng = np.random.default_rng(seed)

        counts = np.zeros((len(self.teams), len(bracket.KNOCKOUT_STAGES)))
        for _ in range(n_runs):
            self.simulate_once(rng, counts)

        probs = counts / n_runs
        out = pd.DataFrame(probs, columns=list(bracket.KNOCKOUT_STAGES))
        out.insert(0, "team", self.teams)
        out.insert(1, "group", self.team_group)
        out = out.sort_values(bracket.STAGE_CHAMPION, ascending=False).reset_index(drop=True)
        logger.info("Simulated %d tournaments; champion favourite: %s", n_runs, out.iloc[0]["team"])
        return out
