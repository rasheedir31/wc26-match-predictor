# Elo rating engine.
#
# Reusable core shared by the Elo *feature* (pre-match ratings, this module) and the
# Elo *model* - implemented from scratch, no library.
#
# Elo update for a match::
#
#     E_home = 1 / (1 + 10 ** (-(R_home + H - R_away) / S))     # expected home score
#     R_home' = R_home + K * (A_home - E_home)                  # A_home in {1, 0.5, 0}
#     R_away' = R_away - K * (A_home - E_home)                  # zero-sum update
#
# where ``H`` is the home-advantage rating bump (applied only at non-neutral venues),
# ``S`` the logistic scale (400), ``K`` the update step, and ``A_home`` the actual
# score (win=1, draw=0.5, loss=0). Ratings are zero-sum per match so the pool mean
# is conserved.

from __future__ import annotations

from wc26.config import settings
from wc26.schema import result_to_outcome

ELO_SCALE = 400.0  # standard logistic scale: a 400-point edge => ~10:1 expected odds


class EloRatings:
    # Mutable Elo rating table, updated match by match in chronological order.

    def __init__(
        self,
        k: float | None = None,
        home_advantage: float | None = None,
        initial: float | None = None,
    ) -> None:
        self.k = settings.model.elo_k if k is None else k
        self.home_advantage = (
            settings.model.elo_home_advantage if home_advantage is None else home_advantage
        )
        self.initial = settings.model.elo_initial_rating if initial is None else initial
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        # Current rating for ``team`` (the initial rating if unseen).
        return self.ratings.get(team, self.initial)

    def expected_home(self, home: str, away: str, neutral: bool) -> float:
        # Expected score for the home team in [0, 1] (its win probability + half its
        # draw probability under the Elo model).
        bump = 0.0 if neutral else self.home_advantage
        diff = (self.get(home) + bump) - self.get(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / ELO_SCALE))

    def update(self, home: str, away: str, home_score: int, away_score: int, neutral: bool) -> None:
        # Apply one match result, mutating both teams' ratings (zero-sum).
        outcome = result_to_outcome(home_score, away_score)
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[outcome.value]
        expected = self.expected_home(home, away, neutral)
        delta = self.k * (actual_home - expected)
        self.ratings[home] = self.get(home) + delta
        self.ratings[away] = self.get(away) - delta
