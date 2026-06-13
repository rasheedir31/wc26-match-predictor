# Recent-form features: rolling points-per-game and rolling goal difference.
#
# Both are computed over each team's most recent ``form_window`` matches *before* the
# current one. Maintained as a per-team fixed-length deque updated in chronological
# order, so the value read at a match never includes that match or any later one.

from __future__ import annotations

from collections import defaultdict, deque
from functools import partial

from wc26.config import settings
from wc26.schema import Outcome, result_to_outcome


class RecentForm:
    # Per-team rolling history of (points, goal_difference) over recent matches.

    def __init__(self, window: int | None = None) -> None:
        self.window = settings.features.form_window if window is None else window
        self._points_win = settings.features.points_win
        self._points_draw = settings.features.points_draw
        self._points_loss = settings.features.points_loss
        # partial (not a lambda) so the featurizer remains picklable for the snapshot.
        self._hist: dict[str, deque[tuple[float, int]]] = defaultdict(
            partial(deque, maxlen=self.window)
        )

    def _avg(self, team: str) -> tuple[float | None, float | None]:
        # Average (points-per-game, goal-difference) over the window, or (None, None)
        # if the team has no prior matches yet.
        h = self._hist.get(team)
        if not h:
            return None, None
        n = len(h)
        avg_points = sum(p for p, _ in h) / n
        avg_gd = sum(gd for _, gd in h) / n
        return avg_points, avg_gd

    def pre_match(self, home: str, away: str) -> dict[str, float | None]:
        # Pre-match form values for both teams (None where no history exists).
        hp, hgd = self._avg(home)
        ap, agd = self._avg(away)
        return {"form_home": hp, "form_away": ap, "gd_home": hgd, "gd_away": agd}

    def _points(self, outcome: Outcome, *, is_home: bool) -> float:
        home_won = outcome is Outcome.HOME
        if outcome is Outcome.DRAW:
            return self._points_draw
        won = home_won if is_home else not home_won
        return self._points_win if won else self._points_loss

    def update(self, home: str, away: str, home_score: int, away_score: int) -> None:
        # Append this match's (points, goal_diff) to both teams' histories.
        outcome = result_to_outcome(home_score, away_score)
        gd = home_score - away_score
        self._hist[home].append((self._points(outcome, is_home=True), gd))
        self._hist[away].append((self._points(outcome, is_home=False), -gd))
