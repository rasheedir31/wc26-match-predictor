# Head-to-head feature: the home team's historical record vs this opponent.
#
# For each unordered pair we keep the most recent ``h2h_max_matches`` meetings. The
# feature is a points-style win rate from the *current home team's* perspective -
# ``(wins + 0.5 * draws) / meetings`` - plus the number of meetings counted (so a
# model can discount a rate backed by few games).

from __future__ import annotations

from collections import defaultdict, deque
from functools import partial

from wc26.config import settings
from wc26.schema import result_to_outcome


def _pair_key(a: str, b: str) -> tuple[str, str]:
    # Canonical (order-independent) key for a pair of teams.
    return (a, b) if a <= b else (b, a)


class HeadToHead:
    # Per-pair rolling history of recent meeting winners.

    def __init__(self, max_matches: int | None = None) -> None:
        self.max_matches = settings.features.h2h_max_matches if max_matches is None else max_matches
        # Stores the winning team name, or None for a draw, most-recent-last.
        # partial (not a lambda) so the featurizer remains picklable for the snapshot.
        self._hist: dict[tuple[str, str], deque[str | None]] = defaultdict(
            partial(deque, maxlen=self.max_matches)
        )

    def pre_match(self, home: str, away: str) -> dict[str, float | None]:
        # Home win-rate and meeting count over recent prior meetings.
        #
        # Returns ``h2h_home_rate=None`` when the pair has never met.
        h = self._hist.get(_pair_key(home, away))
        if not h:
            return {"h2h_home_rate": None, "h2h_matches": 0}
        wins = sum(1 for w in h if w == home)
        draws = sum(1 for w in h if w is None)
        rate = (wins + 0.5 * draws) / len(h)
        return {"h2h_home_rate": rate, "h2h_matches": len(h)}

    def update(self, home: str, away: str, home_score: int, away_score: int) -> None:
        # Append this meeting's winner (or None for a draw) to the pair history.
        outcome = result_to_outcome(home_score, away_score)
        winner: str | None
        if outcome.value == "H":
            winner = home
        elif outcome.value == "A":
            winner = away
        else:
            winner = None
        self._hist[_pair_key(home, away)].append(winner)
