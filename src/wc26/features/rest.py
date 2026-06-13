# Rest-days feature: days since each team's previous match.
#
# A team's first observed match has no prior, so rest is reported as the configured
# cap (treated as fully rested). Long gaps are likewise capped so an 18-month break
# doesn't dominate the feature.

from __future__ import annotations

import pandas as pd

from wc26.config import settings


class RestTracker:
    # Tracks each team's most recent match date to compute rest days.

    def __init__(self, cap: int | None = None) -> None:
        self.cap = settings.features.rest_days_cap if cap is None else cap
        self._last: dict[str, pd.Timestamp] = {}

    def pre_match(self, home: str, away: str, date: pd.Timestamp) -> dict[str, int]:
        # Rest days for both teams as of ``date`` (capped; cap if no prior match).
        return {
            "rest_home": self._rest(home, date),
            "rest_away": self._rest(away, date),
        }

    def _rest(self, team: str, date: pd.Timestamp) -> int:
        last = self._last.get(team)
        if last is None:
            return self.cap
        days = (date - last).days
        return min(max(days, 0), self.cap)

    def update(self, home: str, away: str, date: pd.Timestamp) -> None:
        # Record ``date`` as the latest match date for both teams.
        self._last[home] = date
        self._last[away] = date
