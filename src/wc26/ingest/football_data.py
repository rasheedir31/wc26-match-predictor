# football-data.org - optional live WC26 results source for the monitor.
#
# Fetches the *finished* 2026 World Cup matches from football-data.org v4 and returns
# them in the project's match schema, with team names mapped to the martj42 spelling
# the models use. The monitor's live loop uses this when ``FOOTBALL_DATA_TOKEN`` is
# set in ``.env``; otherwise it falls back to the martj42 dataset.
#
# Auth is the ``X-Auth-Token`` header. The World Cup (competition code ``WC``) is in
# football-data.org's free tier. Docs: https://www.football-data.org/documentation/quickstart

from __future__ import annotations

import logging

import pandas as pd
import requests

from wc26 import schema
from wc26.config import settings

logger = logging.getLogger(__name__)

_FINISHED = "FINISHED"  # football-data.org match status for a completed game

_COLUMNS = [
    schema.COL_MATCH_ID,
    schema.COL_DATE,
    schema.COL_HOME_TEAM,
    schema.COL_AWAY_TEAM,
    schema.COL_HOME_SCORE,
    schema.COL_AWAY_SCORE,
    schema.COL_TOURNAMENT,
    schema.COL_NEUTRAL,
    "actual",
]

# football-data.org team name -> martj42 spelling (only where they differ). Extend
# as needed; unmapped names pass through and are logged if they aren't WC26 teams.
_NAME_MAP = {
    "United States of America": "United States",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "IR Iran": "Iran",
    "Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
}


def _normalize_team(name: str) -> str:
    return _NAME_MAP.get(name, name)


def parse_matches(payload: dict) -> pd.DataFrame:
    # Parse a football-data.org ``/matches`` response into finished WC26 matches.
    #
    # Pure function (no network) so parsing + name mapping is unit-testable. Returns
    # the match-schema columns plus a computed ``actual`` 1X2 outcome.
    rows: list[dict[str, object]] = []
    for m in payload.get("matches", []):
        if m.get("status") != _FINISHED:
            continue
        full_time = (m.get("score") or {}).get("fullTime") or {}
        hs, as_ = full_time.get("home"), full_time.get("away")
        if hs is None or as_ is None:
            continue
        home = _normalize_team((m.get("homeTeam") or {}).get("name", ""))
        away = _normalize_team((m.get("awayTeam") or {}).get("name", ""))
        date = m.get("utcDate")
        rows.append(
            {
                schema.COL_MATCH_ID: str(m.get("id", "")),
                schema.COL_DATE: (
                    pd.to_datetime(date, utc=True).tz_localize(None) if date else pd.NaT
                ),
                schema.COL_HOME_TEAM: home,
                schema.COL_AWAY_TEAM: away,
                schema.COL_HOME_SCORE: int(hs),
                schema.COL_AWAY_SCORE: int(as_),
                schema.COL_TOURNAMENT: "FIFA World Cup",
                schema.COL_NEUTRAL: True,
                "actual": schema.result_to_outcome(int(hs), int(as_)).value,
            }
        )
    return pd.DataFrame(rows, columns=_COLUMNS)


def fetch_wc26_results(*, timeout: int = 30) -> pd.DataFrame:
    # Fetch finished 2026 World Cup matches from football-data.org.
    #
    # Returns an empty frame (never raises) if the token is unset or the request
    # fails, so the caller can fall back to the martj42 dataset.
    if not settings.football_data_token:
        logger.info("football-data.org token not set; skipping API source")
        return pd.DataFrame(columns=_COLUMNS)

    url = f"{settings.football_data_base_url}/competitions/{settings.football_data_competition}/matches"
    try:
        resp = requests.get(
            url,
            params={"status": _FINISHED},
            headers={"X-Auth-Token": settings.football_data_token},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("football-data.org fetch failed (%s); falling back", exc)
        return pd.DataFrame(columns=_COLUMNS)

    df = parse_matches(payload)
    logger.info("football-data.org: %d finished WC26 matches", len(df))
    return df
