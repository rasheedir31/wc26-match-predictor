# Offline tests for the football-data.org parsing + name mapping (no network).

from __future__ import annotations

from wc26 import schema
from wc26.config import settings
from wc26.ingest.football_data import _NAME_MAP, fetch_wc26_results, parse_matches

_SAMPLE = {
    "matches": [
        {
            "id": 501,
            "utcDate": "2026-06-11T13:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"name": "Mexico"},
            "awayTeam": {"name": "South Africa"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        },
        {
            "id": 502,
            "utcDate": "2026-06-11T20:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"name": "Korea Republic"},
            "awayTeam": {"name": "Czechia"},
            "score": {"fullTime": {"home": 1, "away": 1}},
        },
        {  # in play -> must be skipped
            "id": 503,
            "utcDate": "2026-06-12T15:00:00Z",
            "status": "IN_PLAY",
            "homeTeam": {"name": "Canada"},
            "awayTeam": {"name": "Qatar"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
    ]
}


def test_parse_only_finished_matches() -> None:
    assert len(parse_matches(_SAMPLE)) == 2  # the IN_PLAY match is excluded


def test_parse_maps_names_to_martj42() -> None:
    df = parse_matches(_SAMPLE).set_index(schema.COL_MATCH_ID)
    assert df.loc["502", schema.COL_HOME_TEAM] == "South Korea"  # Korea Republic
    assert df.loc["502", schema.COL_AWAY_TEAM] == "Czech Republic"  # Czechia


def test_parse_computes_outcome() -> None:
    df = parse_matches(_SAMPLE).set_index(schema.COL_MATCH_ID)
    assert df.loc["501", "actual"] == "H"  # Mexico 2-1
    assert df.loc["502", "actual"] == "D"  # 1-1
    assert "actual" in df.columns


def test_name_map_targets_are_known_teams() -> None:
    from wc26.ingest.fixtures import load_wc26_groups

    known = set(load_wc26_groups()["team"])
    for target in _NAME_MAP.values():
        assert target in known, f"{target} is not a WC26 team"


def test_fetch_without_token_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(settings, "football_data_token", "")
    out = fetch_wc26_results()
    assert out.empty
    assert "actual" in out.columns  # shape preserved for the caller
