# Tests for drift (PSI) and the live prediction-vs-actual loop.

from __future__ import annotations

import numpy as np
import pandas as pd

from wc26 import schema
from wc26.config import settings
from wc26.monitor.drift import population_stability_index, psi_summary
from wc26.monitor.live import update_live_loop


def test_psi_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert population_stability_index(x, x.copy()) < 1e-6


def test_psi_flags_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=2000)
    cur = rng.normal(3, 1, size=2000)  # large mean shift
    assert population_stability_index(ref, cur) > 0.2


def test_psi_summary_shape(feature_frame) -> None:
    half = len(feature_frame) // 2
    summary = psi_summary(feature_frame.iloc[:half], feature_frame.iloc[half:])
    assert summary["n_features"] > 0
    assert 0.0 <= summary["share_drifted"] <= 1.0
    assert set(summary["psi"]).issubset(set(schema.FEATURE_COLUMNS))


def _wc26_match(home, away, hs, as_, mid=0):
    return {
        schema.COL_MATCH_ID: mid,
        schema.COL_DATE: pd.Timestamp("2026-06-15"),
        schema.COL_HOME_TEAM: home,
        schema.COL_AWAY_TEAM: away,
        schema.COL_HOME_SCORE: hs,
        schema.COL_AWAY_SCORE: as_,
        schema.COL_TOURNAMENT: "FIFA World Cup",
        schema.COL_NEUTRAL: True,
    }


def test_live_loop_aligns_and_reorients(tmp_path, monkeypatch) -> None:
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"
    processed.mkdir()
    interim.mkdir()
    monkeypatch.setattr(settings.paths, "processed_dir", processed)
    monkeypatch.setattr(settings.paths, "interim_dir", interim)
    monkeypatch.setattr(settings.paths, "snapshot_dir", tmp_path / "snapshot")
    # Force the offline martj42 path (don't hit the live football-data.org API).
    monkeypatch.setattr(settings, "football_data_token", "")

    # Frozen prediction: USA (home) favoured over Croatia.
    pd.DataFrame(
        [
            {
                "home_team": "United States",
                "away_team": "Croatia",
                "p_home": 0.5,
                "p_draw": 0.3,
                "p_away": 0.2,
            }
        ]
    ).to_parquet(processed / "group_predictions.parquet", index=False)

    # Two actual WC26 games for the same pair, in opposite home/away orientations.
    pd.DataFrame(
        [
            _wc26_match("United States", "Croatia", 2, 0, mid=1),  # USA home win
            _wc26_match("Croatia", "United States", 1, 1, mid=2),  # swapped orientation, draw
        ]
    ).to_parquet(interim / "results_clean.parquet", index=False)

    result = update_live_loop()
    assert result["n_matches"] == 2

    live = pd.read_parquet(processed / "live_predictions.parquet").set_index("match_id")
    # Game 1: same orientation -> p_home stays 0.5; outcome home.
    assert live.loc["1", "p_home"] == 0.5
    assert live.loc["1", "actual"] == "H"
    # Game 2: orientation swapped -> home prob becomes the prediction's away prob (0.2).
    assert live.loc["2", "p_home"] == 0.2
    assert live.loc["2", "p_away"] == 0.5
    assert live.loc["2", "actual"] == "D"


def test_live_loop_no_results_is_empty(tmp_path, monkeypatch) -> None:
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"
    processed.mkdir()
    interim.mkdir()
    monkeypatch.setattr(settings.paths, "processed_dir", processed)
    monkeypatch.setattr(settings.paths, "interim_dir", interim)
    monkeypatch.setattr(settings.paths, "snapshot_dir", tmp_path / "snapshot")
    # Force the offline martj42 path (don't hit the live football-data.org API).
    monkeypatch.setattr(settings, "football_data_token", "")

    pd.DataFrame(
        [
            {
                "home_team": "Spain",
                "away_team": "Germany",
                "p_home": 0.4,
                "p_draw": 0.3,
                "p_away": 0.3,
            }
        ]
    ).to_parquet(processed / "group_predictions.parquet", index=False)
    # Only a qualifier (not a finals match) -> excluded.
    pd.DataFrame(
        [
            {
                schema.COL_MATCH_ID: 1,
                schema.COL_DATE: pd.Timestamp("2026-03-01"),
                schema.COL_HOME_TEAM: "Spain",
                schema.COL_AWAY_TEAM: "Germany",
                schema.COL_HOME_SCORE: 1,
                schema.COL_AWAY_SCORE: 0,
                schema.COL_TOURNAMENT: "FIFA World Cup qualification",
                schema.COL_NEUTRAL: False,
            }
        ]
    ).to_parquet(interim / "results_clean.parquet", index=False)

    assert update_live_loop()["n_matches"] == 0
