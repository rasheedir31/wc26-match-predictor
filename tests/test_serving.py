# Serving tests: snapshot build/load round-trip and the FastAPI endpoints.

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from wc26.config import settings
from wc26.features.featurizer import MatchupFeaturizer
from wc26.ingest.fixtures import build_group_schedule, load_wc26_groups
from wc26.models.elo import EloModel
from wc26.simulate.montecarlo import TournamentSimulator
from wc26.snapshot import SnapshotStore, build_snapshot, get_store
from wc26.train import _predict_fixtures


@pytest.fixture
def built_snapshot(tmp_path, monkeypatch, feature_frame, synthetic_clean):
    # Create a minimal processed dir, build the snapshot, return its directory.
    processed = tmp_path / "processed"
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(settings.paths, "processed_dir", processed)
    monkeypatch.setattr(settings.paths, "snapshot_dir", snapshot)
    processed.mkdir(parents=True)

    model = EloModel().fit(feature_frame)
    featurizer = MatchupFeaturizer.fitted_on(synthetic_clean)
    joblib.dump(model, processed / "champion_model.joblib")
    joblib.dump(featurizer, processed / "featurizer.joblib")
    (processed / "champion_name.txt").write_text("elo")
    (processed / "calibration.json").write_text(json.dumps({"mean_predicted": [], "empirical": []}))

    groups = load_wc26_groups()
    schedule = build_group_schedule(groups)
    groups.to_parquet(processed / "wc26_groups.parquet", index=False)
    schedule.to_parquet(processed / "wc26_schedule.parquet", index=False)
    pd.DataFrame(
        [
            {
                "model": "elo",
                "oof_log_loss": 0.9,
                "oof_brier": 0.5,
                "oof_rps": 0.18,
                "oof_accuracy": 0.5,
                "is_champion": True,
            }
        ]
    ).to_parquet(processed / "model_comparison.parquet", index=False)
    pd.DataFrame([{"rank": 1, "team": "Argentina", "points": 1880.0}]).to_parquet(
        processed / "fifa_rankings.parquet", index=False
    )

    sim = TournamentSimulator.from_model(model, featurizer, groups)
    sim.run(n_runs=20, seed=1).to_parquet(processed / "tournament_odds.parquet", index=False)
    _predict_fixtures(model, featurizer, schedule).to_parquet(
        processed / "group_predictions.parquet", index=False
    )

    build_snapshot()
    return snapshot


def test_snapshot_store_roundtrip(built_snapshot) -> None:
    store = SnapshotStore(built_snapshot)
    assert store.champion_name == "elo"
    assert len(store.teams()) == 48
    assert len(store.tournament_odds()) == 48

    pred = store.predict("Argentina", "Brazil", neutral=True)
    assert abs(pred["p_home"] + pred["p_draw"] + pred["p_away"] - 1.0) < 1e-9
    assert pred["most_likely"] in {"home", "draw", "away"}

    # No WC26 results scored yet.
    assert store.live_summary()["n_matches"] == 0


@pytest.fixture
def client(built_snapshot, monkeypatch):
    from wc26.api.app import app

    monkeypatch.setattr(settings.paths, "snapshot_dir", built_snapshot)
    get_store.cache_clear()
    with TestClient(app) as c:
        yield c
    get_store.cache_clear()


def test_health(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_teams_and_models(client) -> None:
    assert len(client.get("/teams").json()["teams"]) == 48
    body = client.get("/models").json()
    assert body["champion"] == "elo"
    assert len(body["comparison"]) >= 1


def test_predict_endpoint(client) -> None:
    resp = client.post("/predict", json={"home_team": "Spain", "away_team": "Germany"})
    assert resp.status_code == 200
    body = resp.json()
    assert abs(body["p_home"] + body["p_draw"] + body["p_away"] - 1.0) < 1e-9
    # Same team rejected.
    assert (
        client.post("/predict", json={"home_team": "Spain", "away_team": "Spain"}).status_code
        == 422
    )


def test_odds_endpoints(client) -> None:
    assert len(client.get("/odds?top=5").json()["odds"]) == 5
    assert client.get("/odds/Argentina").status_code == 200
    assert client.get("/odds/Nowhereland").status_code == 404


def test_live_endpoint(client) -> None:
    assert client.get("/live").json()["n_matches"] == 0
