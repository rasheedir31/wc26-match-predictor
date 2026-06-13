# End-to-end pipeline test, hermetic and offline.
#
# Redirects all writable data dirs to a temp folder and forces the ingest layer onto
# the committed seeds (empty source URLs), so the test exercises the real
# ingest -> validate -> features -> load wiring without any network. The Postgres load
# is expected to be skipped (no DB) and tolerated.

from __future__ import annotations

import pandas as pd

from wc26 import schema
from wc26.config import settings
from wc26.pipeline import run_pipeline


def test_pipeline_runs_end_to_end_offline(tmp_path, monkeypatch) -> None:
    # Redirect outputs to tmp; keep seed_dir on the committed seeds.
    monkeypatch.setattr(settings.paths, "data_dir", tmp_path)
    monkeypatch.setattr(settings.paths, "raw_dir", tmp_path / "raw")
    monkeypatch.setattr(settings.paths, "interim_dir", tmp_path / "interim")
    monkeypatch.setattr(settings.paths, "processed_dir", tmp_path / "processed")
    # Also redirect snapshot_dir - the full pipeline now builds the snapshot + runs
    # the monitor, which must NOT touch the committed data/snapshot.
    monkeypatch.setattr(settings.paths, "snapshot_dir", tmp_path / "snapshot")
    # Force the seed fallback (no download attempt).
    monkeypatch.setattr(settings.sources, "results_url", "")
    monkeypatch.setattr(settings.sources, "shootouts_url", "")
    monkeypatch.setattr(settings.sources, "fifa_rankings_url", "")
    # Keep the full ETL+train+simulate run fast and self-contained.
    monkeypatch.setattr(settings.model, "simulation_runs", 50)
    monkeypatch.setattr(
        settings, "mlflow_tracking_uri", "file:///" + str(tmp_path / "mlruns").replace("\\", "/")
    )

    run_pipeline()

    feat = pd.read_parquet(tmp_path / "processed" / "features.parquet")
    assert len(feat) > 0
    for col in schema.FEATURE_COLUMNS:
        assert col in feat.columns
    assert not feat[list(schema.FEATURE_COLUMNS)].isna().any().any()

    # Reference tables were materialised too.
    assert (tmp_path / "processed" / "wc26_groups.parquet").exists()
    assert (tmp_path / "processed" / "wc26_schedule.parquet").exists()

    # Train + simulate artifacts exist; odds are per-model and each is a valid distribution.
    assert (tmp_path / "processed" / "champion_model.joblib").exists()
    assert (tmp_path / "processed" / "model_comparison.parquet").exists()
    odds = pd.read_parquet(tmp_path / "processed" / "tournament_odds.parquet")
    assert "model" in odds.columns
    for _, g in odds.groupby("model"):
        assert len(g) == 48
        assert abs(g["champion"].sum() - 1.0) < 1e-9  # exactly one champion per run, per model
    # One serialized model artifact per evaluated model.
    assert len(list((tmp_path / "processed").glob("model_*.joblib"))) == odds["model"].nunique()
