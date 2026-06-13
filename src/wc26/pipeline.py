# Pipeline orchestration - plain importable task functions.
#
# These are the single source of truth for the ETL+features pipeline. Both the
# Airflow DAG (``dags/wc26_pipeline.py``) and ``make pipeline`` call exactly these
# functions; the GitHub Actions schedule will too. No logic is duplicated
# in the orchestrators - they are thin wrappers.
#
# Each stage reads its inputs from disk and writes its outputs to disk, so stages are
# idempotent and independently runnable (which is what lets Airflow chain them as
# separate tasks via the filesystem rather than passing big frames through XCom).
#
# Artifact layout::
#
#     data/interim/results_raw.parquet      (ingest)
#     data/interim/shootouts_raw.parquet    (ingest)
#     data/interim/results_clean.parquet    (validate)
#     data/processed/fifa_rankings.parquet  (ingest, reference)
#     data/processed/wc26_groups.parquet    (ingest, reference)
#     data/processed/wc26_schedule.parquet  (ingest, reference)
#     data/processed/features.parquet       (features)  <- model training input

from __future__ import annotations

import argparse
import logging

import pandas as pd

from wc26.config import settings
from wc26.etl.load import load_to_postgres, write_parquet
from wc26.etl.validate import validate_results
from wc26.features.build import build_features
from wc26.ingest.fixtures import build_group_schedule, load_wc26_groups
from wc26.ingest.sources import load_fifa_rankings, load_results, load_shootouts

logger = logging.getLogger(__name__)

# Artifact names (without extension).
RESULTS_RAW = "results_raw"
SHOOTOUTS_RAW = "shootouts_raw"
RESULTS_CLEAN = "results_clean"
FIFA_RANKINGS = "fifa_rankings"
WC26_GROUPS = "wc26_groups"
WC26_SCHEDULE = "wc26_schedule"
FEATURES = "features"


def _interim_path(name: str):
    return settings.paths.interim_dir / f"{name}.parquet"


def _processed_path(name: str):
    return settings.paths.processed_dir / f"{name}.parquet"


def run_ingest(*, force: bool = False) -> None:
    # Stage 1 - download/cache sources and persist raw + reference tables.
    settings.paths.ensure()

    load_results(force=force).to_parquet(_interim_path(RESULTS_RAW), index=False)
    load_shootouts(force=force).to_parquet(_interim_path(SHOOTOUTS_RAW), index=False)

    # Reference tables (not match features) go straight to processed/.
    load_fifa_rankings(force=force).to_parquet(_processed_path(FIFA_RANKINGS), index=False)
    groups = load_wc26_groups()
    groups.to_parquet(_processed_path(WC26_GROUPS), index=False)
    build_group_schedule(groups).to_parquet(_processed_path(WC26_SCHEDULE), index=False)

    logger.info("Ingest complete")


def run_validate() -> None:
    # Stage 2 - validate raw results into a clean, typed, sorted frame.
    raw = pd.read_parquet(_interim_path(RESULTS_RAW))
    clean = validate_results(raw)
    clean.to_parquet(_interim_path(RESULTS_CLEAN), index=False)
    logger.info("Validate complete")


def run_features() -> None:
    # Stage 3 - engineer the point-in-time feature matrix.
    clean = pd.read_parquet(_interim_path(RESULTS_CLEAN))
    feat = build_features(clean)
    write_parquet(feat, FEATURES)
    logger.info("Features complete")


def run_load() -> None:
    # Stage 4 - load features (+ reference tables) into Postgres (tolerant).
    feat = pd.read_parquet(_processed_path(FEATURES))
    load_to_postgres(feat, "match_features")
    for name, table in (
        (FIFA_RANKINGS, "fifa_rankings"),
        (WC26_GROUPS, "wc26_groups"),
        (WC26_SCHEDULE, "wc26_schedule"),
    ):
        load_to_postgres(pd.read_parquet(_processed_path(name)), table)
    logger.info("Load complete")


def run_pipeline(*, force: bool = False) -> None:
    # Run the full pipeline end to end: ETL + train + predict + snapshot.
    run_ingest(force=force)
    run_validate()
    run_features()
    run_load()
    _run_train()
    _run_simulate()
    _run_snapshot()
    _run_monitor()
    logger.info("Pipeline finished")


def _run_train() -> None:
    # Imported lazily so the ETL-only stages don't pull in the modelling stack.
    from wc26.train import run_train

    run_train()


def _run_simulate() -> None:
    from wc26.train import run_simulate

    run_simulate()


def _run_snapshot() -> None:
    from wc26.snapshot import build_snapshot

    build_snapshot()


def _run_monitor() -> None:
    from wc26.monitor.run import run_monitor

    run_monitor()


_STAGES = {
    "ingest": run_ingest,
    "validate": run_validate,
    "features": run_features,
    "load": run_load,
    "train": _run_train,
    "simulate": _run_simulate,
    "snapshot": _run_snapshot,
    "monitor": _run_monitor,
    "all": run_pipeline,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="WC26 ETL + feature pipeline")
    parser.add_argument(
        "stage", nargs="?", default="all", choices=sorted(_STAGES), help="stage to run"
    )
    parser.add_argument("--force", action="store_true", help="ignore cache and re-download sources")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    stage = args.stage
    if stage in ("ingest", "all"):
        _STAGES[stage](force=args.force)
    else:
        _STAGES[stage]()


if __name__ == "__main__":
    main()
