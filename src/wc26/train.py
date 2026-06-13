# Training + simulation orchestration.
#
# ``run_train``  - evaluate every model under time-based CV, log each to MLflow, pick
# the champion (lowest pooled log loss), refit the champion on all data, and persist
# the artifacts the app and simulation consume.
#
# ``run_simulate`` - load the champion + featurizer, run the Monte Carlo tournament,
# and write per-team odds and per-fixture group-stage predictions.
#
# Both are plain functions (like the ETL stages) so the Airflow DAG, ``make`` targets,
# and the GitHub Actions schedule all drive the same code.

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from wc26 import schema
from wc26.config import settings
from wc26.evaluate.runner import EvalResult, cross_validate
from wc26.features.featurizer import MatchupFeaturizer
from wc26.ingest.fixtures import build_group_schedule
from wc26.models.base import MatchPredictor
from wc26.models.registry import default_models
from wc26.tracking import mlflow_utils

logger = logging.getLogger(__name__)

# Artifact names under data/processed/.
FEATURES = "features"
RESULTS_CLEAN = "results_clean"
WC26_GROUPS = "wc26_groups"
CHAMPION_MODEL = "champion_model.joblib"
FEATURIZER = "featurizer.joblib"
MODEL_COMPARISON = "model_comparison"
CALIBRATION = "calibration.json"
CHAMPION_NAME = "champion_name.txt"
TOURNAMENT_ODDS = "tournament_odds"
GROUP_PREDICTIONS = "group_predictions"


def _processed(name: str):
    return settings.paths.processed_dir / name


def _comparison_frame(results: dict[str, EvalResult], champion: str) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "model": name,
                **{f"cv_{k}": v for k, v in r.mean_metrics.items()},
                **{f"oof_{k}": v for k, v in r.pooled_metrics.items()},
                "is_champion": name == champion,
            }
        )
    return pd.DataFrame(rows).sort_values("oof_log_loss").reset_index(drop=True)


def run_train() -> None:
    # Evaluate all models, log to MLflow, select + persist the champion.
    settings.paths.ensure()
    feat = pd.read_parquet(_processed(f"{FEATURES}.parquet"))
    results_clean = pd.read_parquet(settings.paths.interim_dir / f"{RESULTS_CLEAN}.parquet")

    mlflow_utils.init_mlflow()
    results: dict[str, EvalResult] = {}
    fitted: dict[str, MatchPredictor] = {}
    run_ids: dict[str, str | None] = {}

    for name, factory in default_models().items():
        logger.info("Evaluating model: %s", name)
        result = cross_validate(name, factory, feat)
        model = factory().fit(feat)  # refit on all data for the artifact / champion
        run_ids[name] = mlflow_utils.log_evaluation(result, params={}, model=model)
        results[name] = result
        fitted[name] = model
        logger.info(
            "  %s pooled: %s", name, {k: round(v, 4) for k, v in result.pooled_metrics.items()}
        )

    best = mlflow_utils.select_best(results.values())
    mlflow_utils.mark_champion(run_ids[best.name])
    logger.info("Champion: %s (pooled log loss %.4f)", best.name, best.pooled_metrics["log_loss"])

    # Persist artifacts for the app + simulation.
    featurizer = MatchupFeaturizer.fitted_on(results_clean)
    # Every fitted model is saved so the dashboard can switch between them; the
    # champion is also saved under a stable name (the API's default).
    for name, model in fitted.items():
        joblib.dump(model, _processed(f"model_{name}.joblib"))
    joblib.dump(fitted[best.name], _processed(CHAMPION_MODEL))
    joblib.dump(featurizer, _processed(FEATURIZER))
    _comparison_frame(results, best.name).to_parquet(
        _processed(f"{MODEL_COMPARISON}.parquet"), index=False
    )
    _processed(CALIBRATION).write_text(json.dumps(results[best.name].calibration, indent=2))
    _processed(CHAMPION_NAME).write_text(best.name)
    logger.info("Train complete")


def run_simulate() -> None:
    # Run the Monte Carlo tournament for *every* model; persist per-model odds +
    # group-stage predictions (a ``model`` column distinguishes them).
    from wc26.simulate.montecarlo import TournamentSimulator

    featurizer: MatchupFeaturizer = joblib.load(_processed(FEATURIZER))
    groups = pd.read_parquet(_processed(f"{WC26_GROUPS}.parquet"))
    schedule = build_group_schedule(groups)
    names = list(pd.read_parquet(_processed(f"{MODEL_COMPARISON}.parquet"))["model"])

    odds_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    for name in names:
        model: MatchPredictor = joblib.load(_processed(f"model_{name}.joblib"))
        odds = TournamentSimulator.from_model(model, featurizer, groups).run()
        odds["model"] = name
        odds_frames.append(odds)
        preds = _predict_fixtures(model, featurizer, schedule)
        preds["model"] = name
        pred_frames.append(preds)
        logger.info("Simulated tournament for model '%s'", name)

    pd.concat(odds_frames, ignore_index=True).to_parquet(
        _processed(f"{TOURNAMENT_ODDS}.parquet"), index=False
    )
    pd.concat(pred_frames, ignore_index=True).to_parquet(
        _processed(f"{GROUP_PREDICTIONS}.parquet"), index=False
    )
    logger.info("Simulate complete (%d models)", len(names))


def _predict_fixtures(
    model: MatchPredictor, featurizer: MatchupFeaturizer, schedule: pd.DataFrame
) -> pd.DataFrame:
    from wc26.features.featurizer import add_diffs, fill_defaults

    now = pd.Timestamp.utcnow().tz_localize(None)
    rows = []
    for r in schedule.itertuples(index=False):
        raw = featurizer.raw_features(
            r.home_team, r.away_team, neutral=True, date=now, tournament="FIFA World Cup"
        )
        raw |= {
            schema.COL_HOME_TEAM: r.home_team,
            schema.COL_AWAY_TEAM: r.away_team,
            schema.COL_DATE: now,
            schema.COL_TOURNAMENT: "FIFA World Cup",
        }
        rows.append(raw)
    proba = model.predict_proba(add_diffs(fill_defaults(pd.DataFrame(rows))))
    out = schedule.copy().reset_index(drop=True)
    out[list(schema.PROBA_COLUMNS)] = proba
    return out
