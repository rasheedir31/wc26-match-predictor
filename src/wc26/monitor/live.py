# Live prediction-vs-actual loop.
#
# As WC26 results arrive (re-ingested into the match data during the tournament), we
# match each played tournament game to our **frozen pre-tournament** group-stage
# prediction, record (prediction, actual) into the snapshot's ``live_predictions``
# table, and compute the running log loss / accuracy. This is the headline portfolio
# visual - a bracket forecast tracked honestly against reality.
#
# Until the tournament kicks off there are no WC26 results, so the log is empty and
# the dashboard shows "-"; the machinery simply populates as games are played.

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from wc26 import schema
from wc26.config import settings
from wc26.snapshot import DB_NAME

logger = logging.getLogger(__name__)

_LIVE_TABLE = "live_predictions"
_PROBA = list(schema.PROBA_COLUMNS)


def _wc26_actuals(results: pd.DataFrame) -> pd.DataFrame:
    # Played WC26 finals matches (not qualifiers) with their 1X2 outcome.
    tour = results[schema.COL_TOURNAMENT].astype(str).str.lower()
    is_wc_final = tour.str.contains("world cup") & ~tour.str.contains("qualif")
    is_2026 = results[schema.COL_DATE].dt.year == 2026
    out = results[is_wc_final & is_2026].copy()
    if out.empty:
        return out
    out["actual"] = [
        schema.result_to_outcome(h, a).value
        for h, a in zip(out[schema.COL_HOME_SCORE], out[schema.COL_AWAY_SCORE], strict=True)
    ]
    return out


def _align_to_prediction(actuals: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    # Join actuals to frozen predictions by unordered team pair, reorienting the
    # home/away probabilities to the actual match's orientation.
    pred_by_pair: dict[frozenset[str], dict[str, object]] = {}
    for r in preds.itertuples(index=False):
        pred_by_pair[frozenset((r.home_team, r.away_team))] = {
            "home_team": r.home_team,
            "p_home": r.p_home,
            "p_draw": r.p_draw,
            "p_away": r.p_away,
        }

    rows = []
    for r in actuals.itertuples(index=False):
        home = getattr(r, schema.COL_HOME_TEAM)
        away = getattr(r, schema.COL_AWAY_TEAM)
        pred = pred_by_pair.get(frozenset((home, away)))
        if pred is None:
            continue  # no frozen pre-tournament prediction (e.g. a knockout pairing)
        # Reorient: if the actual home team was the prediction's away team, swap.
        if pred["home_team"] == home:
            p_home, p_draw, p_away = pred["p_home"], pred["p_draw"], pred["p_away"]
        else:
            p_home, p_draw, p_away = pred["p_away"], pred["p_draw"], pred["p_home"]
        rows.append(
            {
                "match_id": str(getattr(r, schema.COL_MATCH_ID, "")),
                "date": str(getattr(r, schema.COL_DATE)),
                "home_team": home,
                "away_team": away,
                "home_score": int(getattr(r, schema.COL_HOME_SCORE)),
                "away_score": int(getattr(r, schema.COL_AWAY_SCORE)),
                "p_home": float(p_home),
                "p_draw": float(p_draw),
                "p_away": float(p_away),
                "actual": r.actual,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            *_PROBA,
            "actual",
        ],
    )


def _load_actuals() -> pd.DataFrame:
    # WC26 finished matches from API-Football when configured, else from martj42.
    #
    # Both sources return the same columns (incl. a computed ``actual`` 1X2 outcome),
    # so the alignment step is source-agnostic.
    if settings.football_data_token:
        from wc26.ingest.football_data import fetch_wc26_results

        api = fetch_wc26_results()
        if not api.empty:
            logger.info("Live loop: using football-data.org (%d matches)", len(api))
            return api
        logger.info("Live loop: football-data.org returned nothing; falling back to martj42")

    results_path = settings.paths.interim_dir / "results_clean.parquet"
    if results_path.exists():
        return _wc26_actuals(pd.read_parquet(results_path))
    return pd.DataFrame()


def update_live_loop() -> dict[str, float | int]:
    # Refresh the live_predictions log from ingested results; return running metrics.
    processed = settings.paths.processed_dir
    preds_path = processed / "group_predictions.parquet"
    if not preds_path.exists():
        logger.warning("Live loop: predictions missing; nothing to update")
        return {"n_matches": 0}

    preds = pd.read_parquet(preds_path)
    actuals = _load_actuals()

    # group_predictions carries one block of predictions per model; align actuals to
    # each so the live tracker can be viewed per model.
    if "model" in preds.columns:
        frames = []
        for name, sub in preds.groupby("model"):
            aligned = _align_to_prediction(actuals, sub)
            aligned.insert(0, "model", name)
            frames.append(aligned)
        live = (
            pd.concat(frames, ignore_index=True) if frames else _align_to_prediction(actuals, preds)
        )
    else:
        live = _align_to_prediction(actuals, preds)

    # Persist to the processed parquet and (if built) into the snapshot DB.
    live.to_parquet(processed / "live_predictions.parquet", index=False)
    db_path = settings.paths.snapshot_dir / DB_NAME
    if db_path.exists():
        con = sqlite3.connect(db_path)
        try:
            live.to_sql(_LIVE_TABLE, con, if_exists="replace", index=False)
            con.commit()
        finally:
            con.close()

    logger.info("Live loop: %d scored WC26 matches", len(live))
    return {"n_matches": int(len(live))}
