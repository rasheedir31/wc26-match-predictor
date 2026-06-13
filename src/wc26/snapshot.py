# Portable snapshot: build + load.
#
# The snapshot is the self-contained bundle the app serves at runtime - **no external
# database, no warehouse**. It is built from ``data/processed/`` by ``build_snapshot``
# and contains:
#
# - ``snapshot.db``  - a SQLite database with the small tabular outputs (tournament
#   odds, group-stage predictions, model comparison, groups, rankings, and an
#   initially-empty ``live_predictions`` table the monitor fills during the tournament).
# - ``*.joblib``     - the serialized champion model + the fitted featurizer.
# - ``calibration.json`` / ``champion_name.txt`` - small metadata.
#
# ``SnapshotStore`` loads it once into memory (the tables are tiny) and exposes the
# read accessors + an on-the-fly ``predict`` for arbitrary fixtures. Both the FastAPI
# API and the Streamlit dashboard read through it, so the app has a single data layer.

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wc26.config import settings
from wc26.features.featurizer import MatchupFeaturizer
from wc26.models.base import MatchPredictor

logger = logging.getLogger(__name__)

# Tables copied from processed parquet into the snapshot SQLite db.
_SNAPSHOT_TABLES = (
    "tournament_odds",
    "group_predictions",
    "model_comparison",
    "wc26_groups",
    "wc26_schedule",
    "fifa_rankings",
)

DB_NAME = "snapshot.db"
CHAMPION_MODEL = "champion_model.joblib"
FEATURIZER = "featurizer.joblib"
CALIBRATION = "calibration.json"
CHAMPION_NAME = "champion_name.txt"

# Schema for the live prediction-vs-actual log (filled by the monitor).
_LIVE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS live_predictions (
    model       TEXT,
    match_id    TEXT,
    date        TEXT,
    home_team   TEXT,
    away_team   TEXT,
    home_score  INTEGER,
    away_score  INTEGER,
    p_home      REAL,
    p_draw      REAL,
    p_away      REAL,
    actual      TEXT
)
"""


def build_snapshot() -> Path:
    # Assemble ``data/snapshot/`` from ``data/processed/``. Returns the snapshot dir.
    processed = settings.paths.processed_dir
    dst = settings.paths.snapshot_dir
    dst.mkdir(parents=True, exist_ok=True)

    # Serialized models + metadata.
    for name in (CHAMPION_MODEL, FEATURIZER, CALIBRATION, CHAMPION_NAME):
        src = processed / name
        if src.exists():
            shutil.copy2(src, dst / name)
        else:
            logger.warning("Snapshot: missing %s (run `make train` first)", src)
    # All per-model artifacts (model_<name>.joblib) so the app can switch models.
    for src in processed.glob("model_*.joblib"):
        shutil.copy2(src, dst / src.name)

    # Tabular outputs -> SQLite.
    db_path = dst / DB_NAME
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    try:
        for table in _SNAPSHOT_TABLES:
            pq = processed / f"{table}.parquet"
            if not pq.exists():
                logger.warning("Snapshot: missing table %s", pq)
                continue
            df = pd.read_parquet(pq)
            # SQLite has no native datetime; store dates as ISO strings.
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
            df.to_sql(table, con, if_exists="replace", index=False)
        con.execute(_LIVE_TABLE_DDL)
        con.commit()
    finally:
        con.close()

    logger.info("Built snapshot at %s", dst)
    return dst


class SnapshotStore:
    # In-memory read layer over a built snapshot (model + tables).

    def __init__(self, snapshot_dir: Path | None = None) -> None:
        self.dir = snapshot_dir or settings.paths.snapshot_dir
        self.model: MatchPredictor = joblib.load(self.dir / CHAMPION_MODEL)
        self.featurizer: MatchupFeaturizer = joblib.load(self.dir / FEATURIZER)
        self.champion_name = (self.dir / CHAMPION_NAME).read_text().strip()
        calib_path = self.dir / CALIBRATION
        self.calibration = json.loads(calib_path.read_text()) if calib_path.exists() else {}
        # Every model (model_<name>.joblib) so the app can switch between them.
        self.models: dict[str, MatchPredictor] = {
            p.stem.removeprefix("model_"): joblib.load(p)
            for p in sorted(self.dir.glob("model_*.joblib"))
        }
        if not self.models:  # older snapshot with only the champion
            self.models = {self.champion_name: self.model}
        self._tables = self._load_tables()

    def model_names(self) -> list[str]:
        # Available model names, champion first.
        others = sorted(n for n in self.models if n != self.champion_name)
        return ([self.champion_name] if self.champion_name in self.models else []) + others

    def _resolve(self, model: str | None) -> MatchPredictor:
        return self.models.get(model or self.champion_name, self.model)

    @staticmethod
    def _filter_model(df: pd.DataFrame, model: str | None) -> pd.DataFrame:
        # Filter a frame to one model when it carries a ``model`` column.
        if model is not None and "model" in df.columns:
            df = df[df["model"] == model]
        return df.drop(columns=["model"], errors="ignore").reset_index(drop=True)

    def _load_tables(self) -> dict[str, pd.DataFrame]:
        con = sqlite3.connect(self.dir / DB_NAME)
        try:
            names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)["name"]
            return {n: pd.read_sql(f"SELECT * FROM {n}", con) for n in names}
        finally:
            con.close()

    def table(self, name: str) -> pd.DataFrame:
        return self._tables.get(name, pd.DataFrame()).copy()

    # --- accessors -------------------------------------------------------------

    def tournament_odds(self, model: str | None = None) -> pd.DataFrame:
        return self._filter_model(self.table("tournament_odds"), model)

    def group_predictions(self, model: str | None = None) -> pd.DataFrame:
        return self._filter_model(self.table("group_predictions"), model)

    def model_comparison(self) -> pd.DataFrame:
        return self.table("model_comparison")

    def live_predictions(self, model: str | None = None) -> pd.DataFrame:
        # The per-match live log (prediction vs actual); empty until games are played.
        return self._filter_model(self.table("live_predictions"), model)

    def groups(self) -> pd.DataFrame:
        return self.table("wc26_groups")

    def teams(self) -> list[str]:
        g = self.groups()
        return sorted(g["team"].tolist()) if not g.empty else []

    def predict(
        self, home: str, away: str, *, neutral: bool = True, model: str | None = None
    ) -> dict[str, float | str]:
        # 1X2 prediction for an arbitrary fixture (champion model unless ``model`` given).
        feat = self.featurizer.fixture_features(home, away, neutral=neutral)
        proba = self._resolve(model).predict_proba(feat)[0]
        labels = ("home", "draw", "away")
        return {
            "p_home": float(proba[0]),
            "p_draw": float(proba[1]),
            "p_away": float(proba[2]),
            "most_likely": labels[int(proba.argmax())],
        }

    def live_summary(self, model: str | None = None) -> dict[str, float | int]:
        # Running prediction-vs-actual accuracy / log loss over scored WC26 matches.
        #
        # Empty until the tournament starts and the monitor fills ``live_predictions``.
        live = self.live_predictions(model)
        scored = live.dropna(subset=["actual"]) if not live.empty else live
        if scored.empty:
            return {"n_matches": 0, "accuracy": 0.0, "log_loss": 0.0}

        proba = scored[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        idx = {"H": 0, "D": 1, "A": 2}
        y = scored["actual"].map(idx).to_numpy()
        picked = proba.argmax(axis=1)
        acc = float((picked == y).mean())
        ll = float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0)).mean())
        return {"n_matches": int(len(scored)), "accuracy": acc, "log_loss": ll}


@lru_cache
def get_store() -> SnapshotStore:
    # Cached snapshot store (loaded once per process).
    return SnapshotStore()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build_snapshot()


if __name__ == "__main__":
    main()
