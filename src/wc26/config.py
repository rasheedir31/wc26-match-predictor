# Central configuration via pydantic-settings.
#
# Single source of truth for every path, URL, and hyperparameter in the project.
# see the README conventions: no hardcoded paths, URLs, or magic numbers scattered
# through the codebase - they live here and are read from the environment / `.env`.
#
# Usage::
#
#     from wc26.config import settings
#
#     settings.postgres_url  # warehouse connection
#     settings.paths.raw_dir  # where ingest caches downloads
#     settings.model.elo_k  # Elo K-factor

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/wc26/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseSettings):
    # Filesystem layout.
    #
    # Sub-directories default to ``data_dir``-relative locations but are each
    # individually overridable (by env ``WC26_PATH_*`` or by construction) - e.g. a
    # test can redirect the output dirs to a temp folder while keeping ``seed_dir``
    # on the committed seeds. Anything but ``seed_dir`` is gitignored and reproducible.

    model_config = SettingsConfigDict(env_prefix="WC26_PATH_", extra="ignore")

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    # Left as None so the validator can derive them from data_dir when not set.
    seed_dir: Path | None = None  # committed reference + offline fallbacks
    raw_dir: Path | None = None  # cached raw downloads
    interim_dir: Path | None = None  # intermediate ETL frames
    processed_dir: Path | None = None  # engineered tables ready for the warehouse
    snapshot_dir: Path | None = None  # portable app bundle

    @model_validator(mode="after")
    def _derive_subdirs(self) -> Paths:
        # Seeds live with the project root's data dir by default (committed, not
        # under a redirected/temp data_dir) unless explicitly overridden.
        if self.seed_dir is None:
            self.seed_dir = PROJECT_ROOT / "data" / "seed"
        if self.raw_dir is None:
            self.raw_dir = self.data_dir / "raw"
        if self.interim_dir is None:
            self.interim_dir = self.data_dir / "interim"
        if self.processed_dir is None:
            self.processed_dir = self.data_dir / "processed"
        if self.snapshot_dir is None:
            self.snapshot_dir = self.data_dir / "snapshot"
        return self

    def ensure(self) -> None:
        # Create the writable data directories if they do not yet exist.
        #
        # ``seed_dir`` is intentionally excluded - it is committed, not generated.
        for d in (
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.snapshot_dir,
        ):
            assert d is not None  # populated by the validator
            d.mkdir(parents=True, exist_ok=True)


class DataSources(BaseSettings):
    # Upstream data source URLs (overridable for testing / mirrors).

    model_config = SettingsConfigDict(env_prefix="WC26_SOURCE_", extra="ignore")

    # martj42/international_results - international match results CSV.
    results_url: str = (
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    )
    shootouts_url: str = (
        "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
    )
    # FIFA rankings and WC26 fixtures are wired up.
    fifa_rankings_url: str = ""
    wc26_fixtures_url: str = ""


class ModelParams(BaseSettings):
    # Model and simulation hyperparameters. Deterministic seeds live here too.

    model_config = SettingsConfigDict(env_prefix="WC26_MODEL_", extra="ignore")

    random_seed: int = 42

    # Elo
    elo_k: float = 32.0
    elo_home_advantage: float = 65.0
    elo_initial_rating: float = 1500.0

    # Dixon-Coles
    # Exponential time-decay half-life in days for match weighting.
    dc_time_decay_half_life_days: float = 365.0 * 2
    # Ridge (L2) shrinkage on attack/defense - regularises low-data teams and
    # resolves the attack/defense level redundancy (identifiability).
    dc_ridge: float = 0.01
    # Max goals per side in the score-matrix used for prediction.
    dc_max_goals: int = 10

    # Monte Carlo tournament simulation
    simulation_runs: int = 10_000
    # Knockout draws are broken by extra-time/penalties; bias the coin toward the
    # stronger side by this fraction of its regulation win-probability edge.
    knockout_strength_bias: float = 0.5


class FeatureParams(BaseSettings):
    # Feature-engineering windows and fill values. No magic numbers in code.

    model_config = SettingsConfigDict(env_prefix="WC26_FEATURE_", extra="ignore")

    # Rolling window (number of recent matches) for form and goal-difference.
    form_window: int = 5
    # Maximum prior meetings counted for the head-to-head feature.
    h2h_max_matches: int = 10
    # Cap on rest-days (matches after a long gap / a team's first match).
    rest_days_cap: int = 30
    # Points awarded for the form feature (avg points-per-game over the window).
    points_win: float = 3.0
    points_draw: float = 1.0
    points_loss: float = 0.0


class EvalParams(BaseSettings):
    # Evaluation / time-based CV settings.

    model_config = SettingsConfigDict(env_prefix="WC26_EVAL_", extra="ignore")

    # Number of expanding-window time-based CV folds.
    n_time_splits: int = 5
    # Reliability-curve bins for the calibration check.
    calibration_bins: int = 10


class Settings(BaseSettings):
    # Top-level settings aggregating all config groups.

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Warehouse (local Postgres; never deployed - see the README guardrails) ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "wc26"
    postgres_password: str = "wc26"
    postgres_db: str = "wc26"

    # --- Experiment tracking ---
    mlflow_tracking_uri: str = Field(default="file:./mlruns")
    mlflow_experiment_name: str = "wc26"

    # --- App (FastAPI + Streamlit) ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501

    # --- football-data.org (optional live-results source for the monitor) ---
    # Secret: set FOOTBALL_DATA_TOKEN in .env (gitignored). When empty, the live
    # loop falls back to the martj42 dataset. Competition WC = FIFA World Cup
    # (included in football-data.org's free tier).
    football_data_token: str = ""
    football_data_base_url: str = "https://api.football-data.org/v4"
    football_data_competition: str = "WC"

    # --- Grouped config ---
    paths: Paths = Field(default_factory=Paths)
    sources: DataSources = Field(default_factory=DataSources)
    model: ModelParams = Field(default_factory=ModelParams)
    features: FeatureParams = Field(default_factory=FeatureParams)
    eval: EvalParams = Field(default_factory=EvalParams)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def postgres_url(self) -> str:
        # SQLAlchemy URL for the local warehouse.
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    # Return a cached Settings instance (read env / .env once per process).
    return Settings()


# Convenient module-level handle.
settings = get_settings()
