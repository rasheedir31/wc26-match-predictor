# smoke tests: the package imports and config loads with sane defaults.
#
# Real feature/model/eval/simulation tests arrive in Phases 1-2.

from __future__ import annotations

import wc26
from wc26.config import get_settings, settings


def test_package_version() -> None:
    assert wc26.__version__ == "0.1.0"


def test_settings_load() -> None:
    s = get_settings()
    # Defaults from config.py.
    assert s.model.random_seed == 42
    assert s.model.simulation_runs >= 10_000
    assert s.eval.n_time_splits >= 1


def test_settings_is_cached() -> None:
    # get_settings is lru_cached and the module handle is the same instance.
    assert get_settings() is settings


def test_postgres_url_shape() -> None:
    url = settings.postgres_url
    assert url.startswith("postgresql+psycopg2://")
    assert settings.postgres_db in url


def test_paths_are_under_project_root() -> None:
    p = settings.paths
    assert p.raw_dir.parent == p.data_dir
    assert p.snapshot_dir.name == "snapshot"
