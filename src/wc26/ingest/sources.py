# Source loaders: return raw (unvalidated) DataFrames for each input.
#
# Each loader tries the cached download first and falls back to committed seed data
# when the source is unreachable. Validation and typing happen downstream in
# ``wc26.etl.validate`` - these functions only locate and read bytes.

from __future__ import annotations

import logging

import pandas as pd

from wc26.config import settings
from wc26.ingest.download import download_to_cache

logger = logging.getLogger(__name__)


def _read_csv_with_fallback(
    url: str,
    cache_name: str,
    seed_name: str,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, str]:
    # Download+read ``url``; on failure read ``data/seed/<seed_name>``.
    #
    # Returns the frame and a provenance tag (``"download"`` or ``"seed"``) so the
    # pipeline can report whether it ran on real or fallback data.
    path = download_to_cache(url, cache_name, force=force)
    if path is not None:
        return pd.read_csv(path), "download"

    seed_path = settings.paths.seed_dir / seed_name
    logger.warning("Falling back to seed data: %s", seed_path)
    return pd.read_csv(seed_path), "seed"


def load_results(*, force: bool = False) -> pd.DataFrame:
    # International match results (martj42 schema). Real source or sample seed.
    df, source = _read_csv_with_fallback(
        settings.sources.results_url,
        cache_name="results.csv",
        seed_name="sample_results.csv",
        force=force,
    )
    logger.info("Loaded %d results rows (source=%s)", len(df), source)
    return df


def load_shootouts(*, force: bool = False) -> pd.DataFrame:
    # Penalty-shootout outcomes for drawn knockout matches (martj42 schema).
    df, source = _read_csv_with_fallback(
        settings.sources.shootouts_url,
        cache_name="shootouts.csv",
        seed_name="sample_shootouts.csv",
        force=force,
    )
    logger.info("Loaded %d shootout rows (source=%s)", len(df), source)
    return df


def load_fifa_rankings(*, force: bool = False) -> pd.DataFrame:
    # FIFA ranking snapshot. Reference data (priors / display), not a model feature.
    #
    # Uses the configured URL when set, otherwise the committed seed snapshot.
    df, source = _read_csv_with_fallback(
        settings.sources.fifa_rankings_url,
        cache_name="fifa_rankings.csv",
        seed_name="fifa_rankings.csv",
        force=force,
    )
    logger.info("Loaded %d FIFA-ranking rows (source=%s)", len(df), source)
    return df
