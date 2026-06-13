# Load engineered features to the warehouse.
#
# Two sinks:
#
# - **Parquet** under ``data/processed/`` - always written. It is the reliable,
#   dependency-free artifact the rest of the pipeline (and the snapshot export) reads.
# - **Postgres** - the local analytical warehouse. Written when a
#   database is reachable; if it is not (e.g. the Compose stack isn't up, or in a CI
#   job without a DB), the load is skipped with a warning rather than failing the
#   whole pipeline. Postgres is never a *runtime* dependency of the deployed app.

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from wc26.config import settings

logger = logging.getLogger(__name__)


def write_parquet(df: pd.DataFrame, name: str) -> Path:
    # Write ``df`` to ``data/processed/<name>.parquet`` and return the path.
    settings.paths.ensure()
    dest = settings.paths.processed_dir / f"{name}.parquet"
    df.to_parquet(dest, index=False)
    logger.info("Wrote %d rows -> %s", len(df), dest)
    return dest


def load_to_postgres(
    df: pd.DataFrame, table: str, *, if_exists: str = "replace", url: str | None = None
) -> bool:
    # Load ``df`` into warehouse table ``table``.
    #
    # ``url`` defaults to the configured Postgres URL; it is parameterised so tests
    # can target an in-memory SQLite engine. Returns True on success, False if the
    # database is unreachable (warned, not raised) so the pipeline can proceed on the
    # parquet artifact alone.
    try:
        engine = create_engine(url or settings.postgres_url)
        with engine.begin() as conn:
            df.to_sql(table, conn, if_exists=if_exists, index=False)
        logger.info("Loaded %d rows into Postgres table '%s'", len(df), table)
        return True
    except SQLAlchemyError as exc:
        logger.warning(
            "Postgres load skipped for '%s' (database unreachable: %s). "
            "Parquet artifact is still available.",
            table,
            exc.__class__.__name__,
        )
        return False
