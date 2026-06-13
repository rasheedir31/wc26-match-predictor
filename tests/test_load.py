# Tests for the load sinks (parquet always; warehouse tolerant of an absent DB).

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from wc26.config import settings
from wc26.etl.load import load_to_postgres, write_parquet


def _sqlite_url(tmp_path) -> str:
    return "sqlite:///" + str(tmp_path / "wh.db").replace("\\", "/")


def test_write_parquet_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings.paths, "processed_dir", tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = write_parquet(df, "thing")
    assert path.exists()
    assert pd.read_parquet(path).equals(df)


def test_load_to_warehouse_success(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    url = _sqlite_url(tmp_path)
    assert load_to_postgres(df, "feat", url=url) is True
    back = pd.read_sql("select * from feat", create_engine(url))
    assert len(back) == 3


def test_load_to_warehouse_unreachable_is_tolerated() -> None:
    # Nothing listens on port 1 -> connection refused -> warned, returns False.
    df = pd.DataFrame({"a": [1]})
    url = "postgresql+psycopg2://wc26:wc26@127.0.0.1:1/wc26"
    assert load_to_postgres(df, "feat", url=url) is False
