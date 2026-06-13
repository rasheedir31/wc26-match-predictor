# Time-based cross-validation splitter.
#
# **Time-based only - never shuffle.** Matches are sorted by date and split with an
# expanding window: each fold trains on an initial block of past matches and tests on
# the next contiguous block of future matches. This mirrors reality (predict forward)
# and prevents leakage that would invalidate every metric.

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from wc26 import schema
from wc26.config import settings


def time_series_folds(
    df: pd.DataFrame, n_splits: int | None = None
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    # Yield (train, test) frames for each expanding-window time fold.
    #
    # The frame is sorted by (date, match_id) first, so splits respect chronology
    # regardless of the input order.
    n_splits = settings.eval.n_time_splits if n_splits is None else n_splits
    ordered = df.sort_values([schema.COL_DATE, schema.COL_MATCH_ID]).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(ordered):
        yield ordered.iloc[train_idx], ordered.iloc[test_idx]
