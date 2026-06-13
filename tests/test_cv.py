# Time-based CV must respect chronology (no future leakage).

from __future__ import annotations

from wc26 import schema
from wc26.evaluate.cv import time_series_folds


def test_folds_are_chronological_and_disjoint(feature_frame) -> None:
    folds = list(time_series_folds(feature_frame, n_splits=4))
    assert len(folds) == 4
    for train, test in folds:
        # Every training match is strictly no later than every test match.
        assert train[schema.COL_DATE].max() <= test[schema.COL_DATE].min()
        # Train grows (expanding window) and test is non-empty.
        assert len(train) > 0 and len(test) > 0


def test_folds_expanding_window(feature_frame) -> None:
    sizes = [len(train) for train, _ in time_series_folds(feature_frame, n_splits=4)]
    assert sizes == sorted(sizes)  # training set never shrinks
