# Model contract tests: every model returns valid (n, 3) probabilities that sum to 1.

from __future__ import annotations

import numpy as np
import pytest

from wc26.models.registry import default_models

MODEL_NAMES = list(default_models())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predict_proba_contract(name, feature_frame) -> None:
    factory = default_models()[name]
    model = factory().fit(feature_frame)
    proba = model.predict_proba(feature_frame)

    assert proba.shape == (len(feature_frame), 3)
    assert np.all(proba >= 0.0)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predict_frame_columns(name, feature_frame) -> None:
    model = default_models()[name]().fit(feature_frame)
    frame = model.predict_frame(feature_frame)
    assert list(frame.columns) == ["p_home", "p_draw", "p_away"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_is_deterministic(name, feature_frame) -> None:
    a = default_models()[name]().fit(feature_frame).predict_proba(feature_frame)
    b = default_models()[name]().fit(feature_frame).predict_proba(feature_frame)
    assert np.allclose(a, b)
